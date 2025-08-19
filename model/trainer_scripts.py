from __future__ import annotations

import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import shap
from loguru import logger

from model.trainer import TrainConfig, UnifiedTrainer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

out_dir = "data"

# ---------------------------
# Utilities
# ---------------------------

def _ensure_dirs(paths: List[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _sanitize(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _impact(val: float) -> str:
    # Your mapping: <0 -> positive, >0 -> negative
    return "positive" if val < 0 else "negative" if val > 0 else "neutral"


# --- LGBM/SHAP helpers (robust for per-query & tiny models) ---

def _lgbm_feature_names(est) -> Optional[List[str]]:
    """Return LightGBM booster feature names if available."""
    try:
        return list(est.booster_.feature_name())
    except Exception:
        return None


def _align_X_to_model_features(X, est):
    """
    Align X columns to the model's feature names.
    If names exist and shapes match, reindex/construct DataFrame accordingly.
    """
    names = _lgbm_feature_names(est)
    if names is None:
        return X, None

    try:
        n_features = X.shape[1]
    except Exception:
        n_features = None

    if n_features == len(names) and not isinstance(X, pd.DataFrame):
        try:
            X = pd.DataFrame(X.toarray() if hasattr(X, "toarray") else X, columns=names)
        except Exception:
            pass

    if isinstance(X, pd.DataFrame):
        X = X.reindex(columns=names, fill_value=0)

    return X, names


def _num_trees(est) -> int:
    """Return number of trees; 0 means booster effectively empty."""
    try:
        return int(est.booster_.num_trees())
    except Exception:
        return -1  # unknown


def _safe_mean_shap_values_est_X(est, X) -> np.ndarray:
    """
    Compute mean SHAP values robustly.
    - Align X to model feature names
    - Dense fallback for sparse
    - Handle 0-tree models
    - Normalize list-of-arrays case
    """
    X, names = _align_X_to_model_features(X, est)
    nt = _num_trees(est)
    if nt == 0:
        n_features = X.shape[1] if hasattr(X, "shape") else (len(names) if names else 0)
        return np.zeros(n_features, dtype=float)

    try:
        expl = shap.TreeExplainer(est)
        shap_vals = expl.shap_values(X)
    except Exception:
        # Try dense input
        if hasattr(X, "toarray"):
            X = X.toarray()
        elif isinstance(X, pd.DataFrame):
            X = X.values
        expl = shap.TreeExplainer(est)
        shap_vals = expl.shap_values(X)

    if isinstance(shap_vals, list):
        if len(shap_vals) == 0:
            return np.zeros(X.shape[1], dtype=float)
        arr = np.asarray([np.asarray(sv) for sv in shap_vals])  # (classes, n, f)
        vals = arr.mean(axis=0)  # (n, f)
    else:
        vals = np.asarray(shap_vals)  # (n, f)
        if vals.ndim == 1:
            vals = np.expand_dims(vals, 1)
        if vals.size == 0:
            return np.zeros(X.shape[1], dtype=float)

    return vals.mean(axis=0)  # (f,)


# --- Feature builder for SHAP (tabular only) ---

def _build_tabular_like_trainer(
    df: pd.DataFrame,
    drop_cols: List[str],
    target_col: str
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    df = df.copy()
    df = df[df[target_col].notnull()]
    y = np.log1p(df[target_col].astype(float).values)
    feat_cols = [c for c in df.columns if c not in set(drop_cols) | {target_col}]
    X = df[feat_cols].copy()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # label-encode like UnifiedTrainer
    from sklearn.preprocessing import LabelEncoder
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X.columns = (
        pd.Index(X.columns)
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    if cat_cols:
        X[cat_cols] = X[cat_cols].astype("category")
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
    return X, y, cat_cols


# --- Robust feature/term column detection ---

def _detect_feat_col(df: pd.DataFrame) -> str:
    """Find the name of the feature/term column in importance CSVs."""
    for c in ["feature", "term", "token", "ngram", "word", "feature_name", "name"]:
        if c in df.columns:
            return c
    return df.columns[0]


def _ensure_feat_col_named_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where the feature column is renamed to 'feature'."""
    feat_col = _detect_feat_col(df)
    if feat_col != "feature":
        df = df.rename(columns={feat_col: "feature"})
    return df


# --- Helpers for TEXT models: pull TF-IDF from trained pipeline ---

def _find_tfidf_vectorizer(obj) -> Optional[TfidfVectorizer]:
    """Recursively search for a fitted TfidfVectorizer inside a model/pipeline."""
    if isinstance(obj, TfidfVectorizer):
        return obj
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            v = _find_tfidf_vectorizer(step)
            if v is not None:
                return v
    if isinstance(obj, ColumnTransformer):
        if hasattr(obj, "transformers_"):
            for _, trans, _ in obj.transformers_:
                v = _find_tfidf_vectorizer(trans)
                if v is not None:
                    return v
    # Common custom attributes some wrappers expose
    for attr in ("vectorizer_", "tfidf_", "tfidf"):
        if hasattr(obj, attr):
            maybe = getattr(obj, attr)
            v = _find_tfidf_vectorizer(maybe)
            if v is not None:
                return v
    return None


def _find_final_estimator(obj):
    """Get the final estimator (tree model) from a pipeline or return the obj itself."""
    if isinstance(obj, Pipeline):
        return _find_final_estimator(obj.steps[-1][1])
    return obj


def _concat_text(df: pd.DataFrame, text_fields: List[str]) -> pd.Series:
    if len(text_fields) == 1:
        return df[text_fields[0]].fillna("").astype(str)
    return df[text_fields].fillna("").astype(str).agg(" ".join, axis=1)


def _compute_text_shap_and_merge(
    df: pd.DataFrame,
    text_fields: List[str],
    model_obj,
    imp_df: pd.DataFrame,
    out_csv_path: str,
    shap_sample: int = 4000
) -> None:
    """
    Extract TF-IDF from fitted model pipeline, transform df[text_fields],
    compute mean SHAP per term, merge into importance CSV and save.
    """
    imp = _ensure_feat_col_named_feature(imp_df.copy())
    vec = _find_tfidf_vectorizer(model_obj)

    if vec is None:
        logger.warning("TF-IDF vectorizer not found; writing neutral impact: {}", out_csv_path)
        if "impact_shap" not in imp.columns:
            imp["mean_shap"] = np.nan
            imp["impact_shap"] = "neutral"
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        imp.to_csv(out_csv_path, index=False)
        return

    est = _find_final_estimator(model_obj)
    corpus = _concat_text(df, text_fields)
    if shap_sample and len(corpus) > shap_sample:
        corpus = corpus.sample(shap_sample, random_state=17)

    X = vec.transform(corpus)  # sparse ok with LGBM+TreeExplainer
    feature_names = vec.get_feature_names_out()

    # Guard: 0-tree tiny models
    if _num_trees(est) == 0:
        logger.warning("0-tree model for {}; writing zeros.", out_csv_path)
        mean_shap = np.zeros(len(feature_names), dtype=float)
    else:
        try:
            mean_shap = _safe_mean_shap_values_est_X(est, X)
        except Exception as e:
            logger.warning("Tree SHAP failed for text model ({}): {}. Falling back to zeros.", out_csv_path, e)
            mean_shap = np.zeros(X.shape[1], dtype=float)

    shap_df = pd.DataFrame({"feature": feature_names, "mean_shap": mean_shap})
    imp = imp.merge(shap_df, on="feature", how="left")
    imp["impact_shap"] = imp["mean_shap"].apply(lambda v: _impact(v) if pd.notnull(v) else "neutral")

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    imp.to_csv(out_csv_path, index=False)


# ---------------------------
# Model 1 — products_features (tabular)
# ---------------------------

def run_model1_products_features_global(df, save_dir="study/global_products_features") -> None:
    model_path = f"{out_dir}/{save_dir}/models/lgbm_model.pkl"
    importance_path = f"{out_dir}/{save_dir}/importance/features.csv"
    summary_path = f"{out_dir}/{save_dir}/summary.csv"

    if df.empty:
        logger.error(f"Missing df for {save_dir}")
        return

    _ensure_dirs([f"{out_dir}/{save_dir}/models", f"{out_dir}/{save_dir}/importance"])

    drop_cols = [
        "productId", "title", "shortTitle", "description", "shortDescription",
        "title_review", "reviewText", "language", "releaseDateUtc", "lastUpdateDateUtc",
    ]

    cfg = TrainConfig(target_col="position", drop_cols=drop_cols, text_fields=None, optuna_trials=30)
    trainer = UnifiedTrainer(cfg)
    result = trainer.fit(df)

    model = result.models["__global__"]
    joblib.dump(model, model_path)

    imp = _ensure_feat_col_named_feature(result.importances["__global__"].copy())
    X_all, _, _ = _build_tabular_like_trainer(df, drop_cols, target_col="position")
    mean_shap = _safe_mean_shap_values_est_X(model, X_all)

    names = _lgbm_feature_names(model)
    shap_df = pd.DataFrame({
        "feature": names if names else list(X_all.columns),
        "mean_shap": mean_shap
    })
    imp = imp.merge(shap_df, on="feature", how="left")
    imp["impact_shap"] = imp["mean_shap"].apply(_impact)
    imp.to_csv(importance_path, index=False)

    result.summary[["rmse"]].to_csv(summary_path, index=False)
    logger.success("Saved model, importance(+impact_shap), summary for Model1 Global")


def run_model1_products_features_per_query(df, save_dir="study/query_products_features") -> None:
    base_models = f"{out_dir}/{save_dir}/models/"
    base_imps = f"{out_dir}/{save_dir}/importance"
    summary_path = f"{out_dir}/{save_dir}/summary.csv"

    if df.empty:
        logger.error(f"Missing df for {save_dir}")
        return

    _ensure_dirs([f"{out_dir}/{save_dir}/models", f"{out_dir}/{save_dir}/importance"])

    drop_cols = [
        "query", "productId", "title", "shortTitle", "description", "shortDescription",
        "title_review", "reviewText", "language", "releaseDateUtc", "lastUpdateDateUtc",
    ]

    cfg = TrainConfig(target_col="position", drop_cols=drop_cols, text_fields=None, min_samples_per_group=10)
    trainer = UnifiedTrainer(cfg)
    result = trainer.fit(df, group_col="query")

    for g, model in result.models.items():
        filename_base = _sanitize(g)
        model_path = os.path.join(base_models, f"{filename_base}.pkl")
        imp_path = os.path.join(base_imps, f"{filename_base}.csv")
        joblib.dump(model, model_path)

        df_g = df[df["query"] == g]
        X_g, _, _ = _build_tabular_like_trainer(df_g, drop_cols, target_col="position")
        mean_shap = _safe_mean_shap_values_est_X(model, X_g)

        names = _lgbm_feature_names(model)
        shap_df = pd.DataFrame({
            "feature": names if names else (list(X_g.columns) if isinstance(X_g, pd.DataFrame) else list(range(len(mean_shap)))),
            "mean_shap": mean_shap
        })

        imp = _ensure_feat_col_named_feature(result.importances[g].copy())
        imp = imp.merge(shap_df, on="feature", how="left")
        if imp["mean_shap"].isna().all():
            logger.warning("[per_query:{}] No SHAP-feature overlap. Check feature naming / dropped constants.", g)
        imp["impact_shap"] = imp["mean_shap"].apply(lambda v: _impact(v) if pd.notnull(v) else "neutral")
        imp.to_csv(imp_path, index=False)

    summary = result.summary.rename(columns={"group": "query"})
    summary[["query", "samples", "rmse"]].to_csv(summary_path, index=False)
    logger.success("Saved per-query artifacts for Model1")


# ---------------------------
# Model 2 — reviews_features (tabular)
# ---------------------------

def run_model2_reviews_features_global(df, save_dir="study/global_reviews_features") -> None:
    model_path = f"{out_dir}/{save_dir}/models/lgbm_model.pkl"
    importance_path = f"{out_dir}/{save_dir}/importance/features.csv"
    summary_path = f"{out_dir}/{save_dir}/summary.csv"

    if df.empty:
        logger.error(f"Missing df for {save_dir}")
        return

    _ensure_dirs([f"{out_dir}/{save_dir}/models", f"{out_dir}/{save_dir}/importance"])


    drop_cols = [
        "productId", "position",
        "title", "shortTitle", "description", "shortDescription",
        "title_review", "reviewText",
    ]

    cfg = TrainConfig(target_col="position", drop_cols=drop_cols, text_fields=None)
    trainer = UnifiedTrainer(cfg)
    result = trainer.fit(df)

    model = result.models["__global__"]
    joblib.dump(model, model_path)

    imp = _ensure_feat_col_named_feature(result.importances["__global__"].copy())
    X_all, _, _ = _build_tabular_like_trainer(df, drop_cols, target_col="position")
    mean_shap = _safe_mean_shap_values_est_X(model, X_all)

    names = _lgbm_feature_names(model)
    shap_df = pd.DataFrame({
        "feature": names if names else list(X_all.columns),
        "mean_shap": mean_shap
    })
    imp = imp.merge(shap_df, on="feature", how="left")
    imp["impact_shap"] = imp["mean_shap"].apply(_impact)
    imp.to_csv(importance_path, index=False)

    result.summary[["rmse"]].to_csv(summary_path, index=False)
    logger.success("Saved model, importance(+impact_shap), summary for Model2 Global")


def run_model2_reviews_features_per_query(df, save_dir="study/query_reviews_features") -> None:
    base_models = f"{out_dir}/{save_dir}/models/"
    base_imps = f"{out_dir}/{save_dir}/importance"
    summary_path = f"{out_dir}/{save_dir}/summary.csv"

    if df.empty:
        logger.error(f"Missing df for {save_dir}")
        return

    _ensure_dirs([f"{out_dir}/{save_dir}/models", f"{out_dir}/{save_dir}/importance"])

    drop_cols = [
        "productId", "position", "query",
        "title", "shortTitle", "description", "shortDescription",
        "title_review", "reviewText",
    ]

    cfg = TrainConfig(target_col="position", drop_cols=drop_cols, text_fields=None, min_samples_per_group=10)
    trainer = UnifiedTrainer(cfg)
    result = trainer.fit(df, group_col="query")

    for g, model in result.models.items():
        filename_base = _sanitize(g)
        model_path = os.path.join(base_models, f"{filename_base}.pkl")
        imp_path = os.path.join(base_imps, f"{filename_base}.csv")
        joblib.dump(model, model_path)

        df_g = df[df["query"] == g]
        X_g, _, _ = _build_tabular_like_trainer(df_g, drop_cols, target_col="position")
        mean_shap = _safe_mean_shap_values_est_X(model, X_g)

        names = _lgbm_feature_names(model)
        shap_df = pd.DataFrame({
            "feature": names if names else (list(X_g.columns) if isinstance(X_g, pd.DataFrame) else list(range(len(mean_shap)))),
            "mean_shap": mean_shap
        })

        imp = _ensure_feat_col_named_feature(result.importances[g].copy())
        imp = imp.merge(shap_df, on="feature", how="left")
        if imp["mean_shap"].isna().all():
            logger.warning("[per_query:{}] No SHAP-feature overlap. Check feature naming / dropped constants.", g)
        imp["impact_shap"] = imp["mean_shap"].apply(lambda v: _impact(v) if pd.notnull(v) else "neutral")
        imp.to_csv(imp_path, index=False)

    summary = result.summary.rename(columns={"group": "query"})
    summary[["query", "samples", "rmse"]].to_csv(summary_path, index=False)
    logger.success("Saved per-query artifacts for Model2")


# ---------------------------
# Model 3 — products_texts (text + SHAP)
# ---------------------------

def run_model3_products_texts_global(df, save_dir="study/global_products_texts") -> None:
    model_path = f"{out_dir}/{save_dir}/models/lgbm_model.pkl"
    importance_path = f"{out_dir}/{save_dir}/importance/features.csv"
    summary_path = f"{out_dir}/{save_dir}/summary.csv"

    if df.empty:
        logger.error(f"Missing df for {save_dir}")
        return

    _ensure_dirs([f"{out_dir}/{save_dir}/models", f"{out_dir}/{save_dir}/importance"])


    df = df[df["position"].notnull()]

    text_fields = ["title", "shortTitle", "description", "shortDescription"]
    cfg = TrainConfig(target_col="position", text_fields=text_fields, tfidf_max_features=2000, tfidf_min_df=5, tfidf_ngram_range=(1, 1))
    trainer = UnifiedTrainer(cfg)
    result = trainer.fit(df)

    model_obj = result.models["__global__"]
    joblib.dump(model_obj, model_path)

    imp_df = result.importances["__global__"].copy()
    _compute_text_shap_and_merge(
        df=df,
        text_fields=text_fields,
        model_obj=model_obj,
        imp_df=imp_df,
        out_csv_path=importance_path,
        shap_sample=4000,
    )

    result.summary[["rmse"]].to_csv(summary_path, index=False)
    logger.success("Saved model, importance(+impact_shap), summary for Model3 Global")


def run_model3_products_texts_per_query(df, save_dir="study/query_products_texts") -> None:
    base_models = f"{out_dir}/{save_dir}/models/"
    base_imps = f"{out_dir}/{save_dir}/importance"
    summary_path = f"{out_dir}/{save_dir}/summary.csv"

    if df.empty:
        logger.error(f"Missing df for {save_dir}")
        return

    _ensure_dirs([f"{out_dir}/{save_dir}/models", f"{out_dir}/{save_dir}/importance"])

    df = df[df["position"].notnull()]

    text_fields = ["title", "shortTitle", "description", "shortDescription"]
    cfg = TrainConfig(target_col="position", text_fields=text_fields, tfidf_max_features=2000, tfidf_min_df=2, tfidf_ngram_range=(1, 1), min_samples_per_group=3)
    trainer = UnifiedTrainer(cfg)
    result = trainer.fit(df, group_col="query")

    for g, model_obj in result.models.items():
        filename_base = _sanitize(g)
        joblib.dump(model_obj, os.path.join(base_models, f"{filename_base}.pkl"))

        imp_df = result.importances[g].copy()
        df_g = df[df["query"] == g]
        _compute_text_shap_and_merge(
            df=df_g,
            text_fields=text_fields,
            model_obj=model_obj,
            imp_df=imp_df,
            out_csv_path=os.path.join(base_imps, f"{filename_base}.csv"),
            shap_sample=4000,
        )

    summary = result.summary.rename(columns={"group": "query"})
    summary[["query", "samples", "rmse"]].to_csv(summary_path, index=False)
    logger.success("Saved per-query artifacts for Model3")


# ---------------------------
# Model 4 — reviews_texts (text + SHAP)
# ---------------------------

def run_model4_reviews_texts_global(df, save_dir="study/global_reviews_texts") -> None:
    model_path = f"{out_dir}/{save_dir}/models/lgbm_model.pkl"
    importance_path = f"{out_dir}/{save_dir}/importance/features.csv"
    summary_path = f"{out_dir}/{save_dir}/summary.csv"

    if df.empty:
        logger.error(f"Missing df for {save_dir}")
        return

    _ensure_dirs([f"{out_dir}/{save_dir}/models", f"{out_dir}/{save_dir}/importance"])

    df = df[df["position"].notnull() & df["reviewText"].notnull()]

    text_fields = ["title", "reviewText"]
    cfg = TrainConfig(target_col="position", text_fields=text_fields, tfidf_max_features=200, tfidf_min_df=5, tfidf_ngram_range=(1, 1))
    trainer = UnifiedTrainer(cfg)
    result = trainer.fit(df)

    model_obj = result.models["__global__"]
    joblib.dump(model_obj, model_path)

    imp_df = result.importances["__global__"].copy()
    _compute_text_shap_and_merge(
        df=df,
        text_fields=text_fields,
        model_obj=model_obj,
        imp_df=imp_df,
        out_csv_path=importance_path,
        shap_sample=4000,
    )

    result.summary[["rmse"]].to_csv(summary_path, index=False)
    logger.success("Saved model, importance(+impact_shap), summary for Model4 Global")


def run_model4_reviews_texts_per_query(df, save_dir="study/query_reviews_texts") -> None:
    base_models = f"{out_dir}/{save_dir}/models/"
    base_imps = f"{out_dir}/{save_dir}/importance"
    summary_path = f"{out_dir}/{save_dir}/summary.csv"

    if df.empty:
        logger.error(f"Missing df for {save_dir}")
        return

    _ensure_dirs([f"{out_dir}/{save_dir}/models", f"{out_dir}/{save_dir}/importance"])

    df = df[df["position"].notnull() & df["reviewText"].notnull()]

    text_fields = ["title", "reviewText"]
    cfg = TrainConfig(target_col="position", text_fields=text_fields, tfidf_max_features=2000, tfidf_min_df=5, tfidf_ngram_range=(1, 2), min_samples_per_group=3)
    trainer = UnifiedTrainer(cfg)
    result = trainer.fit(df, group_col="query")

    for g, model_obj in result.models.items():
        filename_base = _sanitize(g)
        joblib.dump(model_obj, os.path.join(base_models, f"{filename_base}.pkl"))

        imp_df = result.importances[g].copy()
        df_g = df[df["query"] == g]
        _compute_text_shap_and_merge(
            df=df_g,
            text_fields=text_fields,
            model_obj=model_obj,
            imp_df=imp_df,
            out_csv_path=os.path.join(base_imps, f"{filename_base}.csv"),
            shap_sample=4000,
        )

    summary = result.summary.rename(columns={"group": "query"})
    summary[["query", "samples", "rmse"]].to_csv(summary_path, index=False)
    logger.success("Saved per-query artifacts for Model4")


# ---------------------------
# MAIN
# ---------------------------

def run(db_data: dict):
    products_features_df = db_data["products_features"]
    reviews_features_df = db_data["reviews_features"]
    products_texts_df = db_data["products_texts"]
    reviews_texts_df = db_data["reviews_texts"]

    run_model1_products_features_global(products_features_df)
    run_model2_reviews_features_global(reviews_features_df)
    run_model3_products_texts_global(products_texts_df)
    run_model4_reviews_texts_global(reviews_texts_df)

    # run_model1_products_features_per_query(products_features_df)
    # run_model2_reviews_features_per_query(reviews_features_df)
    # run_model3_products_texts_per_query(products_texts_df)
    # run_model4_reviews_texts_per_query(reviews_texts_df)
