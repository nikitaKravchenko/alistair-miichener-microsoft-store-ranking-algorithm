"""
UnifiedTrainer examples that also **persist artifacts to disk** using the SAME
paths as in the original 8 scripts you shared. These examples:

- Use your `UnifiedTrainer` (object-based training).
- Read only from your CSVs (no synthetic data is created).
- Save models, importances, and summaries to the exact paths used in your scripts.
- For tabular cases (Models 1 & 2), we also compute SHAP means to match your CSVs.
- For text cases (Models 3 & 4), we save importances like in your code (no SHAP there).

You can call any `run_*` function individually.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib
import shap
from loguru import logger

from trainer import TrainConfig, UnifiedTrainer


# ---------------------------
# Utilities
# ---------------------------

def _ensure_dirs(paths: List[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _sanitize(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


# --- Feature builder for SHAP (tabular only) ---
# Mirrors UnifiedTrainer's tabular preprocessing so columns line up for SHAP/feature CSVs.

def _build_tabular_like_trainer(df: pd.DataFrame, drop_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
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


# ---------------------------
# Model 1 — products_features (tabular)
# ---------------------------

def run_model1_products_features_global() -> None:
    input_path = "preprocessing/products_features.csv"
    model_path = "study/models/products_lgbm_model.pkl"
    importance_path = "study/importance/products_features.csv"
    summary_path = "study/products_summary.csv"

    if not os.path.exists(input_path):
        logger.error("Missing input: {p}", p=input_path)
        return

    _ensure_dirs(["study/models", "study/importance"])  # per script
    df = pd.read_csv(input_path)

    drop_cols = [
        "productId", "title", "shortTitle", "description", "shortDescription",
        "title_review", "reviewText", "language", "releaseDateUtc", "lastUpdateDateUtc",
    ]

    cfg = TrainConfig(target_col="position", drop_cols=drop_cols, text_fields=None, optuna_trials=30)
    trainer = UnifiedTrainer(cfg)
    result = trainer.fit(df)

    # Persist model
    model = result.models["__global__"]
    joblib.dump(model, model_path)

    # Persist importance (+ SHAP mean & impact like original)
    imp = result.importances["__global__"].copy()
    X_all, y_all, _ = _build_tabular_like_trainer(df, drop_cols, target_col="position")
    expl = shap.TreeExplainer(model)
    shap_vals = expl.shap_values(X_all)
    mean_shap = np.mean(shap_vals, axis=0)

    def _impact(val: float) -> str:
        # match your mapping: <0 -> positive, >0 -> negative
        return "positive" if val < 0 else "negative" if val > 0 else "neutral"

    shap_df = pd.DataFrame({"feature": X_all.columns, "mean_shap": mean_shap})
    imp = imp.merge(shap_df, on="feature", how="left")
    imp["impact_shap"] = imp["mean_shap"].apply(_impact)
    imp.to_csv(importance_path, index=False)

    # Persist summary
    result.summary[["rmse"]].to_csv(summary_path, index=False)
    logger.success("Saved model, importance, summary for Model1 Global")

def run_model1_products_features_per_query() -> None:
    input_path = "preprocessing/products_features.csv"
    base_models = "query_study/products_features/models"
    base_imps = "query_study/products_features/importance"
    summary_path = "query_study/products_features/products_summary.csv"

    if not os.path.exists(input_path):
        logger.error("Missing input: {p}", p=input_path)
        return

    _ensure_dirs([base_models, base_imps])
    df = pd.read_csv(input_path)

    drop_cols = [
        "query", "productId", "title", "shortTitle", "description", "shortDescription",
        "title_review", "reviewText", "language", "releaseDateUtc", "lastUpdateDateUtc",
    ]

    cfg = TrainConfig(target_col="position", drop_cols=drop_cols, text_fields=None, min_samples_per_group=10)
    trainer = UnifiedTrainer(cfg)
    result = trainer.fit(df, group_col="query")

    # Save per-group artifacts
    for g, model in result.models.items():
        filename_base = _sanitize(g)
        model_path = os.path.join(base_models, f"{filename_base}.pkl")
        imp_path = os.path.join(base_imps, f"{filename_base}.csv")
        joblib.dump(model, model_path)

        # Build features for SHAP on that group's rows
        df_g = df[df["query"] == g]
        X_g, _, _ = _build_tabular_like_trainer(df_g, drop_cols, target_col="position")
        expl = shap.TreeExplainer(model)
        shap_vals = expl.shap_values(X_g)
        mean_shap = np.mean(shap_vals, axis=0)

        imp = result.importances[g].copy()
        shap_df = pd.DataFrame({"feature": X_g.columns, "mean_shap": mean_shap})
        imp = imp.merge(shap_df, on="feature", how="left")
        imp["impact_shap"] = imp["mean_shap"].apply(lambda v: "positive" if v < 0 else "negative" if v > 0 else "neutral")
        imp.to_csv(imp_path, index=False)

    # Summary (query, samples, rmse)
    summary = result.summary.rename(columns={"group": "query"})
    summary[["query", "samples", "rmse"]].to_csv(summary_path, index=False)
    logger.success("Saved per-query artifacts for Model1")


# ---------------------------
# Model 2 — reviews_features (tabular)
# ---------------------------

def run_model2_reviews_features_global() -> None:
    input_path = "preprocessing/reviews_features.csv"
    model_path = "study/models/reviews_lgbm_model.pkl"
    importance_path = "study/importance/reviews_features.csv"
    summary_path = "study/reviews_summary.csv"

    if not os.path.exists(input_path):
        logger.error("Missing input: {p}", p=input_path)
        return

    _ensure_dirs(["study/models", "study/importance"])  # per script
    df = pd.read_csv(input_path, engine="python")

    drop_cols = [
        "productId", "position",  # keep target out of features
        "title", "shortTitle", "description", "shortDescription",
        "title_review", "reviewText",
    ]

    cfg = TrainConfig(target_col="position", drop_cols=drop_cols, text_fields=None)
    trainer = UnifiedTrainer(cfg)
    result = trainer.fit(df)

    # Model
    model = result.models["__global__"]
    joblib.dump(model, model_path)

    # Importance (+ SHAP like your global v2)
    imp = result.importances["__global__"].copy()
    X_all, y_all, _ = _build_tabular_like_trainer(df, drop_cols, target_col="position")
    expl = shap.TreeExplainer(model)
    shap_vals = expl.shap_values(X_all)
    mean_shap = np.mean(shap_vals, axis=0)
    imp = imp.merge(pd.DataFrame({"feature": X_all.columns, "mean_shap": mean_shap}), on="feature", how="left")
    imp["impact_shap"] = imp["mean_shap"].apply(lambda v: "positive" if v < 0 else "negative" if v > 0 else "neutral")
    imp.to_csv(importance_path, index=False)

    # Summary
    result.summary[["rmse"]].to_csv(summary_path, index=False)
    logger.success("Saved model, importance, summary for Model2 Global")

def run_model2_reviews_features_per_query() -> None:
    input_path = "preprocessing/reviews_features.csv"
    base_models = "query_study/reviews_features/models"
    base_imps = "query_study/reviews_features/importance"
    summary_path = "query_study/reviews_features/reviews_summary.csv"

    if not os.path.exists(input_path):
        logger.error("Missing input: {p}", p=input_path)
        return

    _ensure_dirs([base_models, base_imps])
    df = pd.read_csv(input_path, engine="python")

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
        expl = shap.TreeExplainer(model)
        shap_vals = expl.shap_values(X_g)
        mean_shap = np.mean(shap_vals, axis=0)

        imp = result.importances[g].copy()
        imp = imp.merge(pd.DataFrame({"feature": X_g.columns, "mean_shap": mean_shap}), on="feature", how="left")
        imp["impact_shap"] = imp["mean_shap"].apply(lambda v: "positive" if v < 0 else "negative" if v > 0 else "neutral")
        imp.to_csv(imp_path, index=False)

    summary = result.summary.rename(columns={"group": "query"})
    summary[["query", "samples", "rmse"]].to_csv(summary_path, index=False)
    logger.success("Saved per-query artifacts for Model2")


# ---------------------------
# Model 3 — products_texts (text)
# ---------------------------

def run_model3_products_texts_global() -> None:
    input_path = "preprocessing/products_texts.csv"
    model_path = "study/models/products_text_lgbm.pkl"
    importance_path = "study/importance/products_text.csv"
    summary_path = "study/products_text_summary.csv"

    if not os.path.exists(input_path):
        logger.error("Missing input: {p}", p=input_path)
        return

    _ensure_dirs(["study/models", "study/importance"])  # per script
    df = pd.read_csv(input_path)
    df = df[df["position"].notnull()]

    text_fields = ["title", "shortTitle", "description", "shortDescription"]
    cfg = TrainConfig(target_col="position", text_fields=text_fields, tfidf_max_features=2000, tfidf_min_df=5, tfidf_ngram_range=(1, 1))
    trainer = UnifiedTrainer(cfg)
    result = trainer.fit(df)

    model = result.models["__global__"]
    joblib.dump(model, model_path)

    # Importance already contains TF-IDF feature names from the trainer
    result.importances["__global__"].to_csv(importance_path, index=False)

    result.summary[["rmse"]].to_csv(summary_path, index=False)
    logger.success("Saved model, importance, summary for Model3 Global")

def run_model3_products_texts_per_query() -> None:
    input_path = "preprocessing/products_texts.csv"
    base_models = "query_study/products_texts/models"
    base_imps = "query_study/products_texts/importance"
    summary_path = "query_study/products_texts/products_texts_summary.csv"

    if not os.path.exists(input_path):
        logger.error("Missing input: {p}", p=input_path)
        return

    _ensure_dirs([base_models, base_imps])
    df = pd.read_csv(input_path)
    df = df[df["position"].notnull()]

    text_fields = ["title", "shortTitle", "description", "shortDescription"]
    cfg = TrainConfig(target_col="position", text_fields=text_fields, tfidf_max_features=2000, tfidf_min_df=2, tfidf_ngram_range=(1, 1), min_samples_per_group=3)
    trainer = UnifiedTrainer(cfg)
    result = trainer.fit(df, group_col="query")

    for g, model in result.models.items():
        filename_base = _sanitize(g)
        joblib.dump(model, os.path.join(base_models, f"{filename_base}.pkl"))
        result.importances[g].to_csv(os.path.join(base_imps, f"{filename_base}.csv"), index=False)

    summary = result.summary.rename(columns={"group": "query"})
    summary[["query", "samples", "rmse"]].to_csv(summary_path, index=False)
    logger.success("Saved per-query artifacts for Model3")


# ---------------------------
# Model 4 — reviews_texts (text)
# ---------------------------

def run_model4_reviews_texts_global() -> None:
    input_path = "preprocessing/reviews_texts.csv"
    model_path = "study/models/reviews_text_lgbm.pkl"
    importance_path = "study/importance/reviews_text.csv"
    summary_path = "study/reviews_text_summary.csv"

    if not os.path.exists(input_path):
        logger.error("Missing input: {p}", p=input_path)
        return

    _ensure_dirs(["study/models", "study/importance"])  # per script
    df = pd.read_csv(input_path, engine="python")
    df = df[df["position"].notnull() & df["reviewText"].notnull()]

    text_fields = ["title", "reviewText"]
    cfg = TrainConfig(target_col="position", text_fields=text_fields, tfidf_max_features=2000, tfidf_min_df=5, tfidf_ngram_range=(1, 2))
    trainer = UnifiedTrainer(cfg)
    result = trainer.fit(df)

    model = result.models["__global__"]
    joblib.dump(model, model_path)
    result.importances["__global__"].to_csv(importance_path, index=False)
    result.summary[["rmse"]].to_csv(summary_path, index=False)
    logger.success("Saved model, importance, summary for Model4 Global")

def run_model4_reviews_texts_per_query() -> None:
    input_path = "preprocessing/reviews_texts.csv"
    base_models = "query_study/reviews_texts/models"
    base_imps = "query_study/reviews_texts/importance"
    summary_path = "query_study/reviews_texts/reviews_text_summary.csv"

    if not os.path.exists(input_path):
        logger.error("Missing input: {p}", p=input_path)
        return

    _ensure_dirs([base_models, base_imps])
    df = pd.read_csv(input_path, engine="python")
    df = df[df["position"].notnull() & df["reviewText"].notnull()]

    text_fields = ["title", "reviewText"]
    cfg = TrainConfig(target_col="position", text_fields=text_fields, tfidf_max_features=2000, tfidf_min_df=5, tfidf_ngram_range=(1, 2), min_samples_per_group=1)
    trainer = UnifiedTrainer(cfg)
    result = trainer.fit(df, group_col="query")

    for g, model in result.models.items():
        filename_base = _sanitize(g)
        joblib.dump(model, os.path.join(base_models, f"{filename_base}.pkl"))
        result.importances[g].to_csv(os.path.join(base_imps, f"{filename_base}.csv"), index=False)

    summary = result.summary.rename(columns={"group": "query"})
    summary[["query", "samples", "rmse"]].to_csv(summary_path, index=False)
    logger.success("Saved per-query artifacts for Model4")


# ---------------------------
# Analysis & Reporting
# ---------------------------

def _write_csv(df: pd.DataFrame, path: str) -> None:
    """Safe CSV writer that creates parent folders."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _load_importance(path: str, top_n: int | None = None) -> pd.DataFrame | None:
    """
    Load a feature/term importance CSV saved by training.
    Tries to detect the correct importance column robustly.
    Returns a normalized, sorted dataframe with columns:
    ['feature', 'importance', 'normalized_importance', *optional 'impact_shap'].
    """
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if df.empty:
        return None

    # Detect feature column
    feat_col = "feature" if "feature" in df.columns else df.columns[0]

    # Detect importance-like column
    candidates = [
        "importance", "gain", "Gain", "score", "weight", "split",
        "importance_gain", "importance_gain_normalized", "mean_gain", "avg_gain"
    ]
    imp_col = next((c for c in candidates if c in df.columns), None)

    # Fallbacks: use SHAP means if present; else first numeric
    if imp_col is None:
        if "mean_shap" in df.columns:
            df["_importance"] = df["mean_shap"].abs()
        else:
            nums = df.select_dtypes(include=["number"]).columns.tolist()
            if not nums:
                return None
            df["_importance"] = df[nums[0]].abs()
    else:
        df["_importance"] = pd.to_numeric(df[imp_col], errors="coerce").abs().fillna(0)

    out = df[[feat_col, "_importance"]].copy()
    out.columns = ["feature", "importance"]

    # Keep SHAP direction if present
    if "impact_shap" in df.columns:
        out["impact_shap"] = df["impact_shap"]

    total = out["importance"].sum()
    out["normalized_importance"] = out["importance"] / total if total > 0 else 0.0
    out = out.sort_values("importance", ascending=False)
    if top_n:
        out = out.head(top_n)
    return out


def analyze_and_summarize_outputs(top_n: int = 30) -> None:
    """
    Aggregate training outputs into easy-to-consume CSVs + a tiny markdown report.
    Creates folder: study/analysis
    Files produced:
      - study/analysis/global_rmse.csv
      - study/analysis/per_query_rmse.csv
      - study/analysis/best_model_per_query.csv
      - study/analysis/overview.csv                (global + per-query aggregates)
      - study/analysis/top_features_*.csv          (4 files: tabular + text)
      - study/analysis/report.md
    """
    import datetime as _dt

    out_dir = "study/analysis"
    _ensure_dirs([out_dir])

    # ---------- Global RMSE ----------
    global_map = {
        "model1_products_features": "study/products_summary.csv",
        "model2_reviews_features": "study/reviews_summary.csv",
        "model3_products_texts": "study/products_text_summary.csv",
        "model4_reviews_texts": "study/reviews_text_summary.csv",
    }

    global_rows = []
    for model, path in global_map.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "rmse" in df.columns and not df.empty:
                    rmse = float(df["rmse"].iloc[0])
                    global_rows.append({"model": model, "rmse": rmse})
            except Exception as e:
                logger.warning(f"Failed to read {path}: {e}")

    global_rmse_df = pd.DataFrame(global_rows).sort_values("rmse") if global_rows else pd.DataFrame(columns=["model", "rmse"])
    _write_csv(global_rmse_df, f"{out_dir}/global_rmse.csv")

    # ---------- Per-query RMSE ----------
    per_query_files = [
        ("model1_products_features", "query_study/products_features/products_summary.csv"),
        ("model2_reviews_features", "query_study/reviews_features/reviews_summary.csv"),
        ("model3_products_texts", "query_study/products_texts/products_texts_summary.csv"),
        ("model4_reviews_texts", "query_study/reviews_texts/reviews_text_summary.csv"),
    ]

    per_parts = []
    for model, path in per_query_files:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # expected columns: query, samples, rmse
                cols = {c.lower(): c for c in df.columns}
                q = cols.get("query")
                r = cols.get("rmse")
                s = cols.get("samples")
                if not (q and r):
                    logger.warning(f"Unexpected columns in {path}; got {df.columns.tolist()}")
                    continue
                part = df[[q, r] + ([s] if s else [])].copy()
                part.columns = ["query", "rmse"] + (["samples"] if s else [])
                part["model"] = model
                per_parts.append(part)
            except Exception as e:
                logger.warning(f"Failed to read {path}: {e}")

    if per_parts:
        per_query_rmse = pd.concat(per_parts, ignore_index=True)
        # Enforce dtypes
        per_query_rmse["rmse"] = pd.to_numeric(per_query_rmse["rmse"], errors="coerce")
        if "samples" in per_query_rmse.columns:
            per_query_rmse["samples"] = pd.to_numeric(per_query_rmse.get("samples"), errors="coerce")
        _write_csv(per_query_rmse, f"{out_dir}/per_query_rmse.csv")

        # Leaderboard: best model per query
        best = (
            per_query_rmse.sort_values(["query", "rmse"])
            .groupby("query", as_index=False)
            .first()
        )
        best = best[["query", "model", "rmse"]]
        _write_csv(best, f"{out_dir}/best_model_per_query.csv")

        # Per-model aggregates across queries
        agg = per_query_rmse.groupby("model", as_index=False).agg(
            avg_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            queries=("query", "nunique")
        )
        wins = best["model"].value_counts().rename_axis("model").reset_index(name="wins")
        overview = agg.merge(wins, on="model", how="left").fillna({"wins": 0})

        # Join with global RMSE if available
        if not global_rmse_df.empty:
            overview = overview.merge(global_rmse_df, on="model", how="left").rename(columns={"rmse": "global_rmse"})

        overview = overview.sort_values(["avg_rmse", "median_rmse", "wins"], ascending=[True, True, False])
        _write_csv(overview, f"{out_dir}/overview.csv")
    else:
        per_query_rmse = pd.DataFrame(columns=["query", "rmse", "samples", "model"])
        overview = pd.DataFrame(columns=["model", "avg_rmse", "median_rmse", "queries", "wins", "global_rmse"])

    # ---------- Top features / terms (global) ----------
    tops = {
        "top_features_products_features.csv": _load_importance("study/importance/products_features.csv", top_n),
        "top_features_reviews_features.csv": _load_importance("study/importance/reviews_features.csv", top_n),
        "top_terms_products_texts.csv": _load_importance("study/importance/products_text.csv", top_n),
        "top_terms_reviews_texts.csv": _load_importance("study/importance/reviews_text.csv", top_n),
    }
    for fname, df in tops.items():
        if df is not None:
            _write_csv(df, f"{out_dir}/{fname}")

    # ---------- Tiny Markdown report ----------
    # We avoid pandas.to_markdown dependency; keep it simple.
    lines = []
    lines.append("# Training Report")
    lines.append(f"_Generated: { _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S') }_")
    lines.append("")
    lines.append("## Global RMSE")
    if not global_rmse_df.empty:
        for _, r in global_rmse_df.iterrows():
            lines.append(f"- **{r['model']}**: RMSE = {r['rmse']:.5f}")
    else:
        lines.append("- No global summaries found.")

    lines.append("")
    lines.append("## Per-Query Overview")
    if "model" in overview.columns and not overview.empty:
        for _, r in overview.iterrows():
            gm = "" if pd.isna(r.get("global_rmse")) else f", global={r['global_rmse']:.5f}"
            lines.append(
                f"- **{r['model']}**: avg={r['avg_rmse']:.5f}, median={r['median_rmse']:.5f}, "
                f"wins={int(r['wins'])}, queries={int(r['queries'])}{gm}"
            )
    else:
        lines.append("- No per-query summaries found.")

    lines.append("")
    lines.append("## Artifacts saved")
    lines.append("- `study/analysis/global_rmse.csv`")
    lines.append("- `study/analysis/per_query_rmse.csv`")
    lines.append("- `study/analysis/best_model_per_query.csv`")
    lines.append("- `study/analysis/overview.csv`")
    lines.append("- `study/analysis/top_features_products_features.csv`")
    lines.append("- `study/analysis/top_features_reviews_features.csv`")
    lines.append("- `study/analysis/top_terms_products_texts.csv`")
    lines.append("- `study/analysis/top_terms_reviews_texts.csv`")

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.success("Analysis complete. See 'study/analysis/' for outputs.")


# ---------------------------
# MAIN (updated)
# ---------------------------

if __name__ == "__main__":
    # 1) Train & persist (unchanged)
    # run_model1_products_features_global()
    # run_model1_products_features_per_query()
    # run_model2_reviews_features_global()
    # run_model2_reviews_features_per_query()
    # run_model3_products_texts_global()
    # run_model3_products_texts_per_query()
    # run_model4_reviews_texts_global()
    # run_model4_reviews_texts_per_query()

    # 2) Analyze outputs
    analyze_and_summarize_outputs(top_n=30)
