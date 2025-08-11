from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class TrainConfig:
    target_col: str = "position"
    log_target: bool = True
    drop_cols: List[str] = field(default_factory=list)
    text_fields: Optional[List[str]] = None
    tfidf_max_features: int = 2000
    tfidf_min_df: int = 5
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    test_size: float = 0.2
    random_state: int = 42
    optuna_trials: int = 30
    min_samples_per_group: int = 10
    force_row_wise: bool = True

class UnifiedTrainer:
    def __init__(self, config: TrainConfig):
        self.cfg = config
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._tfidf: Optional[TfidfVectorizer] = None

    def fit(self, df: pd.DataFrame, group_col: Optional[str] = None, pretrained_models: Optional[Dict[str, lgb.LGBMRegressor]] = None) -> "TrainResult":
        logger.info(f"Starting training. Group col: {group_col}")
        if group_col is None:
            key = "__global__"
            model, importance_df, rmse = self._fit_single(df, pretrained_models.get(key) if pretrained_models else None)
            summary = pd.DataFrame([{ "group": key, "samples": len(df), "rmse": rmse }])
            logger.success(f"Finished training global model with RMSE: {rmse:.4f}")
            return TrainResult(models={key: model}, importances={key: importance_df}, summary=summary)

        models, importances, rows = {}, {}, []
        for g, df_g in df.groupby(group_col):
            if len(df_g) < self.cfg.min_samples_per_group:
                logger.warning(f"Skipping group {g}, only {len(df_g)} samples.")
                continue
            logger.info(f"Training group {g} with {len(df_g)} samples.")
            model, importance_df, rmse = self._fit_single(df_g, (pretrained_models or {}).get(str(g)))
            models[str(g)], importances[str(g)] = model, importance_df
            rows.append({"group": g, "samples": len(df_g), "rmse": rmse})
        summary = pd.DataFrame(rows).sort_values("rmse") if rows else pd.DataFrame(columns=["group", "samples", "rmse"])
        logger.success(f"Finished per-group training with {len(models)} models.")
        return TrainResult(models=models, importances=importances, summary=summary)

    def _fit_single(self, df: pd.DataFrame, pretrained: Optional[lgb.LGBMRegressor] = None) -> Tuple[lgb.LGBMRegressor, pd.DataFrame, float]:
        df = df.copy()
        df = df[df[self.cfg.target_col].notnull()]
        y_raw = df[self.cfg.target_col].astype(float).values
        y = np.log1p(y_raw) if self.cfg.log_target else y_raw
        logger.debug(f"Prepared target vector with shape: {y.shape}")

        if self.cfg.text_fields:
            X, feature_names = self._build_text_matrix(df, self.cfg.text_fields)
            cat_feats = None
        else:
            X, feature_names, cat_feats = self._build_tabular_matrix(df, self.cfg.drop_cols)
        logger.debug(f"Feature matrix shape: {X.shape}")

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.cfg.test_size, random_state=self.cfg.random_state)

        if pretrained is not None:
            logger.info("Continuing training from pretrained model.")
            model = pretrained
            model.set_params(force_row_wise=self.cfg.force_row_wise, random_state=self.cfg.random_state, verbose=-1)
            model.fit(X_train, y_train, init_model=getattr(model, "booster_", None), categorical_feature=cat_feats if cat_feats else None)
        else:
            model = self._train_with_optuna(X_train, y_train, X_val, y_val, cat_feats)

        preds = model.predict(X_val)
        rmse = float(mean_squared_error(y_val, preds))
        logger.info(f"Model RMSE: {rmse:.4f}")

        importance_df = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        total = importance_df["importance"].sum()
        importance_df["importance_percent"] = (100.0 * importance_df["importance"] / total) if total > 0 else 0.0
        importance_df = importance_df.sort_values("importance_percent", ascending=False).reset_index(drop=True)

        return model, importance_df, rmse

    def _build_tabular_matrix(self, df: pd.DataFrame, drop_cols: Iterable[str]) -> Tuple[pd.DataFrame, List[str], Optional[List[str]]]:
        feature_cols = [c for c in df.columns if c not in set(drop_cols) | {self.cfg.target_col}]
        X = df[feature_cols].copy()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self._label_encoders[col] = le
        X.columns = pd.Index(X.columns).str.replace(r"[^\w]", "_", regex=True).str.replace(r"_+", "_", regex=True).str.strip("_")
        if categorical_cols:
            X[categorical_cols] = X[categorical_cols].astype("category")
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
        return X, X.columns.tolist(), categorical_cols or None

    def _build_text_matrix(self, df: pd.DataFrame, fields: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        text = df[fields].fillna("").agg(" ".join, axis=1)
        if self._tfidf is None:
            self._tfidf = TfidfVectorizer(max_features=self.cfg.tfidf_max_features, min_df=self.cfg.tfidf_min_df, ngram_range=self.cfg.tfidf_ngram_range, stop_words="english")
            X_sparse = self._tfidf.fit_transform(text)
        else:
            X_sparse = self._tfidf.transform(text)
        X_df = pd.DataFrame.sparse.from_spmatrix(X_sparse, columns=self._tfidf.get_feature_names_out())
        return X_df, list(X_df.columns)

    def _train_with_optuna(self, X_train, y_train, X_val, y_val, categorical_feature: Optional[List[str]] = None) -> lgb.LGBMRegressor:
        logger.info("Starting Optuna hyperparameter search.")
        def objective(trial: optuna.Trial) -> float:
            params = dict(n_estimators=trial.suggest_int("n_estimators", 100, 600), learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1), num_leaves=trial.suggest_int("num_leaves", 10, 50), min_child_samples=trial.suggest_int("min_child_samples", 10, 50), force_row_wise=self.cfg.force_row_wise, random_state=self.cfg.random_state, verbose=-1)
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, categorical_feature=categorical_feature if categorical_feature else None)
            preds = model.predict(X_val)
            return mean_squared_error(y_val, preds)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.cfg.optuna_trials, show_progress_bar=False)
        best_params = study.best_params
        best_params.update({"force_row_wise": self.cfg.force_row_wise, "random_state": self.cfg.random_state, "verbose": -1})
        logger.success(f"Optuna best params: {best_params}")
        best = lgb.LGBMRegressor(**best_params)
        best.fit(X_train, y_train, categorical_feature=categorical_feature if categorical_feature else None)
        return best

@dataclass
class TrainResult:
    models: Dict[str, lgb.LGBMRegressor]
    importances: Dict[str, pd.DataFrame]
    summary: pd.DataFrame
    def best_groups(self, top_k: int = 10) -> pd.DataFrame:
        return self.summary.sort_values("rmse").head(top_k).reset_index(drop=True) if not self.summary.empty else self.summary
