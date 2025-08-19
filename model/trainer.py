from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable, Union

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# ---------------------------
# Config
# ---------------------------

@dataclass
class TrainConfig:
    target_col: str = "position"
    log_target: bool = True
    drop_cols: List[str] = field(default_factory=list)

    # Text
    text_fields: Optional[List[str]] = None
    tfidf_max_features: int = 2000
    tfidf_min_df: int = 2
    tfidf_ngram_range: Tuple[int, int] = (1, 1)

    # Split
    test_size: float = 0.2
    random_state: int = 42

    # Optuna
    optuna_trials: int = 30

    # Grouped training
    min_samples_per_group: int = 10

    # LightGBM
    force_row_wise: bool = True


# ---------------------------
# Trainer
# ---------------------------

class UnifiedTrainer:
    def __init__(self, config: TrainConfig):
        self.cfg = config
        self._tfidf: Optional[TfidfVectorizer] = None
        self._tfidf_total_features_: Optional[int] = None  # cached total feature count

    # --------- Public API ---------

    def fit(
        self,
        df: pd.DataFrame,
        group_col: Optional[str] = None,
        pretrained_models: Optional[Dict[str, lgb.LGBMRegressor]] = None
    ) -> "TrainResult":
        """
        If group_col is None: train single global model.
        Else: train per-group models (skipping groups with too few samples).
        """
        logger.info(f"Starting training. Group col: {group_col}")
        if group_col is None:
            key = "__global__"
            model, importance_df, rmse = self._fit_single(
                df=df,
                pretrained=(pretrained_models or {}).get(key)
            )
            summary = pd.DataFrame([{"group": key, "samples": len(df), "rmse": rmse}])
            logger.success(f"Finished training global model with RMSE: {rmse:.4f}")
            return TrainResult(models={key: model}, importances={key: importance_df}, summary=summary)

        models: Dict[str, lgb.LGBMRegressor] = {}
        importances: Dict[str, pd.DataFrame] = {}
        rows: List[Dict[str, Union[str, int, float]]] = []

        for g, df_g in df.groupby(group_col, dropna=False):
            n = len(df_g)
            if n < self.cfg.min_samples_per_group:
                logger.warning(f"Skipping group {g!r}, only {n} samples.")
                continue
            logger.info(f"Training group {g!r} with {n} samples.")
            model, importance_df, rmse = self._fit_single(
                df=df_g,
                pretrained=(pretrained_models or {}).get(str(g))
            )
            models[str(g)] = model
            importances[str(g)] = importance_df
            rows.append({"group": g, "samples": n, "rmse": rmse})

        summary = (
            pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
            if rows else pd.DataFrame(columns=["group", "samples", "rmse"])
        )
        logger.success(f"Finished per-group training with {len(models)} models.")
        return TrainResult(models=models, importances=importances, summary=summary)

    # --------- Core single-model fit ---------

    def _fit_single(
        self,
        df: pd.DataFrame,
        pretrained: Optional[lgb.LGBMRegressor] = None
    ) -> Tuple[lgb.LGBMRegressor, pd.DataFrame, float]:
        df = df.copy()
        df = df[df[self.cfg.target_col].notnull()]

        # Target
        y_raw_all = df[self.cfg.target_col].astype(float).values
        y_all = np.log1p(y_raw_all) if self.cfg.log_target else y_raw_all
        logger.debug(f"Prepared target vector with shape: {y_all.shape}")

        # Features
        if self.cfg.text_fields:
            X_all, feature_names = self._build_text_matrix(df, self.cfg.text_fields)
            cat_feats: Optional[List[str]] = None  # no cats in text vectorizer
        else:
            X_all, feature_names, cat_feats = self._build_tabular_matrix(df, self.cfg.drop_cols)

        logger.debug(f"Feature matrix shape: {getattr(X_all, 'shape', None)}")

        # Split; keep both y (possibly log) and y_raw for correct metric on original scale
        X_train, X_val, y_train, y_val, y_raw_train, y_raw_val = train_test_split(
            X_all, y_all, y_raw_all,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state
        )

        # Train (pretrained continuation or optuna)
        if pretrained is not None:
            logger.info("Continuing training from pretrained model.")
            model = pretrained
            model.set_params(
                force_row_wise=self.cfg.force_row_wise,
                random_state=self.cfg.random_state,
                verbose=-1
            )
            model.fit(
                X_train, y_train,
                init_model=getattr(model, "booster_", None),
                categorical_feature=cat_feats if cat_feats else "auto",
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
            )
        else:
            model = self._train_with_optuna(
                X_train, y_train, X_val, y_val, cat_feats
            )

        preds_val = model.predict(
            X_val,
            num_iteration=getattr(model, "best_iteration_", None)
        )

        # Report RMSE on original scale if log_target is True
        if self.cfg.log_target:
            preds_val_lin = np.expm1(preds_val)
            rmse = float(np.sqrt(mean_squared_error(y_raw_val, preds_val_lin)))
        else:
            rmse = float(np.sqrt(mean_squared_error(y_val, preds_val)))

        logger.info(f"Model RMSE: {rmse:.4f}")

        # Feature importance
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_.astype(float)
        })
        total = float(importance_df["importance"].sum())
        importance_df["importance_percent"] = (
            100.0 * importance_df["importance"] / total
        ) if total > 0 else 0.0
        importance_df = importance_df.sort_values(
            "importance_percent", ascending=False
        ).reset_index(drop=True)

        # Attach metadata to model
        try:
            setattr(model, "feature_names_", list(feature_names))
            setattr(model, "target_log_transformed_", bool(self.cfg.log_target))
            if self.cfg.text_fields:
                setattr(model, "is_text_model_", True)
                setattr(model, "text_fields_", list(self.cfg.text_fields))
                setattr(model, "tfidf_", self._tfidf)
                total_feats = getattr(self, "_tfidf_total_features_", None)
                if total_feats is not None:
                    setattr(model, "tfidf_total_features_", int(total_feats))
                setattr(model, "tfidf_limited_features_", int(len(feature_names)))
            else:
                setattr(model, "is_text_model_", False)
                setattr(model, "categorical_features_", list(cat_feats or []))
        except Exception as e:
            logger.warning(f"Failed to attach metadata to model: {e}")

        return model, importance_df, rmse

    # --------- Builders ---------

    def _build_tabular_matrix(
        self,
        df: pd.DataFrame,
        drop_cols: Iterable[str]
    ) -> Tuple[pd.DataFrame, List[str], Optional[List[str]]]:
        """
        Keep native pandas 'category' dtype for categoricals.
        Do not cast the whole frame to float — it breaks categorical handling.
        """
        feature_cols = [c for c in df.columns if c not in set(drop_cols) | {self.cfg.target_col}]
        X = df[feature_cols].copy()

        # Identify column types before renaming
        obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = [c for c in X.columns if c not in obj_cols]

        # Sanitize names ONCE and apply to both lists
        new_names = (
            pd.Index(X.columns)
            .str.replace(r"[^\w]", "_", regex=True)
            .str.replace(r"_+", "_", regex=True)
            .str.strip("_")
        )
        rename_map = dict(zip(X.columns, new_names))
        X = X.rename(columns=rename_map)
        obj_cols = [rename_map[c] for c in obj_cols]
        num_cols = [rename_map[c] for c in num_cols]

        # Cast categoricals natively
        for col in obj_cols:
            X[col] = X[col].astype("category")

        # Numeric cleanup
        if num_cols:
            X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")
            X[num_cols] = X[num_cols].fillna(0.0).astype(np.float32)

        categorical_cols = obj_cols  # after renaming
        feature_names = X.columns.tolist()

        # Log tabular feature count (no cap)
        logger.info(f"Tabular features: total={len(feature_names)} (no cap)")

        return X, feature_names, (categorical_cols or None)

    def _build_text_matrix(
        self,
        df: pd.DataFrame,
        fields: List[str]
    ) -> Tuple[sparse.csr_matrix, List[str]]:
        """
        Build TF-IDF sparse matrix. Also log total vs limited feature counts.
        """
        text = df[fields].fillna("").agg(" ".join, axis=1)

        base_kwargs = dict(
            min_df=self.cfg.tfidf_min_df,
            ngram_range=self.cfg.tfidf_ngram_range,
            stop_words="english",
        )

        if self._tfidf is None:
            # Probe to get the total (uncapped) vocabulary size
            total_features = None
            try:
                probe = TfidfVectorizer(max_features=None, **base_kwargs)
                probe.fit(text)
                total_features = len(probe.get_feature_names_out())
                self._tfidf_total_features_ = int(total_features)
            except Exception as e:
                logger.warning(f"TF-IDF probe failed: {e}")

            # Actual capped vectorizer
            self._tfidf = TfidfVectorizer(
                max_features=self.cfg.tfidf_max_features,
                **base_kwargs
            )
            X_sparse = self._tfidf.fit_transform(text)
            limited_features = len(self._tfidf.get_feature_names_out())

            if total_features is not None:
                logger.info(
                    f"TF-IDF features: total={total_features}, "
                    f"limited={limited_features}, cap={self.cfg.tfidf_max_features}"
                )
            else:
                logger.info(
                    f"TF-IDF features: limited={limited_features}, "
                    f"cap={self.cfg.tfidf_max_features}"
                )
        else:
            # Reuse fitted vectorizer
            X_sparse = self._tfidf.transform(text)
            limited_features = len(self._tfidf.get_feature_names_out())
            total_features = getattr(self, "_tfidf_total_features_", None)

            if total_features is not None:
                logger.info(
                    f"TF-IDF features (cached): total={total_features}, "
                    f"limited={limited_features}, cap={self.cfg.tfidf_max_features}"
                )
            else:
                logger.info(
                    f"TF-IDF features (cached): limited={limited_features}, "
                    f"cap={self.cfg.tfidf_max_features}"
                )

        feature_names = self._tfidf.get_feature_names_out().tolist()
        return X_sparse.tocsr(), feature_names

    # --------- Optuna search ---------

    def _train_with_optuna(
        self,
        X_train: Union[pd.DataFrame, np.ndarray, sparse.csr_matrix],
        y_train: np.ndarray,
        X_val: Union[pd.DataFrame, np.ndarray, sparse.csr_matrix],
        y_val: np.ndarray,
        categorical_feature: Optional[List[str]] = None
    ) -> lgb.LGBMRegressor:
        logger.info("Starting Optuna hyperparameter search.")

        def objective(trial: optuna.Trial) -> float:
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 300, 1200),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1),
                num_leaves=trial.suggest_int("num_leaves", 31, 127),
                min_child_samples=trial.suggest_int("min_child_samples", 10, 80),
                feature_fraction=trial.suggest_float("feature_fraction", 0.6, 1.0),
                bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
                bagging_freq=trial.suggest_int("bagging_freq", 0, 10),
                force_row_wise=self.cfg.force_row_wise,
                random_state=self.cfg.random_state,
                verbose=-1,
            )
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                categorical_feature=categorical_feature if categorical_feature else "auto",
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
            )
            preds = model.predict(X_val, num_iteration=getattr(model, "best_iteration_", None))
            # Optuna minimizes MSE; RMSE ~ sqrt(MSE) is monotonic — fine
            return mean_squared_error(y_val, preds)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.cfg.optuna_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params.update({
            "force_row_wise": self.cfg.force_row_wise,
            "random_state": self.cfg.random_state,
            "verbose": -1
        })
        logger.success(f"Optuna best params: {best_params}")

        best = lgb.LGBMRegressor(**best_params)
        best.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            categorical_feature=categorical_feature if categorical_feature else "auto",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
        )
        return best


# ---------------------------
# Result wrapper
# ---------------------------

@dataclass
class TrainResult:
    models: Dict[str, lgb.LGBMRegressor]
    importances: Dict[str, pd.DataFrame]
    summary: pd.DataFrame

    def best_groups(self, top_k: int = 10) -> pd.DataFrame:
        return (
            self.summary.sort_values("rmse").head(top_k).reset_index(drop=True)
            if not self.summary.empty else self.summary
        )
