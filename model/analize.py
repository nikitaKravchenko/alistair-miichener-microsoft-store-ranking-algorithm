from __future__ import annotations

import os
import glob
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------
# Utilities
# ---------------------------

def _ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _detect_feat_col(df: pd.DataFrame) -> str:
    for c in ["feature", "term", "token", "ngram", "word", "feature_name", "name"]:
        if c in df.columns:
            return c
    return df.columns[0]


def _write_nonempty(df: Optional[pd.DataFrame], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if df is None:
        return
    df.to_csv(path, index=False)


def _load_importance_for_agg(path: str, top_n: Optional[int] = None) -> pd.DataFrame | None:
    """
    Load and standardize an importance CSV.
    Returns columns: feature, importance, normalized_importance, impact_shap (opt), rank.
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None

    # Standardize feature column
    feat_col = _detect_feat_col(df)
    if feat_col != "feature":
        df = df.rename(columns={feat_col: "feature"})

    # Pick an importance-like column or fallback to SHAP means
    candidates = [
        "importance", "gain", "Gain", "score", "weight", "split",
        "importance_gain", "importance_gain_normalized", "mean_gain", "avg_gain"
    ]
    imp_col = next((c for c in candidates if c in df.columns), None)
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

    out = df[["feature", "_importance"]].copy()
    out.columns = ["feature", "importance"]

    if "impact_shap" in df.columns:
        out["impact_shap"] = df["impact_shap"]
    else:
        out["impact_shap"] = np.nan

    total = out["importance"].sum()
    out["normalized_importance"] = out["importance"] / total if total > 0 else 0.0
    out = out.sort_values("importance", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)

    if top_n:
        out = out.head(top_n)

    return out


def _aggregate_per_query_dir(dir_path: str, model_name: str, top_n: Optional[int] = None) -> pd.DataFrame:
    """
    Aggregate all per-query importance CSVs for a model directory.
    Produces: appearances, presence_rate, mean/median normalized_importance, mean_rank,
              shares of SHAP impact (pos/neg/neutral) if available.
    """
    files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
    if not files:
        return pd.DataFrame(columns=[
            "model", "feature", "appearances", "queries_total", "presence_rate",
            "mean_normalized_importance", "median_normalized_importance", "mean_rank",
            "pos_share", "neg_share", "neu_share", "score_equal"
        ])

    parts = []
    for f in files:
        df = _load_importance_for_agg(f, top_n=top_n)
        if df is None or df.empty:
            continue
        df["query_file"] = os.path.basename(f)
        parts.append(df)

    if not parts:
        return pd.DataFrame(columns=[
            "model", "feature", "appearances", "queries_total", "presence_rate",
            "mean_normalized_importance", "median_normalized_importance", "mean_rank",
            "pos_share", "neg_share", "neu_share", "score_equal"
        ])

    all_df = pd.concat(parts, ignore_index=True)
    queries_total = len(files)

    def _share(s: pd.Series, label: str) -> float:
        if "impact_shap" not in all_df.columns:
            return np.nan
        s = s.astype(str)
        if s.empty:
            return np.nan
        return (s == label).mean()

    grouped = all_df.groupby("feature", as_index=False).agg(
        appearances=("feature", "size"),
        mean_normalized_importance=("normalized_importance", "mean"),
        median_normalized_importance=("normalized_importance", "median"),
        mean_rank=("rank", "mean"),
        pos_share=("impact_shap", lambda s: _share(s, "positive")),
        neg_share=("impact_shap", lambda s: _share(s, "negative")),
        neu_share=("impact_shap", lambda s: _share(s, "neutral")),
    )
    grouped["queries_total"] = queries_total
    grouped["presence_rate"] = grouped["appearances"] / grouped["queries_total"]
    grouped["model"] = model_name
    grouped["score_equal"] = grouped["mean_normalized_importance"] * grouped["presence_rate"]

    grouped = grouped.sort_values(
        ["score_equal", "presence_rate", "mean_normalized_importance", "appearances", "mean_rank"],
        ascending=[False, False, False, False, True]
    ).reset_index(drop=True)

    return grouped[
        ["model", "feature", "appearances", "queries_total", "presence_rate",
         "mean_normalized_importance", "median_normalized_importance", "mean_rank",
         "pos_share", "neg_share", "neu_share", "score_equal"]
    ]


def analyze_and_summarize_outputs(top_n: int = 30) -> None:
    """
    Build per-model artifacts (global + per-query aggregate) and cross-model stacks.
    Also writes a tiny markdown report.
    """
    import datetime as _dt

    input_dir = "data"
    out_dir = "data/analysis"
    _ensure_dirs([out_dir])

    # ---------- Global RMSE ----------
    global_map = {
        "model1_products_features": f"{input_dir}/study/global_products_features/summary.csv",
        "model2_reviews_features": f"{input_dir}/study/global_reviews_features/summary.csv",
        "model3_products_texts": f"{input_dir}/study/global_products_texts/summary.csv",
        "model4_reviews_texts": f"{input_dir}/study/global_reviews_texts/summary.csv"
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
                logger.warning("Failed to read {}: {}", path, e)

    global_rmse_df = pd.DataFrame(global_rows).sort_values("rmse") if global_rows else pd.DataFrame(columns=["model", "rmse"])
    _write_nonempty(global_rmse_df, f"{out_dir}/global_rmse.csv")

    # Model weights (for weighted stacks)
    if not global_rmse_df.empty:
        inv = 1.0 / global_rmse_df["rmse"]
        weights = (inv / inv.sum()).to_list()
        model_weights = dict(zip(global_rmse_df["model"], weights))
    else:
        model_weights = {}

    # ---------- Per-query RMSE + overview ----------
    per_query_files = {
        "model1_products_features": f"{input_dir}/study/query_products_features/summary.csv",
        "model2_reviews_features": f"{input_dir}/study/query_reviews_features/summary.csv",
        "model3_products_texts": f"{input_dir}/study/query_products_texts/summary.csv",
        "model4_reviews_texts": f"{input_dir}/study/query_reviews_texts/summary.csv"
    }

    per_parts = []
    for model, path in per_query_files.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                cols = {c.lower(): c for c in df.columns}
                q = cols.get("query")
                r = cols.get("rmse")
                s = cols.get("samples")
                if not (q and r):
                    logger.warning("Unexpected columns in {}; got {}", path, df.columns.tolist())
                    continue
                part = df[[q, r] + ([s] if s else [])].copy()
                part.columns = ["query", "rmse"] + (["samples"] if s else [])
                part["model"] = model
                per_parts.append(part)
            except Exception as e:
                logger.warning("Failed to read {}: {}", path, e)

    if per_parts:
        per_query_rmse = pd.concat(per_parts, ignore_index=True)
        per_query_rmse["rmse"] = pd.to_numeric(per_query_rmse["rmse"], errors="coerce")
        if "samples" in per_query_rmse.columns:
            per_query_rmse["samples"] = pd.to_numeric(per_query_rmse.get("samples"), errors="coerce")
        _write_nonempty(per_query_rmse, f"{out_dir}/per_query_rmse.csv")

        best = (
            per_query_rmse.sort_values(["query", "rmse"])
            .groupby("query", as_index=False)
            .first()
        )
        best = best[["query", "model", "rmse"]]
        _write_nonempty(best, f"{out_dir}/best_model_per_query.csv")

        agg = per_query_rmse.groupby("model", as_index=False).agg(
            avg_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            queries=("query", "nunique")
        )
        wins = best["model"].value_counts().rename_axis("model").reset_index(name="wins")
        overview = agg.merge(wins, on="model", how="left").fillna({"wins": 0})

        if not global_rmse_df.empty:
            overview = overview.merge(global_rmse_df, on="model", how="left").rename(columns={"rmse": "global_rmse"})

        overview = overview.sort_values(["avg_rmse", "median_rmse", "wins"], ascending=[True, True, False])
        _write_nonempty(overview, f"{out_dir}/overview.csv")
    else:
        overview = pd.DataFrame(columns=["model", "avg_rmse", "median_rmse", "queries", "wins", "global_rmse"])

    # ---------- Per-model TOP files (GLOBAL) ----------
    tops_global_map = {
        "model1_products_features": (f"{input_dir}/study/global_products_features/importance/features.csv", f"{out_dir}/model1_global_top.csv"),
        "model2_reviews_features":  (f"{input_dir}/study/global_reviews_features/importance/features.csv",  f"{out_dir}/model2_global_top.csv"),
        "model3_products_texts":    (f"{input_dir}/study/global_products_texts/importance/features.csv",     f"{out_dir}/model3_global_top_terms.csv"),
        "model4_reviews_texts":     (f"{input_dir}/study/global_reviews_texts/importance/features.csv",      f"{out_dir}/model4_global_top_terms.csv"),
    }

    global_stack_parts = []
    for model, (src, dst) in tops_global_map.items():
        df = _load_importance_for_agg(src, top_n=top_n)
        if df is None:
            continue
        df["model"] = model
        _write_nonempty(df[["model", "feature", "importance", "normalized_importance", "impact_shap", "rank"]], dst)
        global_stack_parts.append(df[["model", "feature", "normalized_importance", "impact_shap", "rank"]])

    # ---------- Per-model TOP files (PER-QUERY AGGREGATED) ----------
    per_query_dirs = {
        "model1_products_features": f"{input_dir}/study/query_products_features/importance",
        "model2_reviews_features":  f"{input_dir}/study/query_reviews_features/importance",
        "model3_products_texts":    f"{input_dir}/study/query_products_texts/importance",
        "model4_reviews_texts":     f"{input_dir}/study/query_reviews_texts/importance",
    }

    per_query_agg_parts = []
    for model, dirp in per_query_dirs.items():
        if os.path.isdir(dirp):
            agg_df = _aggregate_per_query_dir(dirp, model_name=model, top_n=top_n)
            if not agg_df.empty:
                per_query_agg_parts.append(agg_df)
                _write_nonempty(agg_df, f"{out_dir}/{model}_per_query_agg.csv")

    # ---------- Cross-model stacks ----------
    if global_stack_parts:
        global_stack = pd.concat(global_stack_parts, ignore_index=True)
        global_stack["score_equal"] = global_stack["normalized_importance"]
        global_stack["weight"] = global_stack["model"].map(model_weights).fillna(
            (1.0 / len(tops_global_map)) if tops_global_map else 0.25
        )
        global_stack["score_weighted"] = global_stack["score_equal"] * global_stack["weight"]
        global_stack = global_stack.sort_values(["score_weighted", "score_equal", "rank"], ascending=[False, False, True])
        _write_nonempty(global_stack, f"{out_dir}/all_models_global_equal_weighted.csv")

    if per_query_agg_parts:
        per_query_stack = pd.concat(per_query_agg_parts, ignore_index=True)
        per_query_stack["weight"] = per_query_stack["model"].map(model_weights).fillna(
            (1.0 / len(per_query_dirs)) if per_query_dirs else 0.25
        )
        per_query_stack["score_weighted"] = per_query_stack["score_equal"] * per_query_stack["weight"]
        per_query_stack = per_query_stack.sort_values(
            ["score_weighted", "score_equal", "presence_rate", "mean_normalized_importance", "appearances", "mean_rank"],
            ascending=[False, False, False, False, False, True]
        )
        _write_nonempty(per_query_stack, f"{out_dir}/all_models_per_query_equal_weighted.csv")

    # ---------- Tiny Markdown report ----------
    lines = []
    lines.append("*Training & Aggregation Report*")
    lines.append(f"_Generated: { _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S') }_")
    lines.append("")
    lines.append("*Global RMSE*")
    if not global_rmse_df.empty:
        for _, r in global_rmse_df.iterrows():
            lines.append(f"- **{r['model']}**: RMSE = {r['rmse']:.5f}")
    else:
        lines.append("- No global summaries found.")

    # lines.append("")
    # lines.append("## New per-model artifacts")
    # for model, (_, dst) in tops_global_map.items():
    #     lines.append(f"- {model}: global top → `{dst.split('/')[-1]}`")
    # for model, dirp in per_query_dirs.items():
    #     if os.path.isdir(dirp):
    #         lines.append(f"- {model}: per-query aggregate → `{model}_per_query_agg.csv`")
    #
    # lines.append("")
    # lines.append("## Cross-model stacks")
    # if os.path.exists(f"{out_dir}/all_models_global_equal_weighted.csv"):
    #     lines.append(f"- Global (equal+weighted): `all_models_global_equal_weighted.csv`")
    # if os.path.exists(f"{out_dir}/all_models_per_query_equal_weighted.csv"):
    #     lines.append(f"- Per-query agg (equal+weighted): `all_models_per_query_equal_weighted.csv`")

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.success("Analysis complete. See '{}' for outputs.", out_dir)


if __name__ == "__main__":
    analyze_and_summarize_outputs(top_n=300)
