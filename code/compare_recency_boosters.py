#!/usr/bin/env python3
"""Compare recency-weighted HGB, XGBoost, and CatBoost pitch-selection models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

from paper_ready_analysis import (
    MODEL_CAT,
    MODEL_NUM,
    OUTPUT_DIR,
    PAPER_DIR,
    TABLE_DIR,
    arsenal_map,
    apply_arsenal_mask,
    available_columns,
    build_arsenal,
    ensure_dirs,
    load_features,
    model_frame,
    pitch_label,
    set_style,
)
from pitch_sequence_pipeline import ROOT, log


FIGURE_DIR = PAPER_DIR / "figures"

EXTRA_CAT = ["if_fielding_alignment", "of_fielding_alignment"]
EXTRA_NUM = [
    "n_thruorder_pitcher",
    "n_priorpa_thisgame_player_at_bat",
    "pitcher_days_since_prev_game",
    "batter_days_since_prev_game",
    "age_pit",
    "age_bat",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recency-weighted CatBoost/XGBoost comparison.")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--half-life-days", type=float, default=45.0)
    parser.add_argument("--skip-catboost", action="store_true", help="Stop after HGB and XGBoost comparisons.")
    return parser.parse_args()


def ordered_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_game_date_dt"] = pd.to_datetime(out["game_date"], errors="coerce")
    out["_original_index"] = np.arange(len(out))
    order_cols = [
        "_game_date_dt",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "_original_index",
    ]
    return out.sort_values(order_cols, kind="mergesort")


def add_prior_pitch_mix(
    df: pd.DataFrame,
    group_cols: list[str],
    prefix: str,
    pitch_types: list[str],
) -> pd.DataFrame:
    ordered = ordered_frame(df)
    group_keys = [ordered[col].fillna("missing").astype(str) for col in group_cols]
    prior_count = ordered.groupby(group_keys, sort=False).cumcount().astype(float)
    ordered[f"{prefix}_prior_pitches"] = prior_count

    shares = []
    for pitch in pitch_types:
        clean_pitch = pitch.lower().replace("-", "_")
        current = ordered["pitch_type"].eq(pitch).astype(float)
        prior_pitch_count = current.groupby(group_keys, sort=False).cumsum() - current
        share_col = f"{prefix}_prior_{clean_pitch}_share"
        ordered[share_col] = np.divide(
            prior_pitch_count,
            prior_count,
            out=np.zeros(len(ordered), dtype=float),
            where=prior_count.to_numpy() > 0,
        )
        shares.append(share_col)

    share_matrix = ordered[shares].to_numpy(dtype=float)
    safe = np.where(share_matrix > 0, share_matrix, 1.0)
    entropy = -(share_matrix * np.log2(safe)).sum(axis=1)
    ordered[f"{prefix}_prior_entropy"] = np.where(prior_count.to_numpy() > 0, entropy, 0.0)
    ordered[f"{prefix}_prior_mode_share"] = np.where(
        prior_count.to_numpy() > 0,
        share_matrix.max(axis=1),
        0.0,
    )

    cols = [f"{prefix}_prior_pitches", f"{prefix}_prior_entropy", f"{prefix}_prior_mode_share"] + shares
    return ordered.sort_values("_original_index")[cols].reset_index(drop=True)


def add_matchup_features(df_2025: pd.DataFrame, df_2026: pd.DataFrame, pitch_types: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = df_2025.copy()
    right = df_2026.copy()
    left["_dataset_marker"] = "2025"
    right["_dataset_marker"] = "2026"
    combined = pd.concat([left, right], ignore_index=True)

    feature_blocks = [
        add_prior_pitch_mix(combined, ["pitcher_name"], "pitcher", pitch_types),
        add_prior_pitch_mix(combined, ["batter_name"], "batter", pitch_types),
        add_prior_pitch_mix(combined, ["pitcher_name", "batter_name"], "matchup", pitch_types),
    ]
    features = pd.concat(feature_blocks, axis=1)
    enhanced = pd.concat([combined.reset_index(drop=True), features], axis=1)

    enhanced["matchup_prior_log_pitches"] = np.log1p(enhanced["matchup_prior_pitches"])
    enhanced["pitcher_prior_log_pitches"] = np.log1p(enhanced["pitcher_prior_pitches"])
    enhanced["batter_prior_log_pitches"] = np.log1p(enhanced["batter_prior_pitches"])

    enhanced_2025 = enhanced.loc[enhanced["_dataset_marker"].eq("2025")].drop(columns=["_dataset_marker"]).reset_index(drop=True)
    enhanced_2026 = enhanced.loc[enhanced["_dataset_marker"].eq("2026")].drop(columns=["_dataset_marker"]).reset_index(drop=True)
    return enhanced_2025, enhanced_2026


def recency_weights(rows: pd.DataFrame, half_life_days: float) -> np.ndarray:
    dates = pd.to_datetime(rows["game_date"], errors="coerce")
    max_date = dates.max()
    age_days = (max_date - dates).dt.days.fillna(0).clip(lower=0)
    weights = np.power(0.5, age_days / half_life_days).to_numpy(dtype=float)
    return weights / weights.mean()


def top_k_accuracy(classes: np.ndarray, probs: np.ndarray, y: pd.Series, k: int) -> float:
    order = np.argsort(probs, axis=1)[:, -min(k, len(classes)) :]
    top = classes[order]
    return float(np.mean([actual in row for actual, row in zip(y.to_numpy(), top)]))


def evaluate_probs(model: str, dataset: str, classes: np.ndarray, probs: np.ndarray, y: pd.Series) -> dict[str, Any]:
    pred = classes[np.argmax(probs, axis=1)]
    class_to_idx = {pitch: idx for idx, pitch in enumerate(classes)}
    actual_prob = np.array([probs[i, class_to_idx[pitch]] if pitch in class_to_idx else 0.0 for i, pitch in enumerate(y)])
    return {
        "model": model,
        "dataset": dataset,
        "rows": int(len(y)),
        "exact_accuracy": float(accuracy_score(y, pred)),
        "top2_accuracy": top_k_accuracy(classes, probs, y, 2),
        "top3_accuracy": top_k_accuracy(classes, probs, y, 3),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "avg_probability_actual_pitch": float(actual_prob.mean()),
        "avg_model_confidence": float(probs.max(axis=1).mean()),
        "log_loss": float(log_loss(y, probs, labels=list(classes))) if y.nunique() > 1 else np.nan,
    }


def align_probabilities(raw: np.ndarray, model_classes: np.ndarray, classes: np.ndarray) -> np.ndarray:
    out = np.zeros((raw.shape[0], len(classes)), dtype=float)
    local_idx = {str(pitch): i for i, pitch in enumerate(model_classes)}
    for j, pitch in enumerate(classes):
        if str(pitch) in local_idx:
            out[:, j] = raw[:, local_idx[str(pitch)]]
    row_sum = out.sum(axis=1)
    missing = row_sum <= 0
    out[missing] = 1 / len(classes)
    out[~missing] = out[~missing] / row_sum[~missing, None]
    return out


def make_hgb(cat_cols: list[str], num_cols: list[str], params: dict[str, Any]) -> Pipeline:
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    pre = ColumnTransformer([("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)])
    return Pipeline(
        [
            ("preprocess", pre),
            ("model", HistGradientBoostingClassifier(random_state=42, **params)),
        ]
    )


def catboost_frame(df: pd.DataFrame, cat_cols: list[str], num_cols: list[str]) -> pd.DataFrame:
    out = df[cat_cols + num_cols].copy()
    for col in cat_cols:
        out[col] = out[col].astype("string").fillna("missing")
    for col in num_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def make_ordinal_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    return ColumnTransformer([("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)])


def score_model(
    metrics: list[dict[str, Any]],
    name: str,
    dataset: str,
    classes: np.ndarray,
    probs: np.ndarray,
    y: pd.Series,
    pitcher_names: pd.Series,
    allowed: dict[str, set[str]],
) -> None:
    metrics.append(evaluate_probs(name, dataset, classes, probs, y))
    masked = apply_arsenal_mask(probs, pitcher_names, classes, allowed)
    metrics.append(evaluate_probs(f"{name}_arsenal_masked", dataset, classes, masked, y))


def write_metrics_checkpoint(metrics: list[dict[str, Any]]) -> pd.DataFrame:
    metrics_df = pd.DataFrame(metrics).sort_values(
        ["dataset", "exact_accuracy", "top3_accuracy"],
        ascending=[True, False, False],
    )
    out_path = TABLE_DIR / "recency_booster_model_comparison.csv"
    metrics_df.to_csv(out_path, index=False)
    return metrics_df


def plot_comparison(metrics: pd.DataFrame) -> None:
    subset = metrics.loc[metrics["model"].str.endswith("arsenal_masked")].copy()
    order = (
        subset.loc[subset["dataset"].eq("2025_holdout")]
        .sort_values("exact_accuracy", ascending=False)["model"]
        .drop_duplicates()
        .tolist()
    )
    subset["model"] = pd.Categorical(subset["model"], categories=order, ordered=True)
    plot_df = subset.melt(
        id_vars=["model", "dataset"],
        value_vars=["exact_accuracy", "top3_accuracy"],
        var_name="metric",
        value_name="score",
    )
    plot_df["metric"] = plot_df["metric"].map({"exact_accuracy": "Exact accuracy", "top3_accuracy": "Top-3 accuracy"})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for ax, metric in zip(axes, ["Exact accuracy", "Top-3 accuracy"]):
        sub = plot_df.loc[plot_df["metric"].eq(metric)]
        sns.barplot(data=sub, x="score", y="model", hue="dataset", ax=ax, errorbar=None)
        ax.set_title(metric)
        ax.set_xlabel("Score")
        ax.set_ylabel("")
        ax.set_xlim(0, 1)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Recency-weighted matchup models", y=1.02, fontweight="bold")
    plt.tight_layout()
    out = FIGURE_DIR / "fig7_recency_booster_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_style()

    df_2025, df_2026 = load_features(args.top_n)
    base_2025 = model_frame(df_2025)
    keep_pitches = pd.Index(sorted(base_2025["pitch_type"].unique()))
    base_2026 = model_frame(df_2026, keep_pitches) if not df_2026.empty else pd.DataFrame()
    pitch_types = sorted(base_2025["pitch_type"].unique())

    log("Building chronological prior pitcher, batter, and matchup features...")
    enhanced_2025, enhanced_2026 = add_matchup_features(base_2025, base_2026, pitch_types)

    stratify = enhanced_2025["pitch_type"] if enhanced_2025["pitch_type"].value_counts().min() >= 2 else None
    train_rows, test_rows = train_test_split(
        enhanced_2025,
        test_size=0.20,
        random_state=42,
        stratify=stratify,
    )
    eval_rows = enhanced_2026.loc[enhanced_2026["pitch_type"].isin(train_rows["pitch_type"].unique())].copy()
    classes = np.asarray(sorted(train_rows["pitch_type"].unique()))

    cat_cols = available_columns(train_rows, MODEL_CAT + EXTRA_CAT)
    matchup_num = [
        col
        for col in train_rows.columns
        if col.startswith("pitcher_prior_")
        or col.startswith("batter_prior_")
        or col.startswith("matchup_prior_")
    ]
    num_cols = available_columns(train_rows, MODEL_NUM + EXTRA_NUM + matchup_num)
    feature_cols = cat_cols + num_cols

    weights = recency_weights(train_rows, args.half_life_days)
    log(
        f"Training rows={len(train_rows):,}; holdout rows={len(test_rows):,}; "
        f"2026 rows={len(eval_rows):,}; features={len(feature_cols)}; half_life={args.half_life_days:g} days"
    )

    log("Building pitcher arsenals for masked probability comparison...")
    arsenal = build_arsenal(train_rows)
    allowed = arsenal_map(arsenal)

    metrics: list[dict[str, Any]] = []

    hgb_params = {"learning_rate": 0.08, "max_iter": 160, "max_leaf_nodes": 31, "l2_regularization": 0.05}
    log(f"Training HGB baseline with current context features: {hgb_params}")
    base_cat = available_columns(train_rows, MODEL_CAT)
    base_num = available_columns(train_rows, MODEL_NUM)
    hgb_base = make_hgb(base_cat, base_num, hgb_params)
    hgb_base.fit(train_rows[base_cat + base_num], train_rows["pitch_type"])
    score_model(
        metrics,
        "hgb_current_features",
        "2025_holdout",
        classes,
        align_probabilities(hgb_base.predict_proba(test_rows[base_cat + base_num]), hgb_base.classes_, classes),
        test_rows["pitch_type"],
        test_rows["pitcher_name"],
        allowed,
    )
    if not eval_rows.empty:
        score_model(
            metrics,
            "hgb_current_features",
            "2026_to_date",
            classes,
            align_probabilities(hgb_base.predict_proba(eval_rows[base_cat + base_num]), hgb_base.classes_, classes),
            eval_rows["pitch_type"],
            eval_rows["pitcher_name"],
            allowed,
        )
    write_metrics_checkpoint(metrics)

    log("Training recency-weighted HGB with matchup features...")
    hgb_matchup = make_hgb(cat_cols, num_cols, hgb_params)
    hgb_matchup.fit(train_rows[feature_cols], train_rows["pitch_type"], model__sample_weight=weights)
    score_model(
        metrics,
        "hgb_recency_matchup",
        "2025_holdout",
        classes,
        align_probabilities(hgb_matchup.predict_proba(test_rows[feature_cols]), hgb_matchup.classes_, classes),
        test_rows["pitch_type"],
        test_rows["pitcher_name"],
        allowed,
    )
    if not eval_rows.empty:
        score_model(
            metrics,
            "hgb_recency_matchup",
            "2026_to_date",
            classes,
            align_probabilities(hgb_matchup.predict_proba(eval_rows[feature_cols]), hgb_matchup.classes_, classes),
            eval_rows["pitch_type"],
            eval_rows["pitcher_name"],
            allowed,
        )
    write_metrics_checkpoint(metrics)

    log("Training recency-weighted XGBoost with robust ordinal-encoded categorical features...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(train_rows["pitch_type"])
    xgb = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=240,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=2.0,
        reg_alpha=0.05,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    xgb_preprocessor = make_ordinal_preprocessor(cat_cols, num_cols)
    X_train_xgb = xgb_preprocessor.fit_transform(train_rows[feature_cols])
    xgb.fit(X_train_xgb, y_train_encoded, sample_weight=weights)
    xgb_classes = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))
    score_model(
        metrics,
        "xgboost_recency_matchup",
        "2025_holdout",
        classes,
        align_probabilities(xgb.predict_proba(xgb_preprocessor.transform(test_rows[feature_cols])), xgb_classes, classes),
        test_rows["pitch_type"],
        test_rows["pitcher_name"],
        allowed,
    )
    if not eval_rows.empty:
        score_model(
            metrics,
            "xgboost_recency_matchup",
            "2026_to_date",
            classes,
            align_probabilities(xgb.predict_proba(xgb_preprocessor.transform(eval_rows[feature_cols])), xgb_classes, classes),
            eval_rows["pitch_type"],
            eval_rows["pitcher_name"],
            allowed,
        )
    write_metrics_checkpoint(metrics)

    if not args.skip_catboost:
        log("Training recency-weighted CatBoost with categorical features...")
        cat_model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=120,
            depth=5,
            learning_rate=0.10,
            l2_leaf_reg=8.0,
            boosting_type="Plain",
            random_seed=42,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        )
        X_train_cat = catboost_frame(train_rows, cat_cols, num_cols)
        cat_model.fit(X_train_cat, train_rows["pitch_type"], cat_features=cat_cols, sample_weight=weights)
        score_model(
            metrics,
            "catboost_recency_matchup",
            "2025_holdout",
            classes,
            align_probabilities(
                cat_model.predict_proba(catboost_frame(test_rows, cat_cols, num_cols)),
                cat_model.classes_,
                classes,
            ),
            test_rows["pitch_type"],
            test_rows["pitcher_name"],
            allowed,
        )
        if not eval_rows.empty:
            score_model(
                metrics,
                "catboost_recency_matchup",
                "2026_to_date",
                classes,
                align_probabilities(
                    cat_model.predict_proba(catboost_frame(eval_rows, cat_cols, num_cols)),
                    cat_model.classes_,
                    classes,
                ),
                eval_rows["pitch_type"],
                eval_rows["pitcher_name"],
                allowed,
            )
        write_metrics_checkpoint(metrics)
    else:
        log("Skipping CatBoost by request.")

    metrics_df = write_metrics_checkpoint(metrics)
    out_path = TABLE_DIR / "recency_booster_model_comparison.csv"
    plot_comparison(metrics_df)

    best_2025 = metrics_df.loc[metrics_df["dataset"].eq("2025_holdout")].iloc[0].to_dict()
    best_2026 = (
        metrics_df.loc[metrics_df["dataset"].eq("2026_to_date")].iloc[0].to_dict()
        if metrics_df["dataset"].eq("2026_to_date").any()
        else None
    )
    summary = {
        "top_n": args.top_n,
        "half_life_days": args.half_life_days,
        "features": {
            "categorical": cat_cols,
            "numeric_count": len(num_cols),
            "matchup_numeric_count": len(matchup_num),
        },
        "models": {
            "hgb_params": hgb_params,
            "xgboost_params": {
                "n_estimators": 240,
                "max_depth": 4,
                "learning_rate": 0.06,
                "subsample": 0.90,
                "colsample_bytree": 0.90,
                "reg_lambda": 2.0,
                "reg_alpha": 0.05,
            },
            "catboost_params": {
                "iterations": 120,
                "depth": 5,
                "learning_rate": 0.10,
                "l2_leaf_reg": 8.0,
                "boosting_type": "Plain",
            },
        },
        "best_2025": best_2025,
        "best_2026": best_2026,
    }
    summary_path = PAPER_DIR / "recency_booster_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    log("\n=== Recency booster comparison ===")
    print(metrics_df.to_string(index=False), flush=True)
    log("\n=== Best models ===")
    print(json.dumps({"best_2025": best_2025, "best_2026": best_2026}, indent=2, default=str), flush=True)
    log("\n=== Files written ===")
    for path in [out_path, summary_path, FIGURE_DIR / "fig7_recency_booster_comparison.png"]:
        log(str(path.relative_to(ROOT)))


if __name__ == "__main__":
    main()
