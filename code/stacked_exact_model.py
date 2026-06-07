#!/usr/bin/env python3
"""Separate exact-accuracy experiment with current-game, recent-start, and stacked probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier

from paper_ready_analysis import (
    MODEL_CAT,
    MODEL_NUM,
    OUTPUT_DIR,
    STATE_SPECS,
    apply_arsenal_mask,
    arsenal_map,
    available_columns,
    build_arsenal,
    load_features,
    model_frame,
    state_machine_probs,
)
from pitch_sequence_pipeline import ROOT, log


STACK_DIR = OUTPUT_DIR / "stacked_exact"
TABLE_DIR = STACK_DIR / "tables"

EXTRA_CAT = ["if_fielding_alignment", "of_fielding_alignment"]
EXTRA_NUM = [
    "n_thruorder_pitcher",
    "n_priorpa_thisgame_player_at_bat",
    "pitcher_days_since_prev_game",
    "batter_days_since_prev_game",
    "age_pit",
    "age_bat",
]

HGB_PARAMS = {
    "learning_rate": 0.08,
    "max_iter": 160,
    "max_leaf_nodes": 31,
    "l2_regularization": 0.05,
}

XGB_PARAMS = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "n_estimators": 220,
    "max_depth": 4,
    "learning_rate": 0.06,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "reg_lambda": 2.0,
    "reg_alpha": 0.05,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run separate stacked exact-accuracy experiment.")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--half-life-days", type=float, default=45.0)
    parser.add_argument(
        "--skip-final-refit",
        action="store_true",
        help="Evaluate with the base models used to train the stacker instead of refitting on the full 80 percent train split.",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def ordered_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_game_date_dt"] = pd.to_datetime(out["game_date"], errors="coerce")
    out["_original_index"] = np.arange(len(out))
    return out.sort_values(
        ["_game_date_dt", "game_pk", "at_bat_number", "pitch_number", "_original_index"],
        kind="mergesort",
    )


def safe_group_keys(df: pd.DataFrame, group_cols: list[str]) -> list[pd.Series]:
    return [df[col].fillna("missing").astype(str) for col in group_cols]


def add_prior_pitch_mix(df: pd.DataFrame, group_cols: list[str], prefix: str, pitch_types: list[str]) -> pd.DataFrame:
    ordered = ordered_frame(df)
    group_keys = safe_group_keys(ordered, group_cols)
    prior_count = ordered.groupby(group_keys, sort=False).cumcount().astype(float)
    ordered[f"{prefix}_prior_pitches"] = prior_count

    share_cols = []
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
        share_cols.append(share_col)

    shares = ordered[share_cols].to_numpy(dtype=float)
    safe = np.where(shares > 0, shares, 1.0)
    entropy = -(shares * np.log2(safe)).sum(axis=1)
    ordered[f"{prefix}_prior_entropy"] = np.where(prior_count.to_numpy() > 0, entropy, 0.0)
    ordered[f"{prefix}_prior_mode_share"] = np.where(prior_count.to_numpy() > 0, shares.max(axis=1), 0.0)
    cols = [f"{prefix}_prior_pitches", f"{prefix}_prior_entropy", f"{prefix}_prior_mode_share"] + share_cols
    return ordered.sort_values("_original_index")[cols].reset_index(drop=True)


def add_recent_start_features(df: pd.DataFrame, pitch_types: list[str], window: int = 3) -> pd.DataFrame:
    ordered = ordered_frame(df)
    group_cols = ["pitcher_name", "game_pk", "_game_date_dt"]
    counts = ordered.groupby(group_cols + ["pitch_type"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=pitch_types, fill_value=0).reset_index()
    counts["game_pitcher_total_pitches"] = counts[pitch_types].sum(axis=1)
    for pitch in pitch_types:
        counts[f"game_share_{pitch}"] = np.divide(
            counts[pitch],
            counts["game_pitcher_total_pitches"],
            out=np.zeros(len(counts), dtype=float),
            where=counts["game_pitcher_total_pitches"].to_numpy() > 0,
        )

    counts = counts.sort_values(["pitcher_name", "_game_date_dt", "game_pk"], kind="mergesort")
    counts["recent_start_prior_games"] = counts.groupby("pitcher_name").cumcount().clip(upper=window).astype(float)
    counts["recent_start_prior_pitches"] = (
        counts.groupby("pitcher_name")["game_pitcher_total_pitches"]
        .transform(lambda s: s.shift().rolling(window, min_periods=1).sum())
        .fillna(0.0)
    )
    recent_cols = ["recent_start_prior_games", "recent_start_prior_pitches"]
    for pitch in pitch_types:
        col = f"recent_start_prior_{pitch.lower().replace('-', '_')}_share"
        counts[col] = (
            counts.groupby("pitcher_name")[f"game_share_{pitch}"]
            .transform(lambda s: s.shift().rolling(window, min_periods=1).mean())
            .fillna(0.0)
        )
        recent_cols.append(col)

    merge_cols = ["pitcher_name", "game_pk"] + recent_cols
    out = ordered[["pitcher_name", "game_pk", "_original_index"]].merge(counts[merge_cols], on=["pitcher_name", "game_pk"], how="left")
    out[recent_cols] = out[recent_cols].fillna(0.0)
    return out.sort_values("_original_index")[recent_cols].reset_index(drop=True)


def add_experiment_features(df_2025: pd.DataFrame, df_2026: pd.DataFrame, pitch_types: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = df_2025.copy()
    right = df_2026.copy()
    left["_dataset_marker"] = "2025"
    right["_dataset_marker"] = "2026"
    combined = pd.concat([left, right], ignore_index=True)

    blocks = [
        add_prior_pitch_mix(combined, ["pitcher_name", "game_pk"], "current_game_pitcher", pitch_types),
        add_prior_pitch_mix(combined, ["pitcher_name", "game_pk", "stand"], "current_game_pitcher_stand", pitch_types),
        add_prior_pitch_mix(combined, ["pitcher_name"], "pitcher", pitch_types),
        add_prior_pitch_mix(combined, ["batter_name"], "batter", pitch_types),
        add_prior_pitch_mix(combined, ["pitcher_name", "batter_name"], "matchup", pitch_types),
        add_recent_start_features(combined, pitch_types, window=3),
    ]
    enhanced = pd.concat([combined.reset_index(drop=True), *blocks], axis=1)
    for col in [
        "current_game_pitcher_prior_pitches",
        "current_game_pitcher_stand_prior_pitches",
        "pitcher_prior_pitches",
        "batter_prior_pitches",
        "matchup_prior_pitches",
        "recent_start_prior_pitches",
    ]:
        if col in enhanced.columns:
            enhanced[f"{col}_log"] = np.log1p(enhanced[col])

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
    return float(np.mean([actual in choices for actual, choices in zip(y.to_numpy(), top)]))


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


def make_ordinal_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    return ColumnTransformer([("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)])


def make_hgb(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_ordinal_preprocessor(cat_cols, num_cols)),
            ("model", HistGradientBoostingClassifier(random_state=42, **HGB_PARAMS)),
        ]
    )


def fit_xgb(train_rows: pd.DataFrame, feature_cols: list[str], cat_cols: list[str], num_cols: list[str], weights: np.ndarray) -> tuple[XGBClassifier, ColumnTransformer, LabelEncoder]:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(train_rows["pitch_type"])
    preprocessor = make_ordinal_preprocessor(cat_cols, num_cols)
    X_train = preprocessor.fit_transform(train_rows[feature_cols])
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_encoded, sample_weight=weights)
    return model, preprocessor, encoder


def predict_xgb(model: XGBClassifier, preprocessor: ColumnTransformer, encoder: LabelEncoder, rows: pd.DataFrame, feature_cols: list[str], classes: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(preprocessor.transform(rows[feature_cols]))
    model_classes = encoder.inverse_transform(np.arange(len(encoder.classes_)))
    return align_probabilities(raw, model_classes, classes)


def stack_matrix(hgb_probs: np.ndarray, xgb_probs: np.ndarray, state_probs: np.ndarray) -> np.ndarray:
    return np.hstack([hgb_probs, xgb_probs, state_probs])


def fit_base_models(
    rows: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    num_cols: list[str],
    half_life_days: float,
) -> dict[str, Any]:
    weights = recency_weights(rows, half_life_days)
    log(f"  fitting HGB on {len(rows):,} rows")
    hgb = make_hgb(cat_cols, num_cols)
    hgb.fit(rows[feature_cols], rows["pitch_type"], model__sample_weight=weights)
    log(f"  fitting XGBoost on {len(rows):,} rows")
    xgb, xgb_preprocessor, xgb_encoder = fit_xgb(rows, feature_cols, cat_cols, num_cols, weights)
    return {"hgb": hgb, "xgb": xgb, "xgb_preprocessor": xgb_preprocessor, "xgb_encoder": xgb_encoder}


def predict_base_models(models: dict[str, Any], rows: pd.DataFrame, feature_cols: list[str], classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hgb_probs = align_probabilities(models["hgb"].predict_proba(rows[feature_cols]), models["hgb"].classes_, classes)
    xgb_probs = predict_xgb(models["xgb"], models["xgb_preprocessor"], models["xgb_encoder"], rows, feature_cols, classes)
    return hgb_probs, xgb_probs


def add_metrics(
    metrics: list[dict[str, Any]],
    name: str,
    dataset: str,
    classes: np.ndarray,
    probs: np.ndarray,
    rows: pd.DataFrame,
    allowed: dict[str, set[str]],
) -> None:
    metrics.append(evaluate_probs(name, dataset, classes, probs, rows["pitch_type"]))
    masked = apply_arsenal_mask(probs, rows["pitcher_name"], classes, allowed)
    metrics.append(evaluate_probs(f"{name}_arsenal_masked", dataset, classes, masked, rows["pitch_type"]))


def main() -> None:
    args = parse_args()
    ensure_dirs()

    df_2025, df_2026 = load_features(args.top_n)
    base_2025 = model_frame(df_2025)
    keep_pitches = pd.Index(sorted(base_2025["pitch_type"].unique()))
    base_2026 = model_frame(df_2026, keep_pitches) if not df_2026.empty else pd.DataFrame()
    pitch_types = sorted(base_2025["pitch_type"].unique())

    log("Building current-game, historical matchup, and recent-start features...")
    enhanced_2025, enhanced_2026 = add_experiment_features(base_2025, base_2026, pitch_types)

    stratify = enhanced_2025["pitch_type"] if enhanced_2025["pitch_type"].value_counts().min() >= 2 else None
    dev_rows, holdout_rows = train_test_split(enhanced_2025, test_size=0.20, random_state=42, stratify=stratify)
    stratify_dev = dev_rows["pitch_type"] if dev_rows["pitch_type"].value_counts().min() >= 2 else None
    base_train_rows, stack_rows = train_test_split(dev_rows, test_size=0.20, random_state=43, stratify=stratify_dev)
    eval_rows = enhanced_2026.loc[enhanced_2026["pitch_type"].isin(base_train_rows["pitch_type"].unique())].copy()
    classes = np.asarray(sorted(base_train_rows["pitch_type"].unique()))

    cat_cols = available_columns(base_train_rows, MODEL_CAT + EXTRA_CAT)
    dynamic_num = [
        col
        for col in base_train_rows.columns
        if col.startswith("current_game_")
        or col.startswith("pitcher_prior_")
        or col.startswith("batter_prior_")
        or col.startswith("matchup_prior_")
        or col.startswith("recent_start_")
    ]
    num_cols = available_columns(base_train_rows, MODEL_NUM + EXTRA_NUM + dynamic_num)
    feature_cols = cat_cols + num_cols

    log(
        f"Split rows: base_train={len(base_train_rows):,}, stack_train={len(stack_rows):,}, "
        f"holdout={len(holdout_rows):,}, 2026={len(eval_rows):,}; features={len(feature_cols)}"
    )

    log("Building arsenals from the full 80 percent development split...")
    arsenal = build_arsenal(dev_rows)
    allowed = arsenal_map(arsenal)

    metrics: list[dict[str, Any]] = []

    log("Training base models for stacker calibration...")
    calibration_models = fit_base_models(base_train_rows, feature_cols, cat_cols, num_cols, args.half_life_days)
    cal_hgb, cal_xgb = predict_base_models(calibration_models, stack_rows, feature_cols, classes)
    cal_state = state_machine_probs(
        base_train_rows,
        stack_rows,
        classes,
        STATE_SPECS["pitcher_count_prev_stand"],
        alpha=1.0,
        min_count=5,
    )
    stacker = LogisticRegression(C=0.75, max_iter=1000, solver="lbfgs")
    stacker.fit(stack_matrix(cal_hgb, cal_xgb, cal_state), stack_rows["pitch_type"], sample_weight=recency_weights(stack_rows, args.half_life_days))

    final_train_rows = base_train_rows if args.skip_final_refit else dev_rows
    if args.skip_final_refit:
        log("Using calibration base models for final scoring because --skip-final-refit was set.")
        final_models = calibration_models
        state_train_rows = base_train_rows
    else:
        log("Refitting base models on the full 80 percent development split for final scoring...")
        final_models = fit_base_models(final_train_rows, feature_cols, cat_cols, num_cols, args.half_life_days)
        state_train_rows = dev_rows

    eval_sets = [("2025_holdout", holdout_rows)]
    if not eval_rows.empty:
        eval_sets.append(("2026_to_date", eval_rows))

    for dataset, rows in eval_sets:
        log(f"Scoring {dataset} ({len(rows):,} rows)...")
        hgb_probs, xgb_probs = predict_base_models(final_models, rows, feature_cols, classes)
        state_probs = state_machine_probs(
            state_train_rows,
            rows,
            classes,
            STATE_SPECS["pitcher_count_prev_stand"],
            alpha=1.0,
            min_count=5,
        )
        stacked_probs = align_probabilities(
            stacker.predict_proba(stack_matrix(hgb_probs, xgb_probs, state_probs)),
            stacker.classes_,
            classes,
        )
        add_metrics(metrics, "hgb_current_game_recent_start", dataset, classes, hgb_probs, rows, allowed)
        add_metrics(metrics, "xgboost_current_game_recent_start", dataset, classes, xgb_probs, rows, allowed)
        add_metrics(metrics, "state_machine_pitcher_count_prev_stand", dataset, classes, state_probs, rows, allowed)
        add_metrics(metrics, "stacked_hgb_xgb_state", dataset, classes, stacked_probs, rows, allowed)

    metrics_df = pd.DataFrame(metrics).sort_values(["dataset", "exact_accuracy", "top3_accuracy"], ascending=[True, False, False])
    metrics_path = TABLE_DIR / "stacked_current_game_model_comparison.csv"
    metrics_df.to_csv(metrics_path, index=False)

    best_2025 = metrics_df.loc[metrics_df["dataset"].eq("2025_holdout")].iloc[0].to_dict()
    best_2026 = (
        metrics_df.loc[metrics_df["dataset"].eq("2026_to_date")].iloc[0].to_dict()
        if metrics_df["dataset"].eq("2026_to_date").any()
        else None
    )
    summary = {
        "top_n": args.top_n,
        "half_life_days": args.half_life_days,
        "split": {
            "base_train_rows": int(len(base_train_rows)),
            "stack_train_rows": int(len(stack_rows)),
            "holdout_rows": int(len(holdout_rows)),
            "eval_2026_rows": int(len(eval_rows)),
            "final_refit": not args.skip_final_refit,
        },
        "features": {
            "categorical": cat_cols,
            "numeric_count": len(num_cols),
            "dynamic_numeric_count": len(dynamic_num),
        },
        "models": {
            "hgb_params": HGB_PARAMS,
            "xgboost_params": XGB_PARAMS,
            "stacker": {"model": "LogisticRegression", "C": 0.75, "inputs": ["hgb_probs", "xgboost_probs", "state_machine_probs"]},
        },
        "best_2025": best_2025,
        "best_2026": best_2026,
    }
    summary_path = TABLE_DIR / "stacked_current_game_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    log("\n=== Stacked exact model comparison ===")
    print(metrics_df.to_string(index=False), flush=True)
    log("\n=== Files written ===")
    for path in [metrics_path, summary_path]:
        log(str(path.relative_to(ROOT)))


if __name__ == "__main__":
    main()
