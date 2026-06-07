#!/usr/bin/env python3
"""Focused boosted-tree tuning and SHAP interpretation for the top-N pitcher model."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from paper_ready_analysis import (
    PAPER_DIR,
    TABLE_DIR,
    arsenal_map,
    build_arsenal,
    ensure_dirs,
    load_features,
    model_frame,
    set_style,
    shap_interpretation,
    tune_models,
)
from pitch_sequence_pipeline import ROOT, log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune boosted pitch-selection models and regenerate SHAP outputs.")
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_style()

    previous_summary_path = PAPER_DIR / "paper_ready_summary.json"
    previous_summary = json.loads(previous_summary_path.read_text(encoding="utf-8")) if previous_summary_path.exists() else {}

    df_2025, df_2026 = load_features(args.top_n)
    train_2025 = model_frame(df_2025)
    keep_pitches = pd.Index(sorted(train_2025["pitch_type"].unique()))
    eval_2026 = model_frame(df_2026, keep_pitches) if not df_2026.empty else pd.DataFrame()

    stratify = train_2025["pitch_type"] if train_2025["pitch_type"].value_counts().min() >= 2 else None
    train_rows, test_rows = train_test_split(
        train_2025,
        test_size=0.20,
        random_state=42,
        stratify=stratify,
    )
    classes = np.asarray(sorted(train_rows["pitch_type"].unique()))
    top_targets = train_rows["pitch_type"].value_counts().head(5).index.tolist()

    log("Building pitcher arsenals for arsenal-masked probability evaluation...")
    arsenal = build_arsenal(train_rows)
    allowed = arsenal_map(arsenal)

    log("Running expanded boosted-tree hyperparameter grid...")
    model_grid = tune_models(train_rows, test_rows, classes, allowed, eval_2026)

    log("Regenerating SHAP interpretation tables and figures...")
    shap_df = shap_interpretation(train_rows, test_rows, top_targets[:4])
    shap_global = pd.read_csv(TABLE_DIR / "shap_global_feature_importance.csv")

    best_2025 = model_grid.loc[model_grid["dataset"].eq("2025_holdout")].iloc[0].to_dict()
    best_2026 = None
    if model_grid["dataset"].eq("2026_to_date").any():
        best_2026 = model_grid.loc[model_grid["dataset"].eq("2026_to_date")].iloc[0].to_dict()

    summary = {
        "top_n": args.top_n,
        "n_2025_rows": int(len(train_2025)),
        "n_2026_rows": int(len(eval_2026)) if not eval_2026.empty else 0,
        "previous_best_pooled_model": previous_summary.get("best_pooled_model"),
        "previous_best_2026_temporal_model": previous_summary.get("best_2026_temporal_model"),
        "new_best_2025_boosted_model": best_2025,
        "new_best_2026_boosted_model": best_2026,
        "top_global_shap_features": shap_global.head(15).to_dict(orient="records"),
        "top_pitch_type_shap_features": shap_df.groupby("target_pitch_type", group_keys=False)
        .head(8)
        .to_dict(orient="records"),
    }
    out_path = PAPER_DIR / "boosted_tuning_shap_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    log("\n=== Best expanded-grid boosted models ===")
    print(
        model_grid.groupby("dataset", group_keys=False)
        .head(8)
        .to_string(index=False),
        flush=True,
    )
    log("\n=== Top global SHAP features ===")
    print(shap_global.head(15).to_string(index=False), flush=True)
    log("\n=== Files written ===")
    for path in [
        TABLE_DIR / "pooled_model_tuning_grid.csv",
        TABLE_DIR / "shap_pitch_type_feature_importance.csv",
        TABLE_DIR / "shap_global_feature_importance.csv",
        PAPER_DIR / "boosted_tuning_shap_summary.json",
    ]:
        log(str(path.relative_to(ROOT)))


if __name__ == "__main__":
    main()
