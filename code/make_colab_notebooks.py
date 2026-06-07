#!/usr/bin/env python3
"""Generate numbered Colab notebooks for the public pitch-sequencing repo."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(text).strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    source = textwrap.dedent(text).strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


STARTER = code(
    """
    # Colab/local setup. Keep helper functions at the top of every notebook.
    import os
    import sys
    import subprocess
    from pathlib import Path

    IN_COLAB = "google.colab" in sys.modules
    REPO_URL = ""  # Optional: set after the public GitHub repo exists.
    TOP_N = 100

    if IN_COLAB:
        %pip -q install -r requirements.txt || %pip -q install pybaseball pandas numpy scikit-learn matplotlib seaborn pyarrow shap statsmodels xgboost tqdm
        if REPO_URL and not Path("/content/pitch-sequencing").exists():
            !git clone {REPO_URL} /content/pitch-sequencing
        BASE_DIR = Path("/content/pitch-sequencing") if Path("/content/pitch-sequencing").exists() else Path.cwd()
    else:
        BASE_DIR = Path.cwd()

    os.chdir(BASE_DIR)
    sys.path.insert(0, str(BASE_DIR / "code"))

    DATA_DIR = BASE_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    OUTPUT_DIR = BASE_DIR / "output"

    for path in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, OUTPUT_DIR / "figures"]:
        path.mkdir(parents=True, exist_ok=True)

    def run_step(args):
        print("Running:", " ".join(map(str, args)))
        result = subprocess.run(args, cwd=BASE_DIR, text=True, capture_output=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        result.check_returncode()

    def frame_diag(df, label):
        print(f"{label}: rows={len(df):,}, cols={df.shape[1]:,}")
        if "pitcher_name" in df:
            print(f"{label}: pitchers={df['pitcher_name'].nunique():,}")
        if "pitch_type" in df:
            print(f"{label}: pitch_types={df['pitch_type'].nunique():,}")
            print(df["pitch_type"].value_counts().head(12))
        return df.head()

    def merge_diag(left, right, keys, label):
        print(f"[merge:{label}] before left_rows={len(left):,}, right_rows={len(right):,}, keys={keys}")
        print(f"[merge:{label}] right_duplicate_keys={right.duplicated(keys).sum():,}")
        merged = left.merge(right, on=keys, how="left", validate="many_to_one", indicator=True)
        print(f"[merge:{label}] after rows={len(merged):,}, unmatched={merged['_merge'].eq('left_only').sum():,}")
        return merged.drop(columns=["_merge"])

    def show_csv(path, rows=8):
        import pandas as pd
        path = BASE_DIR / path
        print(path)
        df = pd.read_csv(path)
        display(df.head(rows))
        return df
    """
)


NOTEBOOKS = {
    "00_pull.ipynb": [
        md(
            """
            # 00 Pull Raw Statcast Data

            Pull the 2025 top-100 pitcher panel from the Statcast arsenal leaderboard and cache raw pitch-level data.
            This notebook is intentionally about inputs and coverage, not modeling.
            """
        ),
        STARTER,
        code(
            """
            from pitch_sequence_pipeline import (
                cache,
                ensure_dirs,
                pull_pitcher_data,
                select_pitchers,
            )

            ensure_dirs()
            cache.enable()

            pitchers = select_pitchers(2025, TOP_N)
            frame_diag(pitchers, "selected_pitchers")
            display(pitchers.head(15))

            raw_2025 = pull_pitcher_data(
                season=2025,
                start_date="2025-03-18",
                end_date="2025-09-28",
                pitchers=pitchers,
                raw_label="statcast",
            )
            frame_diag(raw_2025, "raw_2025")
            """
        ),
        md("Outputs: `output/top_100_pitchers_2025.csv` and cached pitcher parquet files in `data/raw/`."),
    ],
    "01_merge.ipynb": [
        md(
            """
            # 01 Merge Names And Engineer Sequence Features

            Build the modeling table. The pipeline prints diagnostics before and after the lineup-handedness merge.
            """
        ),
        STARTER,
        code(
            """
            from pitch_sequence_pipeline import (
                add_player_names,
                build_sequence_features,
                cache,
                ensure_dirs,
                pull_pitcher_data,
                sequence_features_path,
            )
            import pandas as pd

            ensure_dirs()
            cache.enable()

            pitchers = pd.read_csv(OUTPUT_DIR / f"top_{TOP_N}_pitchers_2025.csv")
            raw_2025 = pull_pitcher_data(2025, "2025-03-18", "2025-09-28", pitchers, raw_label="statcast")
            frame_diag(raw_2025, "raw_before_name_merge")

            named_2025 = add_player_names(raw_2025, pitchers)
            frame_diag(named_2025, "named_after_player_mapping")

            features_2025 = build_sequence_features(named_2025, output_path=sequence_features_path(2025, TOP_N))
            frame_diag(features_2025, "features_2025")
            display(features_2025[["game_date", "pitcher_name", "batter_name", "stand", "lineup_left_share", "count", "prev_pitch_type", "pitch_type"]].head())
            """
        ),
        md("Outputs: `data/processed/sequence_features_2025_top100.parquet` plus batter lookup tables."),
    ],
    "02_state_machine_ols.ipynb": [
        md(
            """
            # 02 State Machine, OLS Betas, And Per-Pitcher Models

            This is the professor-aligned core notebook: state transitions by pitcher/batter/count/previous pitch, OLS beta tables, then the per-pitcher boosted loop compared with one pooled model.
            """
        ),
        STARTER,
        code(
            """
            run_step([sys.executable, "code/state_machine_and_ols.py", "--top-n", str(TOP_N)])

            metrics = show_csv("output/state_ols_model_comparison_metrics.csv")
            betas = show_csv("output/ols_beta_pitcher_pitch_intercepts.csv")
            beta_summary = show_csv("output/ols_beta_feature_summary.csv")
            transitions = show_csv("output/state_machine_top3_transitions.csv")
            """
        ),
        md(
            """
            Key outputs:
            `state_machine_transitions.csv`,
            `state_machine_top3_transitions.csv`,
            `ols_pitch_type_coefficients_long.csv`,
            `ols_beta_pitcher_pitch_intercepts.csv`,
            `ols_beta_pitcher_pitch_mean_abs_context.csv`,
            `state_ols_model_comparison_metrics.csv`.
            """
        ),
    ],
    "03_temporal_model_sweep.ipynb": [
        md(
            """
            # 03 Temporal Validation And Model Sweep

            Pull same-pitcher 2026 rows when available, then compare empirical lookup, logistic, random forest, and boosted models.
            """
        ),
        STARTER,
        code(
            """
            run_step([sys.executable, "code/pitch_sequence_extension.py", "--top-n", str(TOP_N)])
            run_step([sys.executable, "code/pitch_choice_diagnostics.py", "--top-n", str(TOP_N)])

            temporal = show_csv("output/temporal_validation_2025_to_2026.csv")
            sweep = show_csv("output/next_pitch_model_sweep_metrics.csv")
            accuracy = show_csv("output/prediction_accuracy_by_pitcher.csv")
            """
        ),
        md(
            """
            Outputs include temporal validation metrics, prediction audits, calibration files, and model-sweep figures.
            The per-pitcher state-machine/XGBoost comparison remains the primary model family; this notebook supplies robustness checks.
            """
        ),
    ],
    "04_paper_outputs.ipynb": [
        md(
            """
            # 04 Paper-Ready Outputs And Course Mapping

            Generate final tables/figures for the written paper and connect the analysis to regression, state machines, boosted trees, SHAP, validation, and agentic reproducibility.
            """
        ),
        STARTER,
        code(
            """
            run_step([sys.executable, "code/paper_ready_analysis.py", "--top-n", str(TOP_N)])

            summary = BASE_DIR / "output/paper_ready/paper_ready_summary.json"
            print(summary.read_text())
            show_csv("output/paper_ready/tables/paper_model_comparison_top_models.csv")
            show_csv("output/paper_ready/tables/ols_pooled_fixed_effects_coefficients.csv")
            show_csv("output/paper_ready/tables/shap_pitch_type_feature_importance.csv")
            """
        ),
        md(
            """
            Final deliverables live in `output/paper_ready/`.
            Use these for the paper's results section and appendix tables.
            """
        ),
    ],
}


def write_notebook(path: Path, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "colab": {"name": path.name},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    for filename, cells in NOTEBOOKS.items():
        write_notebook(ROOT / filename, cells)


if __name__ == "__main__":
    main()
