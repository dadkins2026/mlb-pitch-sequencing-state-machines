#!/usr/bin/env python3
"""Add prediction-vs-actual visualizations and 2026 temporal validation."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("PYBASEBALL_CACHE", str(Path(".cache/pybaseball").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from pitch_sequence_pipeline import (
    PROCESSED_DIR,
    RAW_DIR,
    ROOT,
    add_player_names,
    build_sequence_features,
    cache,
    log,
    make_hgb_classifier,
    make_hgb_regressor,
    make_next_pitch_model,
    pitch_label,
    pull_pitcher_data,
    sequence_features_path,
    temporal_raw_label,
    temporal_sequence_features_path,
    top_k_accuracy,
)


OUTPUT_DIR = ROOT / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"


NEXT_CAT = [
    "pitcher_name",
    "batter_name",
    "stand",
    "p_throws",
    "count",
    "base_state",
    "prev_pitch_type",
    "prev2_pitch_type",
    "prev_description",
    "inning_topbot",
]

NEXT_NUM = [
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "pitch_number",
    "prev_zone",
    "prev_release_speed",
    "prev_plate_x",
    "prev_plate_z",
    "score_diff_batting_team",
    "lineup_left_share",
    "lineup_right_share",
    "lineup_switch_share",
    "lineup_batter_count",
]

PITCH_MEASURE_NUM = ["release_speed", "release_spin_rate", "zone", "plate_x", "plate_z", "pfx_x", "pfx_z"]


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    log(f"wrote {path.relative_to(ROOT)}")


def load_2025_features(top_n: int) -> pd.DataFrame:
    path = sequence_features_path(2025, top_n)
    if not path.exists():
        raise FileNotFoundError(
            f"Run code/pitch_sequence_pipeline.py first; missing {path.relative_to(ROOT)}"
        )
    return pd.read_parquet(path)


def load_2025_pitchers(top_n: int) -> pd.DataFrame:
    path = OUTPUT_DIR / f"top_{top_n}_pitchers_2025.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Run code/pitch_sequence_pipeline.py first; missing output/top_{top_n}_pitchers_2025.csv"
        )
    return pd.read_csv(path)


def pull_or_load_2026_features(pitchers: pd.DataFrame, top_n: int, start_date: str, end_date: str) -> pd.DataFrame:
    out_path = temporal_sequence_features_path(2025, 2026, top_n)
    if out_path.exists():
        log(f"Using cached 2026 feature table: {out_path.relative_to(ROOT)}")
        return pd.read_parquet(out_path)

    log(f"Pulling 2026 Statcast rows for the same 2025 top-{top_n} pitchers ({start_date} through {end_date})...")
    raw = pull_pitcher_data(
        season=2026,
        start_date=start_date,
        end_date=end_date,
        pitchers=pitchers,
        raw_label=temporal_raw_label(2025, top_n),
    )
    named = add_player_names(raw, pitchers, lookup_path=PROCESSED_DIR / "batter_lookup_2026.csv")
    return build_sequence_features(named, output_path=out_path)


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def model_frame(
    df: pd.DataFrame,
    keep_pitch_types: pd.Index | None = None,
    min_pitch_type_count: int = 1,
) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    if keep_pitch_types is None:
        counts = out["pitch_type"].value_counts()
        keep_pitch_types = counts.loc[counts >= min_pitch_type_count].index
    return out.loc[out["pitch_type"].isin(keep_pitch_types)].copy()


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    next_cat = available_columns(df, NEXT_CAT)
    next_num = available_columns(df, NEXT_NUM)
    outcome_cat = available_columns(df, NEXT_CAT + ["pitch_type"])
    outcome_num = available_columns(df, NEXT_NUM + PITCH_MEASURE_NUM)
    return {
        "next_cat": next_cat,
        "next_num": next_num,
        "next_features": next_cat + next_num,
        "outcome_cat": outcome_cat,
        "outcome_num": outcome_num,
        "outcome_features": outcome_cat + outcome_num,
    }


def safe_auc(y_true: pd.Series, y_prob: np.ndarray) -> float | None:
    if pd.Series(y_true).nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def evaluate_next_pitch(model: Any, X: pd.DataFrame, y: pd.Series, label: str) -> dict[str, Any]:
    mask = y.isin(model.classes_)
    X = X.loc[mask]
    y = y.loc[mask]
    pred = model.predict(X)
    probs = model.predict_proba(X)
    class_to_idx = {pitch: idx for idx, pitch in enumerate(model.classes_)}
    actual_prob = np.array([probs[i, class_to_idx[pitch]] for i, pitch in enumerate(y)])
    return {
        "dataset": label,
        "task": "next_pitch",
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "top2_accuracy": top_k_accuracy(model, X, y, k=2),
        "top3_accuracy": top_k_accuracy(model, X, y, k=3),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "mean_probability_assigned_to_actual_pitch": float(actual_prob.mean()),
        "mean_model_confidence": float(probs.max(axis=1).mean()),
    }


def prediction_audit(model: Any, X: pd.DataFrame, y: pd.Series, rows: pd.DataFrame, label: str) -> pd.DataFrame:
    mask = y.isin(model.classes_)
    X = X.loc[mask]
    y = y.loc[mask]
    rows = rows.loc[y.index].copy()

    probs = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    order = np.argsort(probs, axis=1)[:, ::-1]
    predicted = classes[order[:, 0]]
    class_to_idx = {pitch: idx for idx, pitch in enumerate(classes)}
    actual_idx = np.array([class_to_idx[pitch] for pitch in y])

    top2 = [actual_idx[i] in order[i, :2] for i in range(len(y))]
    top3 = [actual_idx[i] in order[i, :3] for i in range(len(y))]

    audit_cols = [
        "game_date",
        "pitcher_name",
        "batter_name",
        "stand",
        "p_throws",
        "count",
        "balls",
        "strikes",
        "outs_when_up",
        "inning",
        "inning_topbot",
        "base_state",
        "prev_pitch_type",
        "prev2_pitch_type",
        "prev_description",
        "description",
        "events",
    ]
    out = rows[[col for col in audit_cols if col in rows.columns]].copy()
    out.insert(0, "dataset", label)
    out["actual_pitch"] = y.to_numpy()
    out["predicted_pitch"] = predicted
    out["actual_pitch_label"] = out["actual_pitch"].map(pitch_label)
    out["predicted_pitch_label"] = out["predicted_pitch"].map(pitch_label)
    out["correct_top1"] = out["actual_pitch"].eq(out["predicted_pitch"])
    out["correct_top2"] = top2
    out["correct_top3"] = top3
    out["model_confidence"] = probs.max(axis=1)
    out["actual_pitch_probability"] = probs[np.arange(len(y)), actual_idx]
    out["top_3_predictions"] = [
        ", ".join(f"{pitch_label(classes[j])} {probs[i, j]:.1%}" for j in order[i, :3])
        for i in range(len(y))
    ]
    return out.reset_index(drop=True)


def evaluate_binary_outcome(model: Any, X: pd.DataFrame, y: pd.Series, label: str) -> dict[str, Any]:
    y = y.astype(int)
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "dataset": label,
        "task": "whiff_or_weak",
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "auc": safe_auc(y, prob),
        "brier": float(brier_score_loss(y, prob)),
        "positive_rate": float(y.mean()),
    }


def evaluate_run_value(model: Any, X: pd.DataFrame, y: pd.Series, label: str) -> dict[str, Any]:
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    if y.empty:
        return {
            "dataset": label,
            "task": "pitcher_run_value",
            "rows": 0,
            "rmse": None,
            "r2": None,
            "target_mean": None,
        }
    pred = model.predict(X)
    return {
        "dataset": label,
        "task": "pitcher_run_value",
        "rows": int(len(y)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)),
        "target_mean": float(y.mean()),
    }


def train_and_compare(features_2025: pd.DataFrame, features_2026: pd.DataFrame) -> dict[str, Any]:
    train_2025 = model_frame(features_2025)
    keep_pitch_types = pd.Index(sorted(train_2025["pitch_type"].unique()))
    eval_2026 = model_frame(features_2026, keep_pitch_types=keep_pitch_types) if not features_2026.empty else pd.DataFrame()
    cols = feature_sets(train_2025)

    X = train_2025[cols["next_features"]]
    y = train_2025["pitch_type"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.value_counts().min() >= 2 else None,
    )

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    next_model = make_hgb_classifier(cols["next_cat"], cols["next_num"])
    log("Training 2025 boosted next-pitch model for temporal validation...")
    next_model.fit(X_train, y_train)

    Xo = train_2025[cols["outcome_features"]]
    yo = train_2025["whiff_or_weak"].astype(int)
    Xo_train, Xo_test, yo_train, yo_test = train_test_split(
        Xo,
        yo,
        test_size=0.2,
        random_state=42,
        stratify=yo if yo.value_counts().min() >= 2 else None,
    )

    outcome_model = make_hgb_classifier(cols["outcome_cat"], cols["outcome_num"])
    log("Training 2025 whiff/weak-contact model for temporal validation...")
    outcome_model.fit(Xo_train, yo_train)

    run_model = make_hgb_regressor(cols["outcome_cat"], cols["outcome_num"])
    yr = train_2025["pitcher_run_value"].astype(float)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xo,
        yr,
        test_size=0.2,
        random_state=42,
    )
    log("Training 2025 run-value model for temporal validation...")
    run_model.fit(Xr_train, yr_train)

    metrics = [
        {
            "dataset": "2025_holdout",
            "task": "next_pitch_dummy",
            "rows": int(len(y_test)),
            "accuracy": float(accuracy_score(y_test, dummy.predict(X_test))),
        },
        evaluate_next_pitch(next_model, X_test, y_test, "2025_holdout"),
        evaluate_binary_outcome(outcome_model, Xo_test, yo_test, "2025_holdout"),
        evaluate_run_value(run_model, Xr_test, yr_test, "2025_holdout"),
    ]

    audit_2025 = prediction_audit(
        next_model,
        X_test,
        y_test,
        train_2025,
        "2025_holdout",
    )
    audit_2025.to_csv(OUTPUT_DIR / "next_pitch_prediction_audit_2025_holdout.csv", index=False)
    audits = [audit_2025]

    if not eval_2026.empty:
        metrics.extend(
            [
                evaluate_next_pitch(
                    next_model,
                    eval_2026[cols["next_features"]],
                    eval_2026["pitch_type"],
                    "2026_to_date",
                ),
                evaluate_binary_outcome(
                    outcome_model,
                    eval_2026[cols["outcome_features"]],
                    eval_2026["whiff_or_weak"],
                    "2026_to_date",
                ),
                evaluate_run_value(
                    run_model,
                    eval_2026[cols["outcome_features"]],
                    eval_2026["pitcher_run_value"].astype(float),
                    "2026_to_date",
                ),
            ]
        )
        audit_2026 = prediction_audit(
            next_model,
            eval_2026[cols["next_features"]],
            eval_2026["pitch_type"],
            eval_2026,
            "2026_to_date",
        )
        audit_2026.to_csv(OUTPUT_DIR / "next_pitch_prediction_audit_2026_to_date.csv", index=False)
        audits.append(audit_2026)

    audit_all = pd.concat(audits, ignore_index=True)
    audit_all.to_csv(OUTPUT_DIR / "next_pitch_prediction_audit_combined.csv", index=False)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTPUT_DIR / "temporal_validation_2025_to_2026.csv", index=False)
    with open(OUTPUT_DIR / "temporal_validation_2025_to_2026.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "train_2025": train_2025,
        "eval_2026": eval_2026,
        "metrics": metrics_df,
        "next_model": next_model,
        "outcome_model": outcome_model,
        "run_model": run_model,
        "feature_sets": cols,
        "Xo_train": Xo_train,
        "yo_train": yo_train,
        "prediction_audit": audit_all,
    }


def plot_pitch_usage_by_pitcher(df: pd.DataFrame) -> None:
    usage = df.groupby(["pitcher_name", "pitch_type"], as_index=False).size().rename(columns={"size": "pitches"})
    usage["share"] = usage["pitches"] / usage.groupby("pitcher_name")["pitches"].transform("sum")
    top_pitch_types = df["pitch_type"].value_counts().head(8).index.tolist()
    usage = usage.loc[usage["pitch_type"].isin(top_pitch_types)].copy()
    pivot = usage.pivot_table(index="pitcher_name", columns="pitch_type", values="share", fill_value=0)
    order = df.groupby("pitcher_name").size().sort_values(ascending=True).index
    pivot = pivot.reindex(order)

    ax = pivot.plot(kind="barh", stacked=True, figsize=(10, 8), width=0.82)
    ax.set_title("2025 Pitch Mix By Pitcher")
    ax.set_xlabel("Share of pitches")
    ax.set_ylabel("")
    ax.legend(title="Pitch", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(FIGURE_DIR / "pitch_usage_by_pitcher_2025.png")


def plot_transition_heatmap(df: pd.DataFrame) -> None:
    trans = df.loc[df["prev_pitch_type"].ne("START")].copy()
    top = trans["pitch_type"].value_counts().head(9).index.tolist()
    trans = trans.loc[trans["pitch_type"].isin(top) & trans["prev_pitch_type"].isin(top)]
    counts = pd.crosstab(trans["prev_pitch_type"], trans["pitch_type"])
    probs = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    probs.to_csv(OUTPUT_DIR / "pitch_transition_probabilities_2025.csv")

    plt.figure(figsize=(9, 7))
    sns.heatmap(probs, cmap="viridis", annot=True, fmt=".0%", cbar_kws={"label": "Next pitch probability"})
    plt.title("2025 Pitch Transition Probabilities")
    plt.xlabel("Current pitch")
    plt.ylabel("Previous pitch")
    save_fig(FIGURE_DIR / "pitch_transition_heatmap_2025.png")


def plot_count_pitch_mix(df: pd.DataFrame) -> None:
    top = df["pitch_type"].value_counts().head(8).index.tolist()
    sub = df.loc[df["pitch_type"].isin(top)].copy()
    counts = pd.crosstab(sub["count"], sub["pitch_type"])
    probs = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    count_order = [f"{b}-{s}" for b in range(4) for s in range(3) if f"{b}-{s}" in probs.index]
    probs = probs.reindex(count_order)
    probs.to_csv(OUTPUT_DIR / "count_pitch_mix_2025.csv")

    plt.figure(figsize=(10, 6))
    sns.heatmap(probs, cmap="mako", annot=True, fmt=".0%", cbar_kws={"label": "Pitch share"})
    plt.title("2025 Pitch Mix By Count")
    plt.xlabel("Pitch type")
    plt.ylabel("Count")
    save_fig(FIGURE_DIR / "count_pitch_mix_2025.png")


def plot_run_value_by_pitch_type(df: pd.DataFrame) -> None:
    summary = (
        df.groupby("pitch_type", as_index=False)
        .agg(
            pitches=("pitch_type", "size"),
            mean_pitcher_run_value=("pitcher_run_value", "mean"),
            whiff_or_weak_rate=("whiff_or_weak", "mean"),
        )
        .query("pitches >= 100")
        .sort_values("mean_pitcher_run_value", ascending=False)
    )
    summary["pitch_label"] = summary["pitch_type"].map(pitch_label)
    summary.to_csv(OUTPUT_DIR / "pitch_type_outcome_summary_2025.csv", index=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=summary, x="mean_pitcher_run_value", y="pitch_label", color="#3568a8")
    plt.axvline(0, color="#333333", linewidth=1)
    plt.title("2025 Average Pitcher Run Value By Pitch Type")
    plt.xlabel("Mean pitcher run value per pitch")
    plt.ylabel("")
    save_fig(FIGURE_DIR / "run_value_by_pitch_type_2025.png")


def plot_pitch_mix_2025_vs_2026(df_2025: pd.DataFrame, df_2026: pd.DataFrame) -> None:
    if df_2026.empty:
        return
    rows = []
    for label, data in [("2025", df_2025), ("2026 to date", df_2026)]:
        share = data["pitch_type"].value_counts(normalize=True)
        rows.extend({"season": label, "pitch_type": k, "share": v} for k, v in share.items())
    mix = pd.DataFrame(rows)
    top = df_2025["pitch_type"].value_counts().head(9).index.tolist()
    mix = mix.loc[mix["pitch_type"].isin(top)].copy()
    mix["pitch_label"] = mix["pitch_type"].map(pitch_label)
    mix.to_csv(OUTPUT_DIR / "pitch_mix_2025_vs_2026.csv", index=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=mix, x="pitch_label", y="share", hue="season")
    plt.title("Pitch Mix Drift: 2025 vs 2026 To Date")
    plt.xlabel("")
    plt.ylabel("Pitch share")
    plt.xticks(rotation=35, ha="right")
    save_fig(FIGURE_DIR / "pitch_mix_2025_vs_2026.png")


def plot_temporal_validation(metrics: pd.DataFrame) -> None:
    next_metrics = metrics.loc[metrics["task"].eq("next_pitch"), ["dataset", "accuracy", "top3_accuracy"]]
    outcome_metrics = metrics.loc[metrics["task"].eq("whiff_or_weak"), ["dataset", "auc", "brier"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if not next_metrics.empty:
        next_long = next_metrics.melt("dataset", var_name="metric", value_name="value")
        sns.barplot(data=next_long, x="metric", y="value", hue="dataset", ax=axes[0])
        axes[0].set_title("Next-Pitch Model")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Score")
        axes[0].set_ylim(0, 1)
    if not outcome_metrics.empty:
        outcome_long = outcome_metrics.melt("dataset", var_name="metric", value_name="value")
        sns.barplot(data=outcome_long, x="metric", y="value", hue="dataset", ax=axes[1])
        axes[1].set_title("Outcome Model")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Score")
        axes[1].set_ylim(0, 1)
    save_fig(FIGURE_DIR / "temporal_validation_2025_to_2026.png")


def plot_prediction_topk(metrics: pd.DataFrame) -> None:
    next_metrics = metrics.loc[
        metrics["task"].eq("next_pitch"),
        ["dataset", "accuracy", "top2_accuracy", "top3_accuracy", "mean_probability_assigned_to_actual_pitch"],
    ].copy()
    if next_metrics.empty:
        return

    rename = {
        "accuracy": "Exact match",
        "top2_accuracy": "Actual in top 2",
        "top3_accuracy": "Actual in top 3",
        "mean_probability_assigned_to_actual_pitch": "Avg P(actual)",
    }
    long = next_metrics.rename(columns=rename).melt("dataset", var_name="metric", value_name="value")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=long, x="metric", y="value", hue="dataset")
    plt.title("Prediction Accuracy Against Actual Pitches")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    save_fig(FIGURE_DIR / "prediction_accuracy_topk.png")


def plot_prediction_confusion(audit: pd.DataFrame, dataset: str) -> None:
    sub = audit.loc[audit["dataset"].eq(dataset)].copy()
    if sub.empty:
        return

    top = sub["actual_pitch"].value_counts().head(9).index.tolist()
    sub = sub.loc[sub["actual_pitch"].isin(top) & sub["predicted_pitch"].isin(top)].copy()
    matrix = pd.crosstab(sub["actual_pitch"], sub["predicted_pitch"])
    matrix = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
    matrix = matrix.reindex(index=top, columns=top, fill_value=0)
    matrix.index = [pitch_label(p) for p in matrix.index]
    matrix.columns = [pitch_label(p) for p in matrix.columns]
    matrix.to_csv(OUTPUT_DIR / f"prediction_confusion_{dataset}.csv")

    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix, cmap="Blues", annot=True, fmt=".0%", cbar_kws={"label": "Predicted share"})
    plt.title(f"Actual vs Predicted Pitch Type: {dataset}")
    plt.xlabel("Predicted pitch")
    plt.ylabel("Actual pitch")
    save_fig(FIGURE_DIR / f"prediction_confusion_{dataset}.png")


def plot_accuracy_by_count(audit: pd.DataFrame) -> None:
    if "count" not in audit.columns:
        return
    summary = (
        audit.groupby(["dataset", "count"], as_index=False)
        .agg(
            pitches=("correct_top1", "size"),
            exact_match=("correct_top1", "mean"),
            top3=("correct_top3", "mean"),
            avg_actual_probability=("actual_pitch_probability", "mean"),
        )
        .query("pitches >= 40")
    )
    count_order = [f"{b}-{s}" for b in range(4) for s in range(3)]
    summary["count"] = pd.Categorical(summary["count"], categories=count_order, ordered=True)
    summary = summary.sort_values(["dataset", "count"])
    summary.to_csv(OUTPUT_DIR / "prediction_accuracy_by_count.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    sns.lineplot(data=summary, x="count", y="exact_match", hue="dataset", marker="o", ax=axes[0])
    axes[0].set_title("Exact-Match Accuracy By Count")
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    sns.lineplot(data=summary, x="count", y="top3", hue="dataset", marker="o", ax=axes[1])
    axes[1].set_title("Top-3 Accuracy By Count")
    axes[1].set_xlabel("Count")
    axes[1].set_ylabel("")
    axes[1].set_ylim(0, 1)
    save_fig(FIGURE_DIR / "prediction_accuracy_by_count.png")


def plot_accuracy_by_pitcher(audit: pd.DataFrame) -> None:
    if "pitcher_name" not in audit.columns:
        return
    summary = (
        audit.groupby(["dataset", "pitcher_name"], as_index=False)
        .agg(
            pitches=("correct_top1", "size"),
            exact_match=("correct_top1", "mean"),
            top3=("correct_top3", "mean"),
            avg_actual_probability=("actual_pitch_probability", "mean"),
        )
        .query("pitches >= 60")
        .sort_values(["dataset", "exact_match"], ascending=[True, False])
    )
    summary.to_csv(OUTPUT_DIR / "prediction_accuracy_by_pitcher.csv", index=False)

    datasets = summary["dataset"].unique().tolist()
    fig, axes = plt.subplots(
        nrows=len(datasets),
        ncols=1,
        figsize=(10, max(4.5, 0.34 * len(summary))),
        sharex=True,
    )
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        sub = summary.loc[summary["dataset"].eq(dataset)].sort_values("exact_match", ascending=True)
        sns.barplot(data=sub, x="exact_match", y="pitcher_name", color="#3b6ea8", ax=ax)
        ax.set_title(f"Exact-Match Accuracy By Pitcher: {dataset}")
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("")
        ax.set_xlim(0, max(0.55, float(summary["exact_match"].max()) + 0.05))
    save_fig(FIGURE_DIR / "prediction_accuracy_by_pitcher.png")


def plot_actual_pitch_probability(audit: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=audit,
        x="actual_pitch_probability",
        hue="dataset",
        bins=30,
        stat="density",
        common_norm=False,
        element="step",
    )
    plt.title("Probability Assigned To The Pitch Actually Thrown")
    plt.xlabel("Model probability of actual pitch")
    plt.ylabel("Density")
    save_fig(FIGURE_DIR / "probability_assigned_to_actual_pitch.png")


def plot_actual_vs_predicted_pitch_mix(audit: pd.DataFrame) -> None:
    rows = []
    for dataset, sub in audit.groupby("dataset"):
        for column, label in [("actual_pitch", "Actual"), ("predicted_pitch", "Predicted")]:
            share = sub[column].value_counts(normalize=True)
            rows.extend(
                {
                    "dataset": dataset,
                    "mix_type": label,
                    "pitch_type": pitch,
                    "share": value,
                    "pitch_label": pitch_label(pitch),
                }
                for pitch, value in share.items()
            )
    mix = pd.DataFrame(rows)
    top = audit["actual_pitch"].value_counts().head(9).index.tolist()
    mix = mix.loc[mix["pitch_type"].isin(top)].copy()
    mix.to_csv(OUTPUT_DIR / "actual_vs_predicted_pitch_mix.csv", index=False)

    datasets = mix["dataset"].unique().tolist()
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 4.5 * len(datasets)), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        sub = mix.loc[mix["dataset"].eq(dataset)]
        sns.barplot(data=sub, x="pitch_label", y="share", hue="mix_type", ax=ax)
        ax.set_title(f"Actual vs Predicted Pitch Mix: {dataset}")
        ax.set_xlabel("")
        ax.set_ylabel("Share")
        ax.tick_params(axis="x", rotation=35)
    save_fig(FIGURE_DIR / "actual_vs_predicted_pitch_mix.png")


def plot_confidence_calibration(audit: pd.DataFrame) -> None:
    bins = np.linspace(0, 1, 11)
    binned = audit.copy()
    binned["confidence_bin"] = pd.cut(binned["model_confidence"], bins=bins, include_lowest=True)
    summary = (
        binned.groupby(["dataset", "confidence_bin"], observed=True, as_index=False)
        .agg(
            pitches=("correct_top1", "size"),
            mean_confidence=("model_confidence", "mean"),
            exact_match=("correct_top1", "mean"),
        )
        .query("pitches >= 20")
    )
    summary.to_csv(OUTPUT_DIR / "prediction_confidence_calibration.csv", index=False)

    plt.figure(figsize=(7, 6))
    sns.lineplot(data=summary, x="mean_confidence", y="exact_match", hue="dataset", marker="o")
    plt.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1)
    plt.title("Prediction Confidence vs Actual Accuracy")
    plt.xlabel("Mean model confidence")
    plt.ylabel("Exact-match accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    save_fig(FIGURE_DIR / "prediction_confidence_calibration.png")


def shap_feature_importance(models: dict[str, Any], max_rows: int) -> pd.DataFrame:
    import shap
    from sklearn.ensemble import RandomForestClassifier

    cols = models["feature_sets"]
    outcome_model = models["outcome_model"]
    preprocessor = outcome_model.named_steps["preprocess"]
    X_train = models["Xo_train"]
    y_train = models["yo_train"]

    X_train_trans = preprocessor.transform(X_train)
    feature_names = cols["outcome_cat"] + cols["outcome_num"]

    shap_model = RandomForestClassifier(
        n_estimators=160,
        min_samples_leaf=25,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    log("Training random forest outcome model for SHAP interpretation...")
    shap_model.fit(X_train_trans, y_train)

    sample_size = min(max_rows, X_train_trans.shape[0])
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(X_train_trans.shape[0], size=sample_size, replace=False)
    X_sample = X_train_trans[sample_idx]

    log(f"Computing SHAP values on {sample_size:,} sampled training rows...")
    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        values = shap_values[1]
    elif getattr(shap_values, "ndim", 0) == 3:
        values = shap_values[:, :, 1]
    else:
        values = shap_values

    importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": np.abs(values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(OUTPUT_DIR / "shap_outcome_feature_importance.csv", index=False)

    top = importance.head(18).sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#2f7f6f")
    plt.title("SHAP Feature Importance For Whiff/Weak-Contact Model")
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("")
    save_fig(FIGURE_DIR / "shap_outcome_feature_importance.png")
    return importance


def make_visuals(
    features_2025: pd.DataFrame,
    features_2026: pd.DataFrame,
    metrics: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plot_pitch_usage_by_pitcher(features_2025)
    plot_transition_heatmap(features_2025)
    plot_count_pitch_mix(features_2025)
    plot_run_value_by_pitch_type(features_2025)
    plot_pitch_mix_2025_vs_2026(features_2025, features_2026)
    plot_temporal_validation(metrics)
    plot_prediction_topk(metrics)
    for dataset in audit["dataset"].unique():
        plot_prediction_confusion(audit, dataset)
    plot_accuracy_by_count(audit)
    plot_accuracy_by_pitcher(audit)
    plot_actual_pitch_probability(audit)
    plot_actual_vs_predicted_pitch_mix(audit)
    plot_confidence_calibration(audit)


def print_summary(
    features_2025: pd.DataFrame,
    features_2026: pd.DataFrame,
    models: dict[str, Any],
    shap_df: pd.DataFrame | None = None,
) -> None:
    log("\n=== Data coverage ===")
    log(f"2025 rows: {len(features_2025):,} ({features_2025['game_date'].min()} through {features_2025['game_date'].max()})")
    if features_2026.empty:
        log("2026 rows: 0")
    else:
        log(f"2026 rows: {len(features_2026):,} ({features_2026['game_date'].min()} through {features_2026['game_date'].max()})")

    log("\n=== 2025-trained model comparison ===")
    print(models["metrics"].to_string(index=False), flush=True)

    audit = models["prediction_audit"]
    log("\n=== Prediction audit samples ===")
    sample_cols = [
        "dataset",
        "pitcher_name",
        "batter_name",
        "count",
        "prev_pitch_type",
        "actual_pitch_label",
        "predicted_pitch_label",
        "correct_top1",
        "correct_top3",
        "actual_pitch_probability",
        "top_3_predictions",
    ]
    print(audit[sample_cols].head(12).to_string(index=False), flush=True)

    if shap_df is not None:
        log("\n=== Top SHAP features for whiff/weak contact ===")
        print(shap_df.head(12).to_string(index=False), flush=True)

    log("\n=== Figure files ===")
    for path in sorted(FIGURE_DIR.glob("*.png")):
        log(str(path.relative_to(ROOT)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize and validate the pitch sequencing project.")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--start-2026", default="2026-03-01")
    parser.add_argument("--end-2026", default=date.today().isoformat())
    parser.add_argument("--with-shap", action="store_true", help="Also compute optional SHAP feature importance.")
    parser.add_argument("--max-shap-rows", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    cache.enable()

    features_2025 = load_2025_features(args.top_n)
    pitchers = load_2025_pitchers(args.top_n)
    features_2026 = pull_or_load_2026_features(pitchers, args.top_n, args.start_2026, args.end_2026)
    models = train_and_compare(features_2025, features_2026)
    make_visuals(features_2025, features_2026, models["metrics"], models["prediction_audit"])
    shap_df = shap_feature_importance(models, max_rows=args.max_shap_rows) if args.with_shap else None
    print_summary(features_2025, features_2026, models, shap_df)


if __name__ == "__main__":
    main()
