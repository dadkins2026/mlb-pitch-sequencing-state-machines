#!/usr/bin/env python3
"""Diagnose and tune next-pitch prediction models."""

from __future__ import annotations

import json
import os
import argparse
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from pitch_sequence_pipeline import PROCESSED_DIR, ROOT, log, pitch_label, sequence_features_path, temporal_sequence_features_path


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


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    log(f"wrote {path.relative_to(ROOT)}")


def load_features(top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_2025 = pd.read_parquet(sequence_features_path(2025, top_n))
    path_2026 = temporal_sequence_features_path(2025, 2026, top_n)
    df_2026 = pd.read_parquet(path_2026) if path_2026.exists() else pd.DataFrame()
    return df_2025, df_2026


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


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    cat = [col for col in NEXT_CAT if col in df.columns]
    num = [col for col in NEXT_NUM if col in df.columns]
    return cat, num, cat + num


def top_k_accuracy_from_probs(classes: np.ndarray, probs: np.ndarray, y: pd.Series, k: int) -> float:
    order = np.argsort(probs, axis=1)[:, -min(k, len(classes)) :]
    top = classes[order]
    return float(np.mean([actual in choices for actual, choices in zip(y.to_numpy(), top)]))


def evaluate_probs(name: str, dataset: str, classes: np.ndarray, probs: np.ndarray, y: pd.Series) -> dict[str, Any]:
    pred = classes[np.argmax(probs, axis=1)]
    class_to_idx = {pitch: idx for idx, pitch in enumerate(classes)}
    actual_prob = np.array([probs[i, class_to_idx[pitch]] if pitch in class_to_idx else 0.0 for i, pitch in enumerate(y)])
    return {
        "model": name,
        "dataset": dataset,
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "top2_accuracy": top_k_accuracy_from_probs(classes, probs, y, 2),
        "top3_accuracy": top_k_accuracy_from_probs(classes, probs, y, 3),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "avg_actual_probability": float(actual_prob.mean()),
        "avg_confidence": float(probs.max(axis=1).mean()),
    }


def build_pitcher_arsenal_table(
    rows: pd.DataFrame,
    y: pd.Series,
    min_usage: float = 0.03,
    min_pitches: int = 30,
) -> pd.DataFrame:
    data = rows.loc[y.index].copy()
    data["_target_pitch"] = y.to_numpy()

    agg_spec: dict[str, tuple[str, str]] = {
        "pitches": ("_target_pitch", "size"),
    }
    optional_means = {
        "avg_release_speed": "release_speed",
        "avg_release_spin_rate": "release_spin_rate",
        "whiff_rate": "whiff",
        "weak_contact_rate": "weak_contact",
        "whiff_or_weak_rate": "whiff_or_weak",
        "mean_pitcher_run_value": "pitcher_run_value",
    }
    for output_col, source_col in optional_means.items():
        if source_col in data.columns:
            agg_spec[output_col] = (source_col, "mean")

    arsenal = (
        data.groupby(["pitcher_name", "_target_pitch"], as_index=False)
        .agg(**agg_spec)
        .rename(columns={"_target_pitch": "pitch_type"})
    )
    arsenal["pitch_label"] = arsenal["pitch_type"].map(pitch_label)
    arsenal["pitcher_total_pitches"] = arsenal.groupby("pitcher_name")["pitches"].transform("sum")
    arsenal["usage_rate"] = arsenal["pitches"] / arsenal["pitcher_total_pitches"]
    arsenal["in_arsenal"] = (arsenal["usage_rate"] >= min_usage) | (arsenal["pitches"] >= min_pitches)

    # Guarantee every pitcher has at least one allowed pitch if thresholds are too strict.
    top_idx = arsenal.sort_values(["pitcher_name", "pitches"], ascending=[True, False]).groupby("pitcher_name").head(1).index
    arsenal.loc[top_idx, "in_arsenal"] = True

    arsenal["arsenal_size"] = arsenal.groupby("pitcher_name")["in_arsenal"].transform("sum").astype(int)
    arsenal = arsenal.sort_values(["pitcher_name", "in_arsenal", "usage_rate"], ascending=[True, False, False])
    arsenal.to_csv(OUTPUT_DIR / "pitcher_arsenals_2025_training.csv", index=False)
    return arsenal


def arsenal_map_from_table(arsenal: pd.DataFrame) -> dict[str, set[str]]:
    allowed = arsenal.loc[arsenal["in_arsenal"]].copy()
    return allowed.groupby("pitcher_name")["pitch_type"].apply(lambda s: set(s.astype(str))).to_dict()


def apply_arsenal_mask(
    probs: np.ndarray,
    pitcher_names: pd.Series,
    classes: np.ndarray,
    arsenal_map: dict[str, set[str]],
) -> np.ndarray:
    class_index = {pitch: idx for idx, pitch in enumerate(classes)}
    masked = np.zeros_like(probs)

    for i, pitcher_name in enumerate(pitcher_names.astype(str).to_numpy()):
        allowed = arsenal_map.get(pitcher_name)
        if not allowed:
            masked[i] = probs[i]
            continue

        allowed_idx = [class_index[pitch] for pitch in allowed if pitch in class_index]
        if not allowed_idx:
            masked[i] = probs[i]
            continue

        row = np.zeros(probs.shape[1], dtype=float)
        row[allowed_idx] = probs[i, allowed_idx]
        total = row.sum()
        if total > 0:
            row = row / total
        else:
            row[allowed_idx] = 1 / len(allowed_idx)
        masked[i] = row

    return masked


def distribution_lookup_predict(
    train: pd.DataFrame,
    target: pd.Series,
    test: pd.DataFrame,
    classes: np.ndarray,
    keys: list[str],
    fallback_keys: list[list[str]],
    alpha: float = 0.0,
) -> np.ndarray:
    class_index = {pitch: i for i, pitch in enumerate(classes)}
    global_counts = target.value_counts().reindex(classes, fill_value=0).astype(float)
    global_probs = (global_counts + alpha) / (global_counts.sum() + alpha * len(classes))

    tables = []
    train_with_y = train.copy()
    train_with_y["_target_pitch"] = target.to_numpy()
    for key_set in [keys] + fallback_keys:
        if not key_set:
            continue
        counts = train_with_y.groupby(key_set + ["_target_pitch"]).size().unstack(fill_value=0)
        counts = counts.reindex(columns=classes, fill_value=0).astype(float)
        probs = (counts + alpha).div(counts.sum(axis=1) + alpha * len(classes), axis=0)
        tables.append((key_set, probs))

    out = np.zeros((len(test), len(classes)), dtype=float)
    for i, row in enumerate(test.itertuples(index=False)):
        row_map = row._asdict()
        chosen = None
        for key_set, table in tables:
            key = tuple(row_map[col] for col in key_set)
            if len(key) == 1:
                key = key[0]
            if key in table.index:
                chosen = table.loc[key].to_numpy(dtype=float)
                break
        if chosen is None:
            chosen = global_probs.to_numpy(dtype=float)
        out[i] = chosen
    return out


def make_logistic(cat: list[str], num: list[str], class_weight: str | None, c_value: float = 1.0) -> Pipeline:
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )
    num_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    pre = ColumnTransformer([("cat", cat_pipe, cat), ("num", num_pipe, num)])
    return Pipeline(
        [
            ("preprocess", pre),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    max_iter=1800,
                    solver="lbfgs",
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def make_hgb(cat: list[str], num: list[str]) -> Pipeline:
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    pre = ColumnTransformer([("cat", cat_pipe, cat), ("num", num_pipe, num)])
    return Pipeline(
        [
            ("preprocess", pre),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=240,
                    max_leaf_nodes=31,
                    l2_regularization=0.02,
                    random_state=42,
                ),
            ),
        ]
    )


def make_random_forest(cat: list[str], num: list[str]) -> Pipeline:
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    pre = ColumnTransformer([("cat", cat_pipe, cat), ("num", num_pipe, num)])
    return Pipeline(
        [
            ("preprocess", pre),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=160,
                    min_samples_leaf=12,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def model_probs(model: Pipeline, X: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    out = np.zeros((len(X), len(classes)), dtype=float)
    class_to_idx = {pitch: i for i, pitch in enumerate(model.classes_)}
    for j, pitch in enumerate(classes):
        if pitch in class_to_idx:
            out[:, j] = raw[:, class_to_idx[pitch]]
    row_sum = out.sum(axis=1)
    missing = row_sum == 0
    out[missing, :] = 1 / len(classes)
    out[~missing, :] = out[~missing, :] / row_sum[~missing, None]
    return out


def audit_best_model(
    model_name: str,
    model: Pipeline,
    classes: np.ndarray,
    X: pd.DataFrame,
    y: pd.Series,
    rows: pd.DataFrame,
    dataset: str,
    arsenal_map: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    probs = model_probs(model, X, classes)
    if arsenal_map is not None:
        probs = apply_arsenal_mask(probs, X["pitcher_name"], classes, arsenal_map)
    order = np.argsort(probs, axis=1)[:, ::-1]
    pred = classes[order[:, 0]]
    class_to_idx = {pitch: idx for idx, pitch in enumerate(classes)}
    actual_idx = np.array([class_to_idx[pitch] for pitch in y])

    out = rows.loc[X.index, ["game_date", "pitcher_name", "batter_name", "count", "prev_pitch_type", "prev_description"]].copy()
    out.insert(0, "dataset", dataset)
    out.insert(0, "model", model_name)
    out["actual_pitch"] = y.to_numpy()
    out["predicted_pitch"] = pred
    out["actual_pitch_label"] = out["actual_pitch"].map(pitch_label)
    out["predicted_pitch_label"] = out["predicted_pitch"].map(pitch_label)
    out["correct_top1"] = out["actual_pitch"].eq(out["predicted_pitch"])
    out["correct_top2"] = [actual_idx[i] in order[i, :2] for i in range(len(y))]
    out["correct_top3"] = [actual_idx[i] in order[i, :3] for i in range(len(y))]
    out["actual_pitch_probability"] = probs[np.arange(len(y)), actual_idx]
    out["model_confidence"] = probs.max(axis=1)
    out["top_3_predictions"] = [
        ", ".join(f"{pitch_label(classes[j])} {probs[i, j]:.1%}" for j in order[i, :3])
        for i in range(len(y))
    ]
    return out


def plot_pitcher_arsenals(arsenal: pd.DataFrame) -> None:
    allowed = arsenal.loc[arsenal["in_arsenal"]].copy()
    allowed.to_csv(OUTPUT_DIR / "pitcher_arsenals_2025_training_allowed_only.csv", index=False)
    pivot = allowed.pivot_table(
        index="pitcher_name",
        columns="pitch_type",
        values="usage_rate",
        fill_value=0,
    )
    order = allowed.groupby("pitcher_name")["arsenal_size"].max().sort_values(ascending=True).index
    pivot = pivot.reindex(order)

    ax = pivot.plot(kind="barh", stacked=True, figsize=(10, 8), width=0.82)
    ax.set_title("Pitcher-Specific Arsenals From 2025 Training Data")
    ax.set_xlabel("Pitch usage share")
    ax.set_ylabel("")
    ax.legend(title="Pitch", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(FIGURE_DIR / "pitcher_specific_arsenals_2025_training.png")


def add_unmasked_and_arsenal_metrics(
    metrics: list[dict[str, Any]],
    name: str,
    dataset: str,
    classes: np.ndarray,
    probs: np.ndarray,
    y: pd.Series,
    pitcher_names: pd.Series,
    arsenal_map: dict[str, set[str]],
) -> None:
    metrics.append(evaluate_probs(name, dataset, classes, probs, y))
    masked = apply_arsenal_mask(probs, pitcher_names, classes, arsenal_map)
    metrics.append(evaluate_probs(f"{name}_arsenal_masked", dataset, classes, masked, y))


def plot_sweep(metrics: pd.DataFrame) -> None:
    subset = metrics.loc[metrics["dataset"].isin(["2025_holdout", "2026_to_date"])].copy()
    subset = subset.loc[~subset["model"].str.contains("dummy")]
    order = (
        subset.loc[subset["dataset"].eq("2025_holdout")]
        .sort_values("accuracy", ascending=False)["model"]
        .tolist()
    )
    subset["model"] = pd.Categorical(subset["model"], categories=order, ordered=True)

    plt.figure(figsize=(11, 6))
    sns.barplot(data=subset, x="accuracy", y="model", hue="dataset")
    plt.title("Next-Pitch Exact-Match Accuracy: Model Sweep")
    plt.xlabel("Accuracy")
    plt.ylabel("")
    save_fig(FIGURE_DIR / "next_pitch_model_sweep_accuracy.png")

    plt.figure(figsize=(11, 6))
    sns.barplot(data=subset, x="top3_accuracy", y="model", hue="dataset")
    plt.title("Next-Pitch Top-3 Accuracy: Model Sweep")
    plt.xlabel("Top-3 accuracy")
    plt.ylabel("")
    plt.xlim(0, 1)
    save_fig(FIGURE_DIR / "next_pitch_model_sweep_top3.png")


def plot_mix_for_best(audit: pd.DataFrame, name: str) -> None:
    rows = []
    for dataset, sub in audit.groupby("dataset"):
        for col, label in [("actual_pitch", "Actual"), ("predicted_pitch", "Predicted")]:
            share = sub[col].value_counts(normalize=True)
            rows.extend(
                {
                    "dataset": dataset,
                    "mix_type": label,
                    "pitch_type": pitch,
                    "pitch_label": pitch_label(pitch),
                    "share": value,
                }
                for pitch, value in share.items()
            )
    mix = pd.DataFrame(rows)
    mix.to_csv(OUTPUT_DIR / f"{name}_actual_vs_predicted_pitch_mix.csv", index=False)

    datasets = mix["dataset"].unique().tolist()
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 4.5 * len(datasets)), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        sub = mix.loc[mix["dataset"].eq(dataset)]
        order = sub.loc[sub["mix_type"].eq("Actual")].sort_values("share", ascending=False)["pitch_label"]
        sns.barplot(data=sub, x="pitch_label", y="share", hue="mix_type", order=order, ax=ax)
        ax.set_title(f"Actual vs Predicted Pitch Mix: {dataset} ({name})")
        ax.set_xlabel("")
        ax.set_ylabel("Share")
        ax.tick_params(axis="x", rotation=35)
    save_fig(FIGURE_DIR / f"{name}_actual_vs_predicted_pitch_mix.png")


def context_entropy(train: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    df = train.copy()
    df["_target_pitch"] = y.to_numpy()
    rows = []
    for label, keys in {
        "pitcher": ["pitcher_name"],
        "pitcher_count": ["pitcher_name", "count"],
        "pitcher_prev": ["pitcher_name", "prev_pitch_type"],
        "pitcher_count_prev": ["pitcher_name", "count", "prev_pitch_type"],
        "pitcher_count_prev_stand": ["pitcher_name", "count", "prev_pitch_type", "stand"],
    }.items():
        counts = df.groupby(keys + ["_target_pitch"]).size().unstack(fill_value=0)
        probs = counts.div(counts.sum(axis=1), axis=0)
        max_share = probs.max(axis=1)
        entropy = -(probs.where(probs > 0, 1).map(np.log2) * probs).sum(axis=1)
        weights = counts.sum(axis=1) / counts.sum(axis=1).sum()
        rows.append(
            {
                "context": label,
                "groups": int(len(counts)),
                "weighted_mode_share": float((max_share * weights).sum()),
                "weighted_entropy_bits": float((entropy * weights).sum()),
                "median_group_pitches": float(counts.sum(axis=1).median()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "pitch_choice_context_entropy.csv", index=False)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run next-pitch model sweep and context diagnostics.")
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="notebook")

    features_2025, features_2026 = load_features(args.top_n)
    train_2025 = model_frame(features_2025)
    keep_pitch_types = pd.Index(sorted(train_2025["pitch_type"].unique()))
    eval_2026 = model_frame(features_2026, keep_pitch_types) if not features_2026.empty else pd.DataFrame()
    cat, num, features = feature_columns(train_2025)

    X = train_2025[features]
    y = train_2025["pitch_type"]
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    classes = np.asarray(sorted(y_train.unique()))
    metrics: list[dict[str, Any]] = []
    arsenal = build_pitcher_arsenal_table(train_2025.loc[X_train.index], y_train)
    arsenal_map = arsenal_map_from_table(arsenal)
    plot_pitcher_arsenals(arsenal)

    lookup_specs = {
        "lookup_global": ([], []),
        "lookup_pitcher": (["pitcher_name"], [[]]),
        "lookup_pitcher_count": (["pitcher_name", "count"], [["pitcher_name"], []]),
        "lookup_pitcher_prev": (["pitcher_name", "prev_pitch_type"], [["pitcher_name"], []]),
        "lookup_pitcher_count_prev": (
            ["pitcher_name", "count", "prev_pitch_type"],
            [["pitcher_name", "count"], ["pitcher_name", "prev_pitch_type"], ["pitcher_name"], []],
        ),
        "lookup_pitcher_count_prev_stand": (
            ["pitcher_name", "count", "prev_pitch_type", "stand"],
            [["pitcher_name", "count", "prev_pitch_type"], ["pitcher_name", "count"], ["pitcher_name"], []],
        ),
    }

    log("Evaluating empirical lookup baselines...")
    for name, (keys, fallbacks) in lookup_specs.items():
        if keys:
            probs = distribution_lookup_predict(X_train, y_train, X_test, classes, keys, fallbacks, alpha=1.0)
        else:
            counts = y_train.value_counts().reindex(classes, fill_value=0).astype(float)
            probs = np.tile((counts / counts.sum()).to_numpy(), (len(X_test), 1))
        add_unmasked_and_arsenal_metrics(
            metrics,
            name,
            "2025_holdout",
            classes,
            probs,
            y_test,
            X_test["pitcher_name"],
            arsenal_map,
        )
        if not eval_2026.empty:
            X_2026 = eval_2026[features]
            y_2026 = eval_2026["pitch_type"]
            if keys:
                probs_2026 = distribution_lookup_predict(X_train, y_train, X_2026, classes, keys, fallbacks, alpha=1.0)
            else:
                counts = y_train.value_counts().reindex(classes, fill_value=0).astype(float)
                probs_2026 = np.tile((counts / counts.sum()).to_numpy(), (len(X_2026), 1))
            add_unmasked_and_arsenal_metrics(
                metrics,
                name,
                "2026_to_date",
                classes,
                probs_2026,
                y_2026,
                X_2026["pitcher_name"],
                arsenal_map,
            )

    models = {
        "hist_gradient_boosting": make_hgb(cat, num),
        "random_forest": make_random_forest(cat, num),
    }

    fitted_models: dict[str, Pipeline] = {}
    log("Training model sweep...")
    for name, model in models.items():
        log(f"  training {name}")
        model.fit(X_train, y_train)
        fitted_models[name] = model
        probs = model_probs(model, X_test, classes)
        add_unmasked_and_arsenal_metrics(
            metrics,
            name,
            "2025_holdout",
            classes,
            probs,
            y_test,
            X_test["pitcher_name"],
            arsenal_map,
        )
        if not eval_2026.empty:
            X_2026 = eval_2026[features]
            y_2026 = eval_2026["pitch_type"]
            probs_2026 = model_probs(model, X_2026, classes)
            add_unmasked_and_arsenal_metrics(
                metrics,
                name,
                "2026_to_date",
                classes,
                probs_2026,
                y_2026,
                X_2026["pitcher_name"],
                arsenal_map,
            )

    metrics_df = pd.DataFrame(metrics).sort_values(["dataset", "accuracy"], ascending=[True, False])
    metrics_df.to_csv(OUTPUT_DIR / "next_pitch_model_sweep_metrics.csv", index=False)
    with open(OUTPUT_DIR / "next_pitch_model_sweep_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    entropy = context_entropy(X_train, y_train)
    plot_sweep(metrics_df)

    best_name = (
        metrics_df.loc[metrics_df["dataset"].eq("2025_holdout")]
        .sort_values(["accuracy", "top3_accuracy"], ascending=False)
        .iloc[0]["model"]
    )
    audit_base_name = "hist_gradient_boosting"
    if audit_base_name in fitted_models:
        best_model = fitted_models[audit_base_name]
        audit_name = f"{audit_base_name}_arsenal_masked"
        audits = [
            audit_best_model(
                audit_name,
                best_model,
                classes,
                X_test,
                y_test,
                train_2025,
                "2025_holdout",
                arsenal_map=arsenal_map,
            )
        ]
        if not eval_2026.empty:
            audits.append(
                audit_best_model(
                    audit_name,
                    best_model,
                    classes,
                    eval_2026[features],
                    eval_2026["pitch_type"],
                    eval_2026,
                    "2026_to_date",
                    arsenal_map=arsenal_map,
                )
            )
        audit = pd.concat(audits, ignore_index=True)
        audit.to_csv(OUTPUT_DIR / f"{audit_name}_prediction_audit.csv", index=False)
        plot_mix_for_best(audit, audit_name)

    log("\n=== Model sweep metrics ===")
    print(metrics_df.to_string(index=False), flush=True)

    log("\n=== Context uncertainty diagnostics ===")
    print(entropy.to_string(index=False), flush=True)

    log("\n=== Pitcher-specific arsenals ===")
    display_cols = ["pitcher_name", "pitch_type", "pitch_label", "pitches", "usage_rate", "arsenal_size"]
    print(
        arsenal.loc[arsenal["in_arsenal"], [c for c in display_cols if c in arsenal.columns]]
        .sort_values(["pitcher_name", "usage_rate"], ascending=[True, False])
        .to_string(index=False),
        flush=True,
    )

    log(f"\nBest 2025 holdout model by exact accuracy: {best_name}")


if __name__ == "__main__":
    main()
