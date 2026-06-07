#!/usr/bin/env python3
"""State-machine, OLS beta analysis, and model comparison for pitch sequencing."""

from __future__ import annotations

import json
import os
import argparse
from dataclasses import dataclass
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

try:
    from xgboost import XGBClassifier
    XGB_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - lets Colab/local fallback cleanly if xgboost is absent.
    XGBClassifier = None
    XGB_IMPORT_ERROR = exc

from pitch_sequence_pipeline import PROCESSED_DIR, ROOT, log, pitch_label, sequence_features_path, temporal_sequence_features_path


OUTPUT_DIR = ROOT / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"

STATE_COLS = ["pitcher_name", "batter_name", "count", "prev_pitch_type", "stand"]
STATE_FALLBACKS = [
    ["pitcher_name", "count", "prev_pitch_type", "stand"],
    ["pitcher_name", "count", "prev_pitch_type"],
    ["pitcher_name", "batter_name", "count"],
    ["pitcher_name", "count"],
    ["pitcher_name", "prev_pitch_type"],
    ["pitcher_name"],
]

MODEL_CAT = [
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

MODEL_NUM = [
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

OLS_FEATURES = ["count", "prev_pitch_type", "stand", "prev_description"]


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
    out = df.loc[df["pitch_type"].notna()].copy()
    if out.empty:
        return out
    if keep_pitch_types is None:
        counts = out["pitch_type"].value_counts()
        keep_pitch_types = counts.loc[counts >= min_pitch_type_count].index
    return out.loc[out["pitch_type"].isin(keep_pitch_types)].copy()


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


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


def build_pitcher_arsenals(
    rows: pd.DataFrame,
    min_usage: float = 0.03,
    min_pitches: int = 30,
) -> pd.DataFrame:
    agg_spec: dict[str, tuple[str, str]] = {
        "pitches": ("pitch_type", "size"),
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
        if source_col in rows.columns:
            agg_spec[output_col] = (source_col, "mean")

    arsenal = (
        rows.groupby(["pitcher_name", "pitch_type"], as_index=False)
        .agg(**agg_spec)
        .sort_values(["pitcher_name", "pitches"], ascending=[True, False])
    )
    arsenal["pitch_label"] = arsenal["pitch_type"].map(pitch_label)
    arsenal["pitcher_total_pitches"] = arsenal.groupby("pitcher_name")["pitches"].transform("sum")
    arsenal["usage_rate"] = arsenal["pitches"] / arsenal["pitcher_total_pitches"]
    arsenal["in_arsenal"] = (arsenal["usage_rate"] >= min_usage) | (arsenal["pitches"] >= min_pitches)

    top_idx = arsenal.groupby("pitcher_name")["pitches"].idxmax()
    arsenal.loc[top_idx, "in_arsenal"] = True
    arsenal["arsenal_size"] = arsenal.groupby("pitcher_name")["in_arsenal"].transform("sum").astype(int)
    arsenal.to_csv(OUTPUT_DIR / "state_ols_pitcher_arsenals.csv", index=False)
    arsenal.loc[arsenal["in_arsenal"]].to_csv(OUTPUT_DIR / "state_ols_pitcher_arsenals_allowed_only.csv", index=False)
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


def plot_pitcher_arsenals(arsenal: pd.DataFrame) -> None:
    allowed = arsenal.loc[arsenal["in_arsenal"]].copy()
    pivot = allowed.pivot_table(index="pitcher_name", columns="pitch_type", values="usage_rate", fill_value=0)
    order = allowed.groupby("pitcher_name")["arsenal_size"].max().sort_values(ascending=True).index
    pivot = pivot.reindex(order)
    ax = pivot.plot(kind="barh", stacked=True, figsize=(10, 8), width=0.82)
    ax.set_title("Pitcher-Specific Arsenals")
    ax.set_xlabel("Pitch usage share")
    ax.set_ylabel("")
    ax.legend(title="Pitch", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(FIGURE_DIR / "state_ols_pitcher_arsenals.png")


def build_state_machine(rows: pd.DataFrame) -> pd.DataFrame:
    state_cols = available_columns(rows, STATE_COLS)
    transitions = (
        rows.groupby(state_cols + ["pitch_type"], as_index=False)
        .size()
        .rename(columns={"size": "transition_count", "pitch_type": "next_pitch_type"})
    )
    transitions["state_total"] = transitions.groupby(state_cols)["transition_count"].transform("sum")
    transitions["transition_probability"] = transitions["transition_count"] / transitions["state_total"]
    transitions["state_rank"] = transitions.groupby(state_cols)["transition_probability"].rank(method="first", ascending=False)
    transitions["next_pitch_label"] = transitions["next_pitch_type"].map(pitch_label)
    transitions = transitions.sort_values(state_cols + ["state_rank"])
    transitions.to_csv(OUTPUT_DIR / "state_machine_transitions.csv", index=False)
    transitions.loc[transitions["state_rank"] <= 3].to_csv(OUTPUT_DIR / "state_machine_top3_transitions.csv", index=False)
    return transitions


def state_machine_probs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    classes: np.ndarray,
    keys: list[str] = STATE_COLS,
    fallback_keys: list[list[str]] = STATE_FALLBACKS,
    alpha: float = 1.0,
) -> np.ndarray:
    global_counts = train["pitch_type"].value_counts().reindex(classes, fill_value=0).astype(float)
    global_probs = ((global_counts + alpha) / (global_counts.sum() + alpha * len(classes))).to_numpy()

    tables = []
    for key_set in [keys] + fallback_keys:
        key_set = [col for col in key_set if col in train.columns and col in test.columns]
        if not key_set:
            continue
        counts = train.groupby(key_set + ["pitch_type"]).size().unstack(fill_value=0)
        counts = counts.reindex(columns=classes, fill_value=0).astype(float)
        probs = (counts + alpha).div(counts.sum(axis=1) + alpha * len(classes), axis=0)
        lookup = {
            key: row.to_numpy(dtype=float)
            for key, row in probs.iterrows()
        }
        tables.append((key_set, lookup))

    out = np.zeros((len(test), len(classes)), dtype=float)
    for i, row in enumerate(test.itertuples(index=False)):
        row_map = row._asdict()
        chosen = None
        for key_set, table in tables:
            key = tuple(row_map[col] for col in key_set)
            if len(key) == 1:
                key = key[0]
            chosen = table.get(key)
            if chosen is not None:
                break
        out[i] = chosen if chosen is not None else global_probs
    return out


def plot_state_machine_heatmaps(rows: pd.DataFrame, max_pitchers: int = 4) -> None:
    top_pitchers = rows["pitcher_name"].value_counts().head(max_pitchers).index.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    for ax, pitcher in zip(axes, top_pitchers):
        sub = rows.loc[(rows["pitcher_name"] == pitcher) & (rows["prev_pitch_type"] != "START")].copy()
        top_pitches = sub["pitch_type"].value_counts().head(7).index.tolist()
        sub = sub.loc[sub["pitch_type"].isin(top_pitches) & sub["prev_pitch_type"].isin(top_pitches)]
        matrix = pd.crosstab(sub["prev_pitch_type"], sub["pitch_type"])
        matrix = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
        sns.heatmap(matrix, cmap="mako", annot=True, fmt=".0%", cbar=False, ax=ax)
        ax.set_title(f"{pitcher}: P(next pitch | previous pitch)")
        ax.set_xlabel("Next pitch")
        ax.set_ylabel("Previous pitch")
    for ax in axes[len(top_pitchers) :]:
        ax.axis("off")
    save_fig(FIGURE_DIR / "state_machine_transition_heatmap_top_pitchers.png")


def fit_pitcher_ols(rows: pd.DataFrame, arsenal: pd.DataFrame) -> pd.DataFrame:
    coef_rows: list[dict[str, Any]] = []
    allowed = arsenal.loc[arsenal["in_arsenal"], ["pitcher_name", "pitch_type"]]
    allowed_map = allowed.groupby("pitcher_name")["pitch_type"].apply(list).to_dict()
    feature_cols = available_columns(rows, OLS_FEATURES)

    for pitcher_name, pitcher_df in rows.groupby("pitcher_name"):
        pitcher_df = pitcher_df.copy()
        targets = allowed_map.get(pitcher_name, [])
        if len(pitcher_df) < 100 or len(targets) < 2:
            continue

        encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        X = encoder.fit_transform(pitcher_df[feature_cols].fillna("missing"))
        feature_names = encoder.get_feature_names_out(feature_cols)

        for target_pitch in targets:
            y = pitcher_df["pitch_type"].eq(target_pitch).astype(float).to_numpy()
            if y.sum() < 20:
                continue
            model = LinearRegression()
            model.fit(X, y)
            coef_rows.append(
                {
                    "pitcher_name": pitcher_name,
                    "target_pitch_type": target_pitch,
                    "target_pitch_label": pitch_label(target_pitch),
                    "feature": "intercept",
                    "beta": float(model.intercept_),
                    "n_rows": int(len(pitcher_df)),
                    "target_rate": float(y.mean()),
                }
            )
            for feature, beta in zip(feature_names, model.coef_):
                coef_rows.append(
                    {
                        "pitcher_name": pitcher_name,
                        "target_pitch_type": target_pitch,
                        "target_pitch_label": pitch_label(target_pitch),
                        "feature": feature,
                        "beta": float(beta),
                        "n_rows": int(len(pitcher_df)),
                        "target_rate": float(y.mean()),
                    }
                )

    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(OUTPUT_DIR / "ols_pitch_type_coefficients_long.csv", index=False)
    if not coef_df.empty:
        pivot = coef_df.pivot_table(
            index=["pitcher_name", "feature"],
            columns="target_pitch_type",
            values="beta",
            aggfunc="mean",
        ).reset_index()
        pivot.to_csv(OUTPUT_DIR / "ols_pitch_type_coefficients_pivot.csv", index=False)
        write_beta_matrices(coef_df)
    return coef_df


def safe_feature_name(feature: str) -> str:
    return (
        feature.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
        .replace("/", "_")
        .replace("=", "_")
    )


def write_beta_matrices(coef_df: pd.DataFrame) -> None:
    intercepts = coef_df.loc[coef_df["feature"].eq("intercept")]
    if not intercepts.empty:
        intercept_matrix = intercepts.pivot_table(
            index="pitcher_name",
            columns="target_pitch_type",
            values="beta",
            aggfunc="mean",
            fill_value=0,
        ).reset_index()
        intercept_matrix.to_csv(OUTPUT_DIR / "ols_beta_pitcher_pitch_intercepts.csv", index=False)

    context = coef_df.loc[coef_df["feature"].ne("intercept")].copy()
    if not context.empty:
        context["abs_beta"] = context["beta"].abs()
        mean_abs_matrix = context.pivot_table(
            index="pitcher_name",
            columns="target_pitch_type",
            values="abs_beta",
            aggfunc="mean",
            fill_value=0,
        ).reset_index()
        mean_abs_matrix.to_csv(OUTPUT_DIR / "ols_beta_pitcher_pitch_mean_abs_context.csv", index=False)

        feature_summary = (
            context.groupby(["target_pitch_type", "target_pitch_label", "feature"], as_index=False)
            .agg(
                mean_beta=("beta", "mean"),
                mean_abs_beta=("abs_beta", "mean"),
                pitchers=("pitcher_name", "nunique"),
                mean_target_rate=("target_rate", "mean"),
            )
            .sort_values(["target_pitch_type", "mean_abs_beta"], ascending=[True, False])
        )
        feature_summary.to_csv(OUTPUT_DIR / "ols_beta_feature_summary.csv", index=False)

        features_to_matrix = ["stand_R", "count_3-2", "count_0-2", "prev_pitch_type_FF", "prev_pitch_type_SL"]
        for feature in features_to_matrix:
            feature_rows = context.loc[context["feature"].eq(feature)]
            if feature_rows.empty:
                continue
            matrix = feature_rows.pivot_table(
                index="pitcher_name",
                columns="target_pitch_type",
                values="beta",
                aggfunc="mean",
                fill_value=0,
            ).reset_index()
            matrix.to_csv(OUTPUT_DIR / f"ols_beta_pitcher_pitch_{safe_feature_name(feature)}.csv", index=False)


def plot_beta_heatmap(coef_df: pd.DataFrame) -> None:
    if coef_df.empty:
        return
    features_of_interest = [
        "count_0-2",
        "count_1-2",
        "count_2-0",
        "count_3-0",
        "count_3-2",
        "prev_pitch_type_FF",
        "prev_pitch_type_SI",
        "prev_pitch_type_SL",
        "prev_pitch_type_ST",
        "prev_pitch_type_CH",
        "stand_R",
    ]
    sub = coef_df.loc[coef_df["feature"].isin(features_of_interest)].copy()
    if sub.empty:
        sub = coef_df.loc[coef_df["feature"].ne("intercept")].copy()
    top = (
        sub.assign(abs_beta=sub["beta"].abs())
        .groupby("feature")["abs_beta"]
        .mean()
        .sort_values(ascending=False)
        .head(12)
        .index
    )
    plot_df = (
        sub.loc[sub["feature"].isin(top)]
        .groupby(["feature", "target_pitch_type"], as_index=False)["beta"]
        .mean()
        .pivot(index="feature", columns="target_pitch_type", values="beta")
        .fillna(0)
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(plot_df, cmap="vlag", center=0, annot=True, fmt="+.2f", cbar_kws={"label": "Mean OLS beta"})
    plt.title("Average Per-Pitcher OLS Coefficients")
    plt.xlabel("Target pitch type")
    plt.ylabel("Feature")
    save_fig(FIGURE_DIR / "ols_beta_coefficient_heatmap.png")


@dataclass
class XGBPitchModel:
    preprocessor: ColumnTransformer
    label_encoder: LabelEncoder
    model: Any
    classes_: np.ndarray

    def predict_proba(self, X: pd.DataFrame, global_classes: np.ndarray) -> np.ndarray:
        Xt = self.preprocessor.transform(X)
        raw = self.model.predict_proba(Xt)
        out = np.zeros((len(X), len(global_classes)), dtype=float)
        model_classes = self.label_encoder.inverse_transform(np.arange(raw.shape[1]))
        class_to_idx = {pitch: i for i, pitch in enumerate(model_classes)}
        for j, pitch in enumerate(global_classes):
            if pitch in class_to_idx:
                out[:, j] = raw[:, class_to_idx[pitch]]
        row_sum = out.sum(axis=1)
        missing = row_sum == 0
        out[missing, :] = 1 / len(global_classes)
        out[~missing, :] = out[~missing, :] / row_sum[~missing, None]
        return out


def make_xgb_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    return ColumnTransformer([("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)])


def make_hgb_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    return ColumnTransformer([("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)])


def fit_xgb_model(X: pd.DataFrame, y: pd.Series, cat_cols: list[str], num_cols: list[str]) -> XGBPitchModel:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    if XGBClassifier is None:
        preprocessor = make_hgb_preprocessor(cat_cols, num_cols)
        Xt = preprocessor.fit_transform(X)
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=45,
            max_leaf_nodes=31,
            l2_regularization=0.02,
            random_state=42,
        )
    else:
        preprocessor = make_xgb_preprocessor(cat_cols, num_cols)
        Xt = preprocessor.fit_transform(X)
        model = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=160,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=1.5,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
    model.fit(Xt, y_encoded)
    return XGBPitchModel(preprocessor, label_encoder, model, label_encoder.classes_)


def fit_per_pitcher_xgb(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    classes: np.ndarray,
    cat_cols: list[str],
    num_cols: list[str],
) -> np.ndarray:
    probs = np.zeros((len(test_rows), len(classes)), dtype=float)
    probs[:] = np.nan
    feature_cols = cat_cols + num_cols
    per_pitcher_cat = [col for col in cat_cols if col != "pitcher_name"]
    per_pitcher_features = per_pitcher_cat + num_cols

    global_state_probs = state_machine_probs(train_rows, test_rows, classes)
    probs[:] = global_state_probs

    pitcher_groups = list(train_rows.groupby("pitcher_name"))
    for idx, (pitcher_name, pitcher_train) in enumerate(pitcher_groups, start=1):
        if idx == 1 or idx % 10 == 0 or idx == len(pitcher_groups):
            log(f"  fitting per-pitcher model {idx}/{len(pitcher_groups)}: {pitcher_name}")
        pitcher_test_idx = test_rows.index[test_rows["pitcher_name"] == pitcher_name]
        if len(pitcher_test_idx) == 0:
            continue
        if len(pitcher_train) < 300 or pitcher_train["pitch_type"].nunique() < 2:
            continue
        try:
            model = fit_xgb_model(
                pitcher_train[per_pitcher_features],
                pitcher_train["pitch_type"],
                per_pitcher_cat,
                num_cols,
            )
            pitcher_test = test_rows.loc[pitcher_test_idx, per_pitcher_features]
            local_probs = model.predict_proba(pitcher_test, classes)
            positions = test_rows.index.get_indexer(pitcher_test_idx)
            probs[positions] = local_probs
        except Exception as exc:
            log(f"  per-pitcher model skipped for {pitcher_name}: {exc}")

    return probs


def plot_model_comparison(metrics: pd.DataFrame) -> None:
    subset = metrics.copy()
    order = (
        subset.loc[subset["dataset"].eq("2025_holdout")]
        .sort_values("accuracy", ascending=False)["model"]
        .tolist()
    )
    subset["model"] = pd.Categorical(subset["model"], categories=order, ordered=True)

    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=subset, x="accuracy", y="model", hue="dataset")
    plt.title("Model Comparison: Exact Next-Pitch Accuracy")
    plt.xlabel("Exact-match accuracy")
    plt.ylabel("")
    save_fig(FIGURE_DIR / "state_ols_model_comparison_accuracy.png")

    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=subset, x="top3_accuracy", y="model", hue="dataset")
    plt.title("Model Comparison: Top-3 Next-Pitch Accuracy")
    plt.xlabel("Top-3 accuracy")
    plt.ylabel("")
    plt.xlim(0, 1)
    save_fig(FIGURE_DIR / "state_ols_model_comparison_top3.png")


def run_model_comparison(train_2025: pd.DataFrame, eval_2026: pd.DataFrame, arsenal_map: dict[str, set[str]]) -> pd.DataFrame:
    cat_cols = available_columns(train_2025, MODEL_CAT)
    num_cols = available_columns(train_2025, MODEL_NUM)
    feature_cols = cat_cols + num_cols

    stratify = train_2025["pitch_type"] if train_2025["pitch_type"].value_counts().min() >= 2 else None
    train_rows, test_rows = train_test_split(
        train_2025,
        test_size=0.20,
        random_state=42,
        stratify=stratify,
    )
    classes = np.asarray(sorted(train_rows["pitch_type"].unique()))
    metrics: list[dict[str, Any]] = []
    booster_label = "xgboost" if XGBClassifier is not None else "hist_gradient_boosting_fallback"

    log("Computing state-machine probabilities for 2025 holdout...")
    state_probs = state_machine_probs(train_rows, test_rows, classes)

    log(f"Training pooled {booster_label} super model...")
    pooled_model = fit_xgb_model(train_rows[feature_cols], train_rows["pitch_type"], cat_cols, num_cols)
    pooled_probs = pooled_model.predict_proba(test_rows[feature_cols], classes)
    log(f"Training per-pitcher {booster_label} loop...")
    per_pitcher_probs = fit_per_pitcher_xgb(train_rows, test_rows, classes, cat_cols, num_cols)
    metrics.append(evaluate_probs(f"per_pitcher_{booster_label}", "2025_holdout", classes, per_pitcher_probs, test_rows["pitch_type"]))
    masked_per_pitcher_probs = apply_arsenal_mask(per_pitcher_probs, test_rows["pitcher_name"], classes, arsenal_map)
    metrics.append(
        evaluate_probs(
            f"per_pitcher_{booster_label}_arsenal_masked",
            "2025_holdout",
            classes,
            masked_per_pitcher_probs,
            test_rows["pitch_type"],
        )
    )
    metrics.append(evaluate_probs("state_machine_lookup", "2025_holdout", classes, state_probs, test_rows["pitch_type"]))
    masked_state_probs = apply_arsenal_mask(state_probs, test_rows["pitcher_name"], classes, arsenal_map)
    metrics.append(
        evaluate_probs("state_machine_lookup_arsenal_masked", "2025_holdout", classes, masked_state_probs, test_rows["pitch_type"])
    )
    metrics.append(evaluate_probs(f"pooled_{booster_label}", "2025_holdout", classes, pooled_probs, test_rows["pitch_type"]))
    masked_pooled_probs = apply_arsenal_mask(pooled_probs, test_rows["pitcher_name"], classes, arsenal_map)
    metrics.append(
        evaluate_probs(f"pooled_{booster_label}_arsenal_masked", "2025_holdout", classes, masked_pooled_probs, test_rows["pitch_type"])
    )

    if not eval_2026.empty:
        eval_2026 = eval_2026.loc[eval_2026["pitch_type"].isin(classes)].copy()
        log("Scoring 2026 rows with per-pitcher loop...")
        per_pitcher_probs_2026 = fit_per_pitcher_xgb(train_rows, eval_2026, classes, cat_cols, num_cols)
        metrics.append(
            evaluate_probs(f"per_pitcher_{booster_label}", "2026_to_date", classes, per_pitcher_probs_2026, eval_2026["pitch_type"])
        )
        metrics.append(
            evaluate_probs(
                f"per_pitcher_{booster_label}_arsenal_masked",
                "2026_to_date",
                classes,
                apply_arsenal_mask(per_pitcher_probs_2026, eval_2026["pitcher_name"], classes, arsenal_map),
                eval_2026["pitch_type"],
            )
        )
        log("Computing state-machine probabilities for 2026 rows...")
        state_probs_2026 = state_machine_probs(train_rows, eval_2026, classes)
        metrics.append(evaluate_probs("state_machine_lookup", "2026_to_date", classes, state_probs_2026, eval_2026["pitch_type"]))
        metrics.append(
            evaluate_probs(
                "state_machine_lookup_arsenal_masked",
                "2026_to_date",
                classes,
                apply_arsenal_mask(state_probs_2026, eval_2026["pitcher_name"], classes, arsenal_map),
                eval_2026["pitch_type"],
            )
        )
        log("Scoring 2026 rows with pooled super model...")
        pooled_probs_2026 = pooled_model.predict_proba(eval_2026[feature_cols], classes)
        metrics.append(evaluate_probs(f"pooled_{booster_label}", "2026_to_date", classes, pooled_probs_2026, eval_2026["pitch_type"]))
        metrics.append(
            evaluate_probs(
                f"pooled_{booster_label}_arsenal_masked",
                "2026_to_date",
                classes,
                apply_arsenal_mask(pooled_probs_2026, eval_2026["pitcher_name"], classes, arsenal_map),
                eval_2026["pitch_type"],
            )
        )

    metrics_df = pd.DataFrame(metrics).sort_values(["dataset", "accuracy"], ascending=[True, False])
    metrics_df.to_csv(OUTPUT_DIR / "state_ols_model_comparison_metrics.csv", index=False)
    with open(OUTPUT_DIR / "state_ols_model_comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    plot_model_comparison(metrics_df)
    return metrics_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run state-machine, OLS beta, pooled, and per-pitcher models.")
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="notebook")

    features_2025, features_2026 = load_features(args.top_n)
    train_2025 = model_frame(features_2025)
    keep_pitch_types = pd.Index(sorted(train_2025["pitch_type"].unique()))
    eval_2026 = model_frame(features_2026, keep_pitch_types=keep_pitch_types) if not features_2026.empty else pd.DataFrame()

    log("Building pitcher-specific arsenals...")
    arsenal = build_pitcher_arsenals(train_2025)
    arsenal_map = arsenal_map_from_table(arsenal)
    plot_pitcher_arsenals(arsenal)

    log("Building state-machine transition tables...")
    transitions = build_state_machine(train_2025)
    plot_state_machine_heatmaps(train_2025)

    log("Fitting per-pitcher OLS linear probability models...")
    coef_df = fit_pitcher_ols(train_2025, arsenal)
    plot_beta_heatmap(coef_df)

    if XGBClassifier is None:
        log(f"XGBoost unavailable locally; using sklearn HistGradientBoosting fallback. Import error: {XGB_IMPORT_ERROR}")
    log("Comparing per-pitcher XGBoost/HGB loops, state-machine lookup, and pooled super model...")
    metrics_df = run_model_comparison(train_2025, eval_2026, arsenal_map)

    log("\n=== State-machine sample ===")
    print(transitions.head(15).to_string(index=False), flush=True)

    log("\n=== OLS beta sample ===")
    if coef_df.empty:
        log("No OLS coefficients were generated.")
    else:
        beta_sample = (
            coef_df.loc[coef_df["feature"].ne("intercept")]
            .assign(abs_beta=lambda x: x["beta"].abs())
            .sort_values("abs_beta", ascending=False)
            .head(15)
            .drop(columns="abs_beta")
        )
        print(beta_sample.to_string(index=False), flush=True)

    log("\n=== Model comparison ===")
    print(metrics_df.to_string(index=False), flush=True)

    log("\n=== Files written ===")
    for path in [
        OUTPUT_DIR / "state_machine_transitions.csv",
        OUTPUT_DIR / "state_machine_top3_transitions.csv",
        OUTPUT_DIR / "ols_pitch_type_coefficients_long.csv",
        OUTPUT_DIR / "ols_pitch_type_coefficients_pivot.csv",
        OUTPUT_DIR / "ols_beta_pitcher_pitch_intercepts.csv",
        OUTPUT_DIR / "ols_beta_pitcher_pitch_mean_abs_context.csv",
        OUTPUT_DIR / "ols_beta_feature_summary.csv",
        OUTPUT_DIR / "state_ols_model_comparison_metrics.csv",
        FIGURE_DIR / "state_machine_transition_heatmap_top_pitchers.png",
        FIGURE_DIR / "ols_beta_coefficient_heatmap.png",
        FIGURE_DIR / "state_ols_model_comparison_accuracy.png",
    ]:
        log(str(path.relative_to(ROOT)))


if __name__ == "__main__":
    main()
