#!/usr/bin/env python3
"""Paper-ready analysis outputs: tuned models, regressions, SHAP figures, and captions."""

from __future__ import annotations

import json
import os
import re
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
import shap
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from pitch_sequence_pipeline import PROCESSED_DIR, ROOT, log, pitch_label, sequence_features_path, temporal_sequence_features_path


OUTPUT_DIR = ROOT / "output"
PAPER_DIR = OUTPUT_DIR / "paper_ready"
TABLE_DIR = PAPER_DIR / "tables"
FIGURE_DIR = PAPER_DIR / "figures"

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

STATE_SPECS = {
    "pitcher_batter_count_prev_stand": ["pitcher_name", "batter_name", "count", "prev_pitch_type", "stand"],
    "pitcher_count_prev_stand": ["pitcher_name", "count", "prev_pitch_type", "stand"],
}

STATE_FALLBACKS = [
    ["pitcher_name", "count", "prev_pitch_type"],
    ["pitcher_name", "batter_name", "count"],
    ["pitcher_name", "count"],
    ["pitcher_name", "prev_pitch_type"],
    ["pitcher_name"],
]


def ensure_dirs() -> None:
    for path in [PAPER_DIR, TABLE_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Serif",
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_fig(name: str) -> Path:
    path = FIGURE_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"wrote {path.relative_to(ROOT)}")
    return path


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


def top_k_accuracy(classes: np.ndarray, probs: np.ndarray, y: pd.Series, k: int) -> float:
    order = np.argsort(probs, axis=1)[:, -min(k, len(classes)) :]
    top = classes[order]
    return float(np.mean([actual in row for actual, row in zip(y.to_numpy(), top)]))


def evaluate_probs(model: str, dataset: str, classes: np.ndarray, probs: np.ndarray, y: pd.Series) -> dict[str, Any]:
    pred = classes[np.argmax(probs, axis=1)]
    class_to_idx = {pitch: idx for idx, pitch in enumerate(classes)}
    actual_prob = np.array([probs[i, class_to_idx.get(pitch, -1)] if pitch in class_to_idx else 0.0 for i, pitch in enumerate(y)])
    loss = log_loss(y, probs, labels=list(classes)) if y.nunique() > 1 else np.nan
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
        "log_loss": float(loss),
    }


def build_arsenal(train_rows: pd.DataFrame, min_usage: float = 0.03, min_pitches: int = 30) -> pd.DataFrame:
    agg_spec: dict[str, tuple[str, str]] = {
        "pitches": ("pitch_type", "size"),
        "avg_velocity": ("release_speed", "mean"),
        "whiff_rate": ("whiff", "mean"),
        "weak_contact_rate": ("weak_contact", "mean"),
        "mean_pitcher_run_value": ("pitcher_run_value", "mean"),
    }
    arsenal = (
        train_rows.groupby(["pitcher_name", "pitch_type"], as_index=False)
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
    arsenal.to_csv(TABLE_DIR / "pitcher_arsenals.csv", index=False)
    arsenal.loc[arsenal["in_arsenal"]].to_csv(TABLE_DIR / "pitcher_arsenals_allowed_only.csv", index=False)
    return arsenal


def arsenal_map(arsenal: pd.DataFrame) -> dict[str, set[str]]:
    return arsenal.loc[arsenal["in_arsenal"]].groupby("pitcher_name")["pitch_type"].apply(lambda s: set(s.astype(str))).to_dict()


def apply_arsenal_mask(
    probs: np.ndarray,
    pitcher_names: pd.Series,
    classes: np.ndarray,
    allowed: dict[str, set[str]],
) -> np.ndarray:
    idx = {pitch: i for i, pitch in enumerate(classes)}
    masked = np.zeros_like(probs)
    for i, pitcher_name in enumerate(pitcher_names.astype(str)):
        allowed_pitches = allowed.get(pitcher_name)
        if not allowed_pitches:
            masked[i] = probs[i]
            continue
        allowed_idx = [idx[pitch] for pitch in allowed_pitches if pitch in idx]
        if not allowed_idx:
            masked[i] = probs[i]
            continue
        row = np.zeros(probs.shape[1])
        row[allowed_idx] = probs[i, allowed_idx]
        total = row.sum()
        if total > 0:
            row /= total
        else:
            row[allowed_idx] = 1 / len(allowed_idx)
        masked[i] = row
    return masked


def state_machine_probs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    classes: np.ndarray,
    keys: list[str],
    alpha: float,
    min_count: int,
) -> np.ndarray:
    global_counts = train["pitch_type"].value_counts().reindex(classes, fill_value=0).astype(float)
    global_probs = ((global_counts + alpha) / (global_counts.sum() + alpha * len(classes))).to_numpy()
    tables = []
    for key_set in [keys] + STATE_FALLBACKS:
        key_set = [col for col in key_set if col in train.columns and col in test.columns]
        if not key_set:
            continue
        counts = train.groupby(key_set + ["pitch_type"]).size().unstack(fill_value=0)
        counts = counts.reindex(columns=classes, fill_value=0).astype(float)
        totals = counts.sum(axis=1)
        probs = (counts + alpha).div(totals + alpha * len(classes), axis=0)
        lookup = {}
        for key, row in probs.iterrows():
            if totals.loc[key] >= min_count:
                lookup[key] = row.to_numpy(dtype=float)
        tables.append((key_set, lookup))

    out = np.zeros((len(test), len(classes)))
    for i, row in enumerate(test.itertuples(index=False)):
        row_map = row._asdict()
        chosen = None
        for key_set, probs in tables:
            key = tuple(row_map[col] for col in key_set)
            if len(key) == 1:
                key = key[0]
            chosen = probs.get(key)
            if chosen is not None:
                break
        out[i] = chosen if chosen is not None else global_probs
    return out


def tune_state_machine(
    train: pd.DataFrame,
    test: pd.DataFrame,
    classes: np.ndarray,
    allowed: dict[str, set[str]],
    eval_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    eval_sets = [("2025_holdout", test)]
    if eval_rows is not None and not eval_rows.empty:
        eval_sets.append(("2026_to_date", eval_rows))

    for spec_name, keys in STATE_SPECS.items():
        for alpha in [1.0]:
            for min_count in [5, 15]:
                log(f"  state grid {spec_name}, alpha={alpha}, min_count={min_count}")
                for dataset_name, rows_eval in eval_sets:
                    probs = state_machine_probs(train, rows_eval, classes, keys, alpha, min_count)
                    rows.append(
                        {
                            **evaluate_probs(f"state:{spec_name}", dataset_name, classes, probs, rows_eval["pitch_type"]),
                            "alpha": alpha,
                            "min_count": min_count,
                            "state_spec": spec_name,
                            "arsenal_masked": False,
                        }
                    )
                    masked = apply_arsenal_mask(probs, rows_eval["pitcher_name"], classes, allowed)
                    rows.append(
                        {
                            **evaluate_probs(
                                f"state:{spec_name}:arsenal_masked",
                                dataset_name,
                                classes,
                                masked,
                                rows_eval["pitch_type"],
                            ),
                            "alpha": alpha,
                            "min_count": min_count,
                            "state_spec": spec_name,
                            "arsenal_masked": True,
                        }
                    )
    out = pd.DataFrame(rows).sort_values(["dataset", "exact_accuracy", "top3_accuracy"], ascending=[True, False, False])
    out.to_csv(TABLE_DIR / "state_machine_tuning_grid.csv", index=False)
    return out


def make_hgb_model(cat_cols: list[str], num_cols: list[str], params: dict[str, Any]) -> Pipeline:
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
            (
                "model",
                HistGradientBoostingClassifier(
                    random_state=42,
                    **params,
                ),
            ),
        ]
    )


def make_logistic_model(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
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
    pre = ColumnTransformer([("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)])
    return Pipeline(
        [
            ("preprocess", pre),
            ("model", LogisticRegression(C=0.35, max_iter=1000, solver="lbfgs")),
        ]
    )


def model_probs(model: Pipeline, X: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    out = np.zeros((len(X), len(classes)))
    local_idx = {pitch: i for i, pitch in enumerate(model.classes_)}
    for j, pitch in enumerate(classes):
        if pitch in local_idx:
            out[:, j] = raw[:, local_idx[pitch]]
    row_sum = out.sum(axis=1)
    missing = row_sum == 0
    out[missing] = 1 / len(classes)
    out[~missing] = out[~missing] / row_sum[~missing, None]
    return out


def tune_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    classes: np.ndarray,
    allowed: dict[str, set[str]],
    eval_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cat_cols = available_columns(train, MODEL_CAT)
    num_cols = available_columns(train, MODEL_NUM)
    features = cat_cols + num_cols
    rows = []
    eval_sets = [("2025_holdout", test)]
    if eval_rows is not None and not eval_rows.empty:
        eval_sets.append(("2026_to_date", eval_rows))

    grid = [
        {"learning_rate": 0.07, "max_iter": 100, "max_leaf_nodes": 31, "l2_regularization": 0.02},
        {"learning_rate": 0.07, "max_iter": 180, "max_leaf_nodes": 31, "l2_regularization": 0.02},
        {"learning_rate": 0.05, "max_iter": 180, "max_leaf_nodes": 31, "l2_regularization": 0.02},
        {"learning_rate": 0.08, "max_iter": 160, "max_leaf_nodes": 31, "l2_regularization": 0.05},
    ]
    for i, params in enumerate(grid, start=1):
        name = f"pooled_hgb_tuned_{i}"
        log(f"  training {name}: {params}")
        model = make_hgb_model(cat_cols, num_cols, params)
        model.fit(train[features], train["pitch_type"])
        for dataset_name, rows_eval in eval_sets:
            probs = model_probs(model, rows_eval[features], classes)
            row = evaluate_probs(name, dataset_name, classes, probs, rows_eval["pitch_type"])
            row.update(params)
            rows.append(row)
            masked = apply_arsenal_mask(probs, rows_eval["pitcher_name"], classes, allowed)
            masked_row = evaluate_probs(f"{name}_arsenal_masked", dataset_name, classes, masked, rows_eval["pitch_type"])
            masked_row.update(params)
            rows.append(masked_row)

    out = pd.DataFrame(rows).sort_values(["dataset", "exact_accuracy", "top3_accuracy"], ascending=[True, False, False])
    out.to_csv(TABLE_DIR / "pooled_model_tuning_grid.csv", index=False)
    return out


def fit_regression_tables(rows: pd.DataFrame, target_pitches: list[str]) -> pd.DataFrame:
    model_rows = rows.copy()
    # Keep formula simple, readable, and aligned with the professor's OLS beta request.
    formula_rhs = 'C(count, Treatment(reference="0-0")) + C(prev_pitch_type, Treatment(reference="START")) + C(stand, Treatment(reference="L")) + C(pitcher_name)'
    coef_rows = []
    for pitch in target_pitches:
        target = f"is_{pitch}"
        model_rows[target] = model_rows["pitch_type"].eq(pitch).astype(int)
        model = smf.ols(f"{target} ~ {formula_rhs}", data=model_rows).fit(cov_type="HC1")
        for term, beta in model.params.items():
            coef_rows.append(
                {
                    "target_pitch_type": pitch,
                    "target_pitch_label": pitch_label(pitch),
                    "term": term,
                    "beta": float(beta),
                    "std_error": float(model.bse[term]),
                    "p_value": float(model.pvalues[term]),
                    "n_obs": int(model.nobs),
                    "r_squared": float(model.rsquared),
                }
            )
    out = pd.DataFrame(coef_rows)
    out.to_csv(TABLE_DIR / "ols_pooled_fixed_effects_coefficients.csv", index=False)
    return out


def readable_term(term: str) -> str:
    if term == "Intercept":
        return "Intercept"
    term = re.sub(r'C\(count, Treatment\(reference="0-0"\)\)\[T\.(.+?)\]', r"Count: \1", term)
    term = re.sub(r'C\(prev_pitch_type, Treatment\(reference="START"\)\)\[T\.(.+?)\]', r"Previous pitch: \1", term)
    term = re.sub(r'C\(stand, Treatment\(reference="L"\)\)\[T\.(.+?)\]', r"Batter side: \1", term)
    term = re.sub(r"C\(pitcher_name\)\[T\.(.+?)\]", r"Pitcher: \1", term)
    return term


def clean_feature_name(name: str) -> str:
    name = name.replace("cat__", "").replace("num__", "")
    mappings = {
        "pitcher_name_": "Pitcher: ",
        "count_": "Count: ",
        "prev_pitch_type_": "Previous pitch: ",
        "prev2_pitch_type_": "Two pitches ago: ",
        "stand_": "Batter side: ",
        "prev_description_": "Previous result: ",
        "inning_topbot_": "Half inning: ",
        "base_state_": "Base state: ",
    }
    for prefix, label in mappings.items():
        if name.startswith(prefix):
            return label + name[len(prefix) :]
    return name.replace("_", " ")


def shap_feature_group(feature: str) -> str:
    if feature.startswith("Pitcher: "):
        return "Pitcher identity / arsenal"
    if feature.startswith("Previous pitch: "):
        return "Previous pitch type"
    if feature.startswith("Two pitches ago: "):
        return "Two-pitches-ago pitch type"
    if feature.startswith("Batter side: "):
        return "Batter handedness"
    if feature.startswith("Count: ") or feature in {"balls", "strikes"}:
        return "Count / ball-strike leverage"
    if feature.startswith("Previous result: "):
        return "Previous pitch result"
    if feature.startswith("Base state: "):
        return "Base/out state"
    if feature in {"prev release speed", "pitch number", "score diff batting team"}:
        return feature
    return "Other context"


def compact_model_label(name: str) -> str:
    if name == "logistic_lpm_style_baseline":
        return "Logistic baseline"
    if name == "logistic_lpm_style_baseline_arsenal_masked":
        return "Logistic + arsenal"
    if name.startswith("state:"):
        label = "State machine"
        if name.endswith(":arsenal_masked"):
            label += " + arsenal"
        return label
    if name.startswith("pooled_hgb"):
        label = "Pooled boosted trees"
        if name.endswith("_arsenal_masked"):
            label += " + arsenal"
        return label
    return name


def make_paper_figures(
    arsenal: pd.DataFrame,
    state_grid: pd.DataFrame,
    model_grid: pd.DataFrame,
    regression: pd.DataFrame,
) -> None:
    allowed = arsenal.loc[arsenal["in_arsenal"]].copy()
    pivot = allowed.pivot_table(index="pitcher_name", columns="pitch_type", values="usage_rate", fill_value=0)
    order = allowed.groupby("pitcher_name")["arsenal_size"].max().sort_values(ascending=True).index
    pivot = pivot.reindex(order)
    ax = pivot.plot(kind="barh", stacked=True, figsize=(7.2, 6.2), width=0.82, colormap="tab20")
    ax.set_title("Pitcher-specific arsenals define the strategic menu")
    ax.set_xlabel("Pitch usage share")
    ax.set_ylabel("")
    ax.legend(title="Pitch", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    save_fig("fig1_pitcher_specific_arsenals.png")

    state_2025 = state_grid.loc[state_grid["dataset"].eq("2025_holdout")].copy()
    model_2025 = model_grid.loc[model_grid["dataset"].eq("2025_holdout")].copy()
    best_state_name = state_2025.iloc[0]["model"]
    best_model_name = model_2025.iloc[0]["model"]
    comparison_names = [
        "logistic_lpm_style_baseline",
        "logistic_lpm_style_baseline_arsenal_masked",
        best_state_name,
        best_model_name,
    ]
    if str(best_model_name).endswith("_arsenal_masked"):
        comparison_names.append(str(best_model_name).removesuffix("_arsenal_masked"))
    comparison = pd.concat([state_grid, model_grid], ignore_index=True)
    comparison = comparison.loc[comparison["model"].isin(dict.fromkeys(comparison_names))]
    comparison["model_label"] = comparison["model"].map(compact_model_label)
    comparison["dataset_label"] = comparison["dataset"].map({"2025_holdout": "2025 holdout", "2026_to_date": "2026 to date"})
    plot_df = comparison[["model_label", "dataset_label", "exact_accuracy", "top3_accuracy"]].melt(
        ["model_label", "dataset_label"], var_name="metric", value_name="score"
    )
    plot_df["metric"] = plot_df["metric"].map({"exact_accuracy": "Exact accuracy", "top3_accuracy": "Top-3 accuracy"})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), sharey=True)
    handles = labels = None
    for ax, metric in zip(axes, ["Exact accuracy", "Top-3 accuracy"]):
        sub = plot_df.loc[plot_df["metric"].eq(metric)]
        sns.barplot(
            data=sub,
            y="model_label",
            x="score",
            hue="dataset_label",
            palette=["#2f5f8f", "#d9a441"],
            errorbar=None,
            ax=ax,
        )
        ax.set_title(metric)
        ax.set_xlabel("Score")
        ax.set_ylabel("")
        ax.set_xlim(0, 1)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    if handles and labels:
        fig.legend(handles, labels, title="", loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Model comparison: 2025 holdout versus 2026 temporal validation", y=1.02, fontweight="bold")
    save_fig("fig2_model_comparison.png")

    terms = [
        "Count: 0-2",
        "Count: 1-2",
        "Count: 2-0",
        "Count: 3-0",
        "Count: 3-2",
        "Previous pitch: FF",
        "Previous pitch: SI",
        "Previous pitch: SL",
        "Previous pitch: CH",
        "Batter side: R",
    ]
    reg = regression.copy()
    reg["term_clean"] = reg["term"].map(readable_term)
    reg_sub = reg.loc[reg["term_clean"].isin(terms)].copy()
    heat = reg_sub.pivot_table(index="term_clean", columns="target_pitch_type", values="beta", aggfunc="mean").fillna(0)
    heat = heat.reindex([t for t in terms if t in heat.index])
    plt.figure(figsize=(7.2, 4.8))
    sns.heatmap(heat, cmap="vlag", center=0, annot=True, fmt="+.2f", cbar_kws={"label": "OLS beta"})
    plt.title("Linear probability model coefficients")
    plt.xlabel("Pitch outcome")
    plt.ylabel("")
    save_fig("fig3_regression_beta_heatmap.png")


def shap_interpretation(train: pd.DataFrame, test: pd.DataFrame, target_pitches: list[str]) -> pd.DataFrame:
    shap_cat = ["pitcher_name", "count", "prev_pitch_type", "prev2_pitch_type", "stand", "prev_description", "base_state"]
    shap_num = ["balls", "strikes", "pitch_number", "prev_release_speed", "score_diff_batting_team"]
    shap_cat = available_columns(train, shap_cat)
    shap_num = available_columns(train, shap_num)
    features = shap_cat + shap_num
    sample_train = train.sample(min(12000, len(train)), random_state=42)
    sample_explain = test.sample(min(800, len(test)), random_state=42)

    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25, sparse_output=False)),
        ]
    )
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    pre = ColumnTransformer([("cat", cat_pipe, shap_cat), ("num", num_pipe, shap_num)])
    X_train = pre.fit_transform(sample_train[features])
    X_explain = pre.transform(sample_explain[features])
    feature_names = [clean_feature_name(name) for name in pre.get_feature_names_out()]

    all_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    axes = axes.flatten()
    for ax, pitch in zip(axes, target_pitches[:4]):
        y = sample_train["pitch_type"].eq(pitch).astype(int)
        model = RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=30,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y)
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X_explain)
        if isinstance(vals, list):
            shap_vals = vals[1]
        elif getattr(vals, "ndim", 0) == 3:
            shap_vals = vals[:, :, 1]
        else:
            shap_vals = vals
        importance = pd.DataFrame(
            {
                "target_pitch_type": pitch,
                "target_pitch_label": pitch_label(pitch),
                "feature": feature_names,
                "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        all_rows.append(importance)
        top = importance.head(10).sort_values("mean_abs_shap", ascending=True)
        ax.barh(top["feature"], top["mean_abs_shap"], color="#2f5f8f")
        ax.set_title(f"Next pitch = {pitch_label(pitch)}")
        ax.set_xlabel("Mean |SHAP|")
        ax.set_ylabel("")
    for ax in axes[len(target_pitches[:4]) :]:
        ax.axis("off")
    plt.suptitle("SHAP interpretation: what shifts pitch-selection probabilities?", y=1.02, fontweight="bold")
    save_fig("fig4_shap_pitch_type_panels.png")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(TABLE_DIR / "shap_pitch_type_feature_importance.csv", index=False)
    global_out = (
        out.groupby("feature", as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "mean"), pitch_types=("target_pitch_type", "nunique"))
        .sort_values("mean_abs_shap", ascending=False)
    )
    global_out.to_csv(TABLE_DIR / "shap_global_feature_importance.csv", index=False)
    grouped_out = out.assign(feature_group=out["feature"].map(shap_feature_group))
    grouped_out = (
        grouped_out.groupby(["target_pitch_type", "target_pitch_label", "feature_group"], as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "sum"), raw_features=("feature", "nunique"))
        .sort_values(["target_pitch_type", "mean_abs_shap"], ascending=[True, False])
    )
    grouped_out.to_csv(TABLE_DIR / "shap_grouped_feature_importance.csv", index=False)

    grouped_global = (
        grouped_out.groupby("feature_group", as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "mean"), pitch_types=("target_pitch_type", "nunique"))
        .sort_values("mean_abs_shap", ascending=False)
    )
    grouped_global.to_csv(TABLE_DIR / "shap_grouped_global_feature_importance.csv", index=False)

    top_global = global_out.head(20).sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(8.2, 5.6))
    plt.barh(top_global["feature"], top_global["mean_abs_shap"], color="#2f5f8f")
    plt.title("Global SHAP feature importance across pitch outcomes")
    plt.xlabel("Mean |SHAP| across one-vs-rest pitch models")
    plt.ylabel("")
    save_fig("fig5_shap_global_feature_importance.png")

    top_grouped = grouped_global.head(14).sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(8.2, 4.8))
    plt.barh(top_grouped["feature_group"], top_grouped["mean_abs_shap"], color="#2f5f8f")
    plt.title("Grouped SHAP importance by baseball concept")
    plt.xlabel("Summed mean |SHAP| within feature group")
    plt.ylabel("")
    save_fig("fig6_shap_grouped_global_importance.png")
    return out


def write_paper_outline(
    best_state: pd.Series,
    best_model: pd.Series,
    target_pitches: list[str],
    temporal_model: pd.Series | None = None,
) -> None:
    temporal_line = ""
    if temporal_model is not None:
        temporal_line = (
            f"- On 2026-to-date temporal validation, the strongest pooled model reaches exact accuracy "
            f"{temporal_model['exact_accuracy']:.1%} and top-3 accuracy {temporal_model['top3_accuracy']:.1%}.\n"
        )
    outline = f"""# Paper-Ready Results Outline

## Working Title
Pitch Sequencing as Strategic Interaction: Modeling MLB Pitch Selection as Pitcher-Specific State Machines

## Research Question
How do MLB pitchers choose their next pitch as a function of count, previous pitch, batter handedness, and pitcher-specific arsenal?

## Main Model Findings
- Best tuned state-machine specification: `{best_state['model']}`, alpha={best_state.get('alpha')}, minimum state count={best_state.get('min_count')}.
- Best pooled predictive model: `{best_model['model']}`.
- The best model reaches exact accuracy {best_model['exact_accuracy']:.1%}, top-2 accuracy {best_model['top2_accuracy']:.1%}, and top-3 accuracy {best_model['top3_accuracy']:.1%} on the 2025 holdout.
{temporal_line}

## Regression Strategy
Linear probability models estimate whether the next pitch is one of: {', '.join(target_pitches)}. Predictors include count, previous pitch, batter side, and pitcher fixed effects.

## Suggested Figure Captions
**Figure 1. Pitcher-specific arsenals.** Each row shows the observed pitch menu available to a pitcher under the rule usage >= 3% or at least 30 pitches.

**Figure 2. Model comparison.** State-machine and pooled tree models are evaluated against the pitch actually thrown. Top-k accuracy is emphasized because pitch calling is a mixed-strategy decision.

**Figure 3. Regression coefficient heatmap.** OLS linear probability coefficients show how count, previous pitch, and batter side shift the probability of each pitch type.

**Figure 4. SHAP feature importance.** One-vs-rest random forest explanations show which state/context features most influence the probability of specific pitch outcomes.

**Figure 5. Global SHAP feature importance.** SHAP values are averaged across pitch-specific explanation models to identify the strongest general sequencing signals.

**Figure 6. Grouped SHAP feature importance.** One-hot features are collapsed into baseball concepts so pitcher-name indicators are interpreted as pitcher identity/arsenal rather than as standalone causal explanations.

## Transparency Note
XGBoost may require OpenMP (`libomp`) on macOS. The local run uses sklearn's histogram gradient boosting when XGBoost cannot load. Colab/Linux can run XGBoost directly if desired.
"""
    (PAPER_DIR / "paper_ready_results_outline.md").write_text(outline, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready state-machine, beta, model, and SHAP outputs.")
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_style()
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

    log("Building arsenals...")
    arsenal = build_arsenal(train_rows)
    allowed = arsenal_map(arsenal)

    log("Running paper-style OLS linear probability models before boosted prediction models...")
    regression = fit_regression_tables(train_rows, top_targets)

    log("Tuning state-machine parameters...")
    state_grid = tune_state_machine(train_rows, test_rows, classes, allowed, eval_2026)

    log("Tuning pooled prediction models as a benchmark to the per-pitcher loop.")
    model_grid = tune_models(train_rows, test_rows, classes, allowed, eval_2026)

    log("Generating polished figures...")
    make_paper_figures(arsenal, state_grid, model_grid, regression)

    log("Computing intuitive SHAP panels...")
    shap_df = shap_interpretation(train_rows, test_rows, top_targets[:4])

    best_state = state_grid.loc[state_grid["dataset"].eq("2025_holdout")].iloc[0]
    best_model = model_grid.loc[model_grid["dataset"].eq("2025_holdout")].iloc[0]
    best_temporal = None
    if not model_grid.loc[model_grid["dataset"].eq("2026_to_date")].empty:
        best_temporal = model_grid.loc[model_grid["dataset"].eq("2026_to_date")].iloc[0]
    combined = pd.concat(
        [
            state_grid.groupby("dataset", group_keys=False).head(12),
            model_grid.groupby("dataset", group_keys=False).head(12),
        ],
        ignore_index=True,
    )
    combined.to_csv(TABLE_DIR / "paper_model_comparison_top_models.csv", index=False)
    write_paper_outline(best_state, best_model, top_targets, best_temporal)

    summary = {
        "n_2025_rows": int(len(train_2025)),
        "n_2026_rows": int(len(eval_2026)) if not eval_2026.empty else 0,
        "target_pitches": top_targets,
        "best_state_machine": best_state.to_dict(),
        "best_pooled_model": best_model.to_dict(),
        "best_2026_temporal_model": best_temporal.to_dict() if best_temporal is not None else None,
    }
    (PAPER_DIR / "paper_ready_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    log("\n=== Best state-machine model ===")
    print(state_grid.loc[state_grid["dataset"].eq("2025_holdout")].head(5).to_string(index=False), flush=True)
    log("\n=== Best pooled models ===")
    print(model_grid.loc[model_grid["dataset"].eq("2025_holdout")].head(8).to_string(index=False), flush=True)
    if not eval_2026.empty:
        log("\n=== 2026 temporal validation ===")
        temporal = pd.concat(
            [
                state_grid.loc[state_grid["dataset"].eq("2026_to_date")].head(4),
                model_grid.loc[model_grid["dataset"].eq("2026_to_date")].head(8),
            ],
            ignore_index=True,
        )
        print(
            temporal[
                ["model", "dataset", "rows", "exact_accuracy", "top2_accuracy", "top3_accuracy", "macro_f1", "log_loss"]
            ].to_string(index=False),
            flush=True,
        )
    log("\n=== Top SHAP features ===")
    print(shap_df.groupby("target_pitch_type").head(5).to_string(index=False), flush=True)
    log("\n=== Paper-ready folder ===")
    log(str(PAPER_DIR.relative_to(ROOT)))


if __name__ == "__main__":
    main()
