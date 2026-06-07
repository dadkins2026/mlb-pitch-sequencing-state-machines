#!/usr/bin/env python3
"""Generate a final-paper visualization pack from completed model outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pitch_sequence_pipeline import ROOT, pitch_label, log


OUTPUT_DIR = ROOT / "output"
PACK_DIR = OUTPUT_DIR / "final_paper_visuals"
FIGURE_DIR = PACK_DIR / "figures"
TABLE_DIR = PACK_DIR / "tables"

PITCH_ORDER = ["FF", "SI", "SL", "CH", "CU", "FC", "ST", "KC", "FS", "CS", "SV"]
PITCH_COLORS = {
    "FF": "#2f5f8f",
    "SI": "#5a8f6f",
    "SL": "#b44b4b",
    "CH": "#d9a441",
    "CU": "#6f5aa8",
    "FC": "#7a7f87",
    "ST": "#b06aa2",
    "KC": "#476d7c",
    "FS": "#8f6b4a",
    "CS": "#5f8fa3",
    "SV": "#9a6f8f",
}


def ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.title_fontsize": 8.5,
        }
    )


def save_fig(name: str) -> Path:
    path = FIGURE_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"wrote {path.relative_to(ROOT)}")
    return path


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / path)


def model_label(name: str) -> str:
    labels = {
        "state_machine_pitcher_count_prev_stand": "State machine",
        "state_machine_pitcher_count_prev_stand_arsenal_masked": "State machine + arsenal",
        "hgb_current_features": "HGB current features",
        "hgb_current_features_arsenal_masked": "HGB current + arsenal",
        "hgb_recency_matchup": "HGB recency/matchup",
        "hgb_recency_matchup_arsenal_masked": "HGB recency/matchup + arsenal",
        "hgb_current_game_recent_start": "HGB current-game/recent-start",
        "hgb_current_game_recent_start_arsenal_masked": "HGB current-game/recent-start + arsenal",
        "xgboost_recency_matchup": "XGBoost recency/matchup",
        "xgboost_recency_matchup_arsenal_masked": "XGBoost recency/matchup + arsenal",
        "xgboost_current_game_recent_start": "XGBoost current-game/recent-start",
        "xgboost_current_game_recent_start_arsenal_masked": "XGBoost current-game/recent-start + arsenal",
        "stacked_hgb_xgb_state": "Stacked HGB + XGB + state",
        "stacked_hgb_xgb_state_arsenal_masked": "Stacked HGB + XGB + state + arsenal",
    }
    return labels.get(name, name.replace("_", " "))


def pct_axis(ax) -> None:
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")


def load_comparison_models() -> pd.DataFrame:
    frames = []
    for rel in [
        "paper_ready/tables/recency_booster_model_comparison.csv",
        "stacked_exact/tables/stacked_current_game_model_comparison.csv",
        "paper_ready/tables/state_machine_tuning_grid.csv",
    ]:
        path = OUTPUT_DIR / rel
        if path.exists():
            df = pd.read_csv(path)
            df["source_file"] = rel
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if "accuracy" in out.columns and "exact_accuracy" not in out.columns:
        out = out.rename(columns={"accuracy": "exact_accuracy"})
    out["model_label"] = out["model"].map(model_label)
    return out


def select_narrative_models(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "state_machine_pitcher_count_prev_stand_arsenal_masked",
        "hgb_current_features_arsenal_masked",
        "hgb_recency_matchup_arsenal_masked",
        "hgb_current_game_recent_start",
        "xgboost_current_game_recent_start",
        "stacked_hgb_xgb_state",
        "stacked_hgb_xgb_state_arsenal_masked",
    ]
    out = df.loc[df["model"].isin(keep)].copy()
    return out.sort_values(["dataset", "exact_accuracy"], ascending=[True, False])


def plot_model_comparison() -> None:
    comp = select_narrative_models(load_comparison_models())
    comp.to_csv(TABLE_DIR / "figure_model_comparison_data.csv", index=False)
    plot_df = comp.melt(
        id_vars=["model_label", "dataset"],
        value_vars=["exact_accuracy", "top3_accuracy"],
        var_name="metric",
        value_name="score",
    )
    plot_df["metric"] = plot_df["metric"].map({"exact_accuracy": "Exact accuracy", "top3_accuracy": "Top-3 accuracy"})
    order = (
        comp.loc[comp["dataset"].eq("2025_holdout")]
        .sort_values("exact_accuracy", ascending=False)["model_label"]
        .drop_duplicates()
        .tolist()
    )
    plot_df["model_label"] = pd.Categorical(plot_df["model_label"], categories=order, ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), sharey=True)
    for ax, metric in zip(axes, ["Exact accuracy", "Top-3 accuracy"]):
        sub = plot_df.loc[plot_df["metric"].eq(metric)]
        sns.barplot(data=sub, x="score", y="model_label", hue="dataset", ax=ax, errorbar=None, palette=["#2f5f8f", "#d9a441"])
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlim(0.30 if metric == "Exact accuracy" else 0.75, 0.90 if metric == "Top-3 accuracy" else 0.47)
        pct_axis(ax)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Predicting the next pitch: exact choices and top-3 shortlists", y=1.03, fontweight="bold")
    save_fig("01_model_comparison_exact_top3.png")


def plot_accuracy_lift() -> None:
    comp = load_comparison_models()
    temporal = json.loads((OUTPUT_DIR / "temporal_validation_2025_to_2026.json").read_text())
    dummy_2025 = next(row["accuracy"] for row in temporal if row["dataset"] == "2025_holdout" and row["task"] == "next_pitch_dummy")
    rows = [
        {"dataset": "2025_holdout", "stage": "Dummy baseline", "exact_accuracy": dummy_2025},
        {"dataset": "2025_holdout", "stage": "State machine", "exact_accuracy": 0.422923},
        {"dataset": "2025_holdout", "stage": "Current HGB", "exact_accuracy": 0.432956},
        {"dataset": "2025_holdout", "stage": "Current-game HGB", "exact_accuracy": 0.435689},
        {"dataset": "2025_holdout", "stage": "Stacked model", "exact_accuracy": 0.441404},
        {"dataset": "2026_to_date", "stage": "State machine", "exact_accuracy": 0.374104},
        {"dataset": "2026_to_date", "stage": "Current HGB", "exact_accuracy": 0.390511},
        {"dataset": "2026_to_date", "stage": "Current-game HGB", "exact_accuracy": 0.397353},
        {"dataset": "2026_to_date", "stage": "Stacked model", "exact_accuracy": 0.403855},
    ]
    lift = pd.DataFrame(rows)
    lift.to_csv(TABLE_DIR / "figure_accuracy_lift_data.csv", index=False)
    plt.figure(figsize=(8.8, 4.8))
    sns.lineplot(data=lift, x="stage", y="exact_accuracy", hue="dataset", marker="o", linewidth=2.4, palette=["#2f5f8f", "#d9a441"])
    plt.title("Exact-accuracy lift from baseline to stacked model")
    plt.xlabel("")
    plt.ylabel("Exact accuracy")
    plt.ylim(0.30, 0.46)
    plt.gca().yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    plt.xticks(rotation=25, ha="right")
    save_fig("02_exact_accuracy_lift.png")


def plot_temporal_scatter() -> None:
    comp = select_narrative_models(load_comparison_models())
    pivot = comp.pivot_table(index="model_label", columns="dataset", values=["exact_accuracy", "top3_accuracy"], aggfunc="max")
    rows = []
    for model in pivot.index:
        try:
            rows.append(
                {
                    "model_label": model,
                    "exact_2025": pivot.loc[model, ("exact_accuracy", "2025_holdout")],
                    "exact_2026": pivot.loc[model, ("exact_accuracy", "2026_to_date")],
                    "top3_2025": pivot.loc[model, ("top3_accuracy", "2025_holdout")],
                    "top3_2026": pivot.loc[model, ("top3_accuracy", "2026_to_date")],
                }
            )
        except KeyError:
            continue
    scatter = pd.DataFrame(rows).dropna()
    scatter.to_csv(TABLE_DIR / "figure_temporal_scatter_data.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for ax, x, y, title, lims in [
        (axes[0], "exact_2025", "exact_2026", "Temporal robustness: exact accuracy", (0.36, 0.46)),
        (axes[1], "top3_2025", "top3_2026", "Temporal robustness: top-3 accuracy", (0.78, 0.88)),
    ]:
        sns.scatterplot(data=scatter, x=x, y=y, s=85, color="#2f5f8f", ax=ax)
        for _, row in scatter.iterrows():
            ax.text(row[x] + 0.001, row[y] + 0.001, row["model_label"].replace(" + ", "\n+ "), fontsize=7.2)
        ax.set_title(title)
        ax.set_xlabel("2025 holdout")
        ax.set_ylabel("2026 to date")
        ax.set_xlim(*lims)
        ax.set_ylim(lims[0] - 0.02, lims[1] - 0.02)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    save_fig("03_temporal_generalization_scatter.png")


def plot_context_entropy() -> None:
    entropy = read_csv("pitch_choice_context_entropy.csv")
    entropy["context_label"] = entropy["context"].map(
        {
            "pitcher": "Pitcher",
            "pitcher_count": "Pitcher + count",
            "pitcher_prev": "Pitcher + previous pitch",
            "pitcher_count_prev": "Pitcher + count + previous",
            "pitcher_count_prev_stand": "Pitcher + count + previous + side",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    sns.barplot(data=entropy, x="weighted_entropy_bits", y="context_label", color="#2f5f8f", ax=axes[0])
    axes[0].set_title("State context reduces pitch-choice entropy")
    axes[0].set_xlabel("Weighted entropy (bits)")
    axes[0].set_ylabel("")
    sns.barplot(data=entropy, x="weighted_mode_share", y="context_label", color="#d9a441", ax=axes[1])
    axes[1].set_title("Most likely pitch becomes more dominant")
    axes[1].set_xlabel("Weighted mode share")
    axes[1].set_ylabel("")
    axes[1].xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    save_fig("04_state_context_entropy_mode_share.png")


def plot_count_grid() -> None:
    count_df = read_csv("prediction_accuracy_by_count.csv")
    count_order = ["0-0", "0-1", "0-2", "1-0", "1-1", "1-2", "2-0", "2-1", "2-2", "3-0", "3-1", "3-2"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax, metric, title in [(axes[0], "exact_match", "Exact accuracy by count"), (axes[1], "top3", "Top-3 accuracy by count")]:
        heat = count_df.pivot_table(index="dataset", columns="count", values=metric).reindex(columns=count_order)
        sns.heatmap(heat, annot=True, fmt=".0%", cmap="Blues", cbar=False, ax=ax, linewidths=0.5)
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.set_ylabel("")
    save_fig("05_count_accuracy_heatmap.png")


def plot_pitcher_predictability() -> None:
    df = read_csv("prediction_accuracy_by_pitcher.csv")
    sub = df.loc[df["dataset"].eq("2025_holdout")].copy()
    top = sub.sort_values("exact_match", ascending=False).head(12).assign(group="Most predictable")
    low = sub.sort_values("exact_match", ascending=True).head(12).assign(group="Least predictable")
    plot_df = pd.concat([top, low], ignore_index=True)
    plot_df["pitcher_order"] = pd.Categorical(plot_df["pitcher_name"], categories=plot_df.sort_values("exact_match")["pitcher_name"], ordered=True)
    plt.figure(figsize=(8.2, 6.8))
    sns.barplot(data=plot_df, x="exact_match", y="pitcher_order", hue="group", dodge=False, palette=["#2f5f8f", "#d9a441"])
    plt.title("Pitcher-level predictability varies sharply")
    plt.xlabel("Exact-match accuracy")
    plt.ylabel("")
    plt.gca().xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    plt.legend(title="")
    save_fig("06_pitcher_predictability_extremes.png")


def plot_pitcher_arsenals() -> None:
    arsenal = read_csv("paper_ready/tables/pitcher_arsenals_allowed_only.csv")
    top_pitchers = (
        arsenal.groupby("pitcher_name")["pitcher_total_pitches"].max().sort_values(ascending=False).head(35).index
    )
    sub = arsenal.loc[arsenal["pitcher_name"].isin(top_pitchers)].copy()
    pitch_cols = [p for p in PITCH_ORDER if p in sub["pitch_type"].unique()]
    pivot = sub.pivot_table(index="pitcher_name", columns="pitch_type", values="usage_rate", fill_value=0).reindex(columns=pitch_cols, fill_value=0)
    pivot = pivot.loc[sub.groupby("pitcher_name")["pitcher_total_pitches"].max().sort_values().index.intersection(pivot.index)]
    ax = pivot.plot(kind="barh", stacked=True, figsize=(9.0, 8.4), color=[PITCH_COLORS.get(p, "#777777") for p in pivot.columns], width=0.82)
    ax.set_title("Pitcher-specific arsenals are the strategic menu")
    ax.set_xlabel("Usage share")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(title="Pitch", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    save_fig("07_pitcher_arsenal_usage_top35.png")


def plot_ols_intercepts() -> None:
    beta = read_csv("ols_beta_pitcher_pitch_intercepts.csv")
    arsenal = read_csv("paper_ready/tables/pitcher_arsenals_allowed_only.csv")
    top_pitchers = arsenal.groupby("pitcher_name")["pitcher_total_pitches"].max().sort_values(ascending=False).head(35).index
    beta = beta.loc[beta["pitcher_name"].isin(top_pitchers)].set_index("pitcher_name")
    cols = [p for p in PITCH_ORDER if p in beta.columns]
    beta = beta[cols]
    plt.figure(figsize=(8.8, 8.2))
    sns.heatmap(beta, cmap="vlag", center=0, linewidths=0.1, cbar_kws={"label": "OLS pitch-type beta"})
    plt.title("OLS pitcher fixed effects by pitch type")
    plt.xlabel("Pitch type")
    plt.ylabel("")
    save_fig("08_ols_pitcher_pitch_beta_heatmap.png")


def clean_term(term: str) -> str:
    if term == "Intercept":
        return "Intercept"
    patterns = [
        (r'C\(count, Treatment\(reference="0-0"\)\)\[T\.(.+?)\]', r"Count \1"),
        (r'C\(prev_pitch_type, Treatment\(reference="START"\)\)\[T\.(.+?)\]', r"Previous pitch \1"),
        (r'C\(stand, Treatment\(reference="L"\)\)\[T\.(.+?)\]', r"Batter side \1"),
        (r"C\(pitcher_name\)\[T\.(.+?)\]", r"Pitcher \1"),
    ]
    out = term
    for pattern, replacement in patterns:
        out = re.sub(pattern, replacement, out)
    return out


def plot_ols_context_heatmap() -> None:
    reg = read_csv("paper_ready/tables/ols_pooled_fixed_effects_coefficients.csv")
    reg["term_clean"] = reg["term"].map(clean_term)
    terms = [
        "Count 0-1",
        "Count 0-2",
        "Count 1-2",
        "Count 2-0",
        "Count 3-0",
        "Count 3-2",
        "Previous pitch FF",
        "Previous pitch SI",
        "Previous pitch SL",
        "Previous pitch CH",
        "Batter side R",
    ]
    sub = reg.loc[reg["term_clean"].isin(terms)].copy()
    heat = sub.pivot_table(index="term_clean", columns="target_pitch_type", values="beta", aggfunc="mean").reindex(terms)
    cols = [p for p in PITCH_ORDER if p in heat.columns]
    plt.figure(figsize=(8.2, 5.6))
    sns.heatmap(heat[cols], cmap="vlag", center=0, annot=True, fmt="+.2f", linewidths=0.4, cbar_kws={"label": "OLS beta"})
    plt.title("OLS context effects: count, sequence, and batter side")
    plt.xlabel("Pitch outcome")
    plt.ylabel("")
    save_fig("09_ols_context_beta_heatmap.png")


def plot_ols_feature_summary() -> None:
    df = read_csv("ols_beta_feature_summary.csv")
    keep = df.loc[df["feature"].str.startswith(("count_", "prev_pitch_type_", "stand_"))].copy()
    keep = keep.loc[~keep["feature"].isin(["prev_pitch_type_UN"])]
    top = keep.sort_values("mean_abs_beta", ascending=False).head(25)
    top["label"] = top["target_pitch_type"] + " | " + top["feature"].str.replace("_", " ", regex=False)
    plt.figure(figsize=(8.8, 7.0))
    sns.barplot(data=top.sort_values("mean_abs_beta"), x="mean_abs_beta", y="label", hue="target_pitch_type", dodge=False, palette=PITCH_COLORS)
    plt.title("Largest average OLS context effects across pitchers")
    plt.xlabel("Mean absolute beta")
    plt.ylabel("")
    plt.legend(title="Pitch", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    save_fig("10_ols_top_context_effects.png")


def plot_shap_grouped() -> None:
    df = read_csv("paper_ready/tables/shap_grouped_global_feature_importance.csv")
    plt.figure(figsize=(8.3, 4.8))
    sns.barplot(data=df.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature_group", color="#2f5f8f")
    plt.title("Grouped SHAP: baseball concepts driving pitch choice")
    plt.xlabel("Mean |SHAP| across pitch outcomes")
    plt.ylabel("")
    save_fig("11_shap_grouped_global_importance.png")


def plot_shap_pitch_specific() -> None:
    df = read_csv("paper_ready/tables/shap_pitch_type_feature_importance.csv")
    targets = [p for p in ["FF", "SI", "CH", "SL"] if p in df["target_pitch_type"].unique()]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.4))
    axes = axes.flatten()
    for ax, pitch in zip(axes, targets):
        sub = df.loc[df["target_pitch_type"].eq(pitch)].head(10).sort_values("mean_abs_shap")
        ax.barh(sub["feature"], sub["mean_abs_shap"], color=PITCH_COLORS.get(pitch, "#2f5f8f"))
        ax.set_title(pitch_label(pitch))
        ax.set_xlabel("Mean |SHAP|")
        ax.set_ylabel("")
    for ax in axes[len(targets) :]:
        ax.axis("off")
    fig.suptitle("Pitch-specific SHAP explanations", y=1.02, fontweight="bold")
    save_fig("12_shap_pitch_specific_panels.png")


def plot_shap_grouped_heatmap() -> None:
    df = read_csv("paper_ready/tables/shap_grouped_feature_importance.csv")
    heat = df.pivot_table(index="feature_group", columns="target_pitch_type", values="mean_abs_shap", aggfunc="sum").fillna(0)
    cols = [p for p in ["FF", "SI", "CH", "SL"] if p in heat.columns]
    order = heat[cols].mean(axis=1).sort_values(ascending=False).index
    plt.figure(figsize=(7.8, 5.2))
    sns.heatmap(heat.loc[order, cols], cmap="YlGnBu", linewidths=0.4, cbar_kws={"label": "Grouped mean |SHAP|"})
    plt.title("Grouped SHAP by pitch outcome")
    plt.xlabel("Pitch outcome")
    plt.ylabel("")
    save_fig("13_shap_grouped_by_pitch_heatmap.png")


def plot_transition_matrix() -> None:
    trans = read_csv("pitch_transition_probabilities_2025.csv").set_index("prev_pitch_type")
    cols = [p for p in PITCH_ORDER if p in trans.columns]
    rows = [p for p in PITCH_ORDER if p in trans.index]
    plt.figure(figsize=(7.4, 6.0))
    sns.heatmap(trans.loc[rows, cols], cmap="Blues", annot=True, fmt=".0%", linewidths=0.4, cbar_kws={"label": "Transition probability"})
    plt.title("Empirical pitch transition matrix")
    plt.xlabel("Next pitch")
    plt.ylabel("Previous pitch")
    save_fig("14_pitch_transition_matrix.png")


def plot_state_certainty_distribution() -> None:
    top = read_csv("state_machine_top3_transitions.csv")
    top1 = top.loc[(top["state_rank"].eq(1)) & (top["state_total"] >= 20)].copy()
    plt.figure(figsize=(8.4, 4.8))
    sns.histplot(data=top1, x="transition_probability", weights="state_total", bins=30, color="#2f5f8f")
    plt.title("How often does a state have a dominant next pitch?")
    plt.xlabel("Top transition probability for state")
    plt.ylabel("Weighted state count")
    plt.gca().xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    save_fig("15_state_machine_certainty_distribution.png")


def plot_pitch_mix_shift() -> None:
    mix = read_csv("pitch_mix_2025_vs_2026.csv")
    if {"pitch_type", "season", "share"}.issubset(mix.columns):
        plot_df = mix.copy()
    else:
        plot_df = mix.melt(id_vars=["pitch_type"], var_name="season", value_name="share")
    order = [p for p in PITCH_ORDER if p in plot_df["pitch_type"].unique()]
    plt.figure(figsize=(8.6, 4.8))
    sns.barplot(data=plot_df, x="pitch_type", y="share", hue="season", order=order, palette=["#2f5f8f", "#d9a441"], errorbar=None)
    plt.title("Pitch mix shifts from 2025 training to 2026 validation")
    plt.xlabel("Pitch type")
    plt.ylabel("Share")
    plt.gca().yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    save_fig("16_pitch_mix_2025_vs_2026.png")


def plot_outcome_by_pitch() -> None:
    out = read_csv("pitch_type_outcome_summary_2025.csv")
    metrics = [c for c in ["whiff_rate", "weak_contact_rate", "whiff_or_weak_rate", "mean_pitcher_run_value"] if c in out.columns]
    if not metrics:
        return
    order = [p for p in PITCH_ORDER if p in out["pitch_type"].unique()]
    fig, axes = plt.subplots(1, min(3, len(metrics)), figsize=(12, 4.2))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, metric in zip(axes, metrics[:3]):
        sns.barplot(data=out, x="pitch_type", y=metric, order=order, color="#2f5f8f", ax=ax, errorbar=None)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylabel("")
        if metric.endswith("_rate"):
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.suptitle("Pitch outcomes contextualize why arsenals matter", y=1.03, fontweight="bold")
    save_fig("17_pitch_type_outcomes.png")


def write_index(figures: list[tuple[str, str]]) -> None:
    lines = [
        "# Final Paper Visualization Pack",
        "",
        "Generated from existing project outputs. These figures are intended for paper drafting and narrative selection.",
        "",
        "| File | Suggested paper use |",
        "| --- | --- |",
    ]
    for name, use in figures:
        lines.append(f"| `figures/{name}` | {use} |")
    (PACK_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    set_style()
    figures = [
        ("01_model_comparison_exact_top3.png", "Main predictive comparison across state-machine, HGB, XGBoost, and stacked models."),
        ("02_exact_accuracy_lift.png", "Simple narrative arc showing exact-accuracy improvements from baseline to stacked model."),
        ("03_temporal_generalization_scatter.png", "Shows whether model improvements survive out-of-time 2026 validation."),
        ("04_state_context_entropy_mode_share.png", "State-machine argument: context reduces uncertainty and raises mode share."),
        ("05_count_accuracy_heatmap.png", "Count-level grid showing where prediction is easier or harder."),
        ("06_pitcher_predictability_extremes.png", "Pitcher heterogeneity: some arsenals/sequences are much more readable."),
        ("07_pitcher_arsenal_usage_top35.png", "Arsenal visualization for baseball intuition and pitcher-specific modeling."),
        ("08_ols_pitcher_pitch_beta_heatmap.png", "Regression/fixed-effect beta matrix by pitcher and pitch type."),
        ("09_ols_context_beta_heatmap.png", "OLS context effects for count, previous pitch, and batter side."),
        ("10_ols_top_context_effects.png", "Largest average OLS context effects across pitchers."),
        ("11_shap_grouped_global_importance.png", "Grouped SHAP interpretation without over-reading individual one-hot labels."),
        ("12_shap_pitch_specific_panels.png", "Pitch-specific SHAP explanations."),
        ("13_shap_grouped_by_pitch_heatmap.png", "SHAP feature groups by pitch outcome."),
        ("14_pitch_transition_matrix.png", "Empirical sequence transition matrix."),
        ("15_state_machine_certainty_distribution.png", "Distribution of state-machine certainty across observed states."),
        ("16_pitch_mix_2025_vs_2026.png", "Training vs validation pitch-mix comparison."),
        ("17_pitch_type_outcomes.png", "Outcome context by pitch type."),
    ]
    plot_model_comparison()
    plot_accuracy_lift()
    plot_temporal_scatter()
    plot_context_entropy()
    plot_count_grid()
    plot_pitcher_predictability()
    plot_pitcher_arsenals()
    plot_ols_intercepts()
    plot_ols_context_heatmap()
    plot_ols_feature_summary()
    plot_shap_grouped()
    plot_shap_pitch_specific()
    plot_shap_grouped_heatmap()
    plot_transition_matrix()
    plot_state_certainty_distribution()
    plot_pitch_mix_shift()
    plot_outcome_by_pitch()
    write_index(figures)
    log(f"Final paper visualization pack: {PACK_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
