#!/usr/bin/env python3
"""Export analysis artifacts into static assets for the React dashboard."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "output" if (ROOT / "output" / "next_pitch_prediction_audit_combined.csv").exists() else ROOT / "outputs"
PUBLIC = ROOT / "public"
PUBLIC_DATA = PUBLIC / "data"
PUBLIC_FIGURES = PUBLIC / "figures"
MIN_PAIR_PITCHES = int(os.environ.get("SITE_MATCHUP_MIN_PAIR_PITCHES", "10"))
MIN_SITUATION_PAIR_PITCHES = int(os.environ.get("SITE_MATCHUP_MIN_SITUATION_PAIR_PITCHES", "20"))
MIN_TRANSITION_STATE_TOTAL = int(os.environ.get("SITE_TRANSITION_MIN_STATE_TOTAL", "5"))


FIGURES = [
    {
        "id": "state-machine-heatmap",
        "title": "State Machine Transition Heatmap",
        "description": "Pitch-to-pitch transition patterns for the top pitcher sample.",
        "source": "figures/state_machine_transition_heatmap_top_pitchers.png",
    },
    {
        "id": "model-accuracy",
        "title": "Model Accuracy by Approach",
        "description": "Comparison of state-machine lookup and boosted prediction models.",
        "source": "figures/state_ols_model_comparison_accuracy.png",
    },
    {
        "id": "model-top3",
        "title": "Top-3 Pitch Menu Accuracy",
        "description": "How often the actual pitch appeared in the model's top three options.",
        "source": "figures/state_ols_model_comparison_top3.png",
    },
    {
        "id": "temporal-validation",
        "title": "2025 to 2026 Temporal Validation",
        "description": "A 2025-trained model evaluated on 2026-to-date pitch choices.",
        "source": "figures/temporal_validation_2025_to_2026.png",
    },
    {
        "id": "pitch-mix-shift",
        "title": "2025 vs 2026 Pitch Mix",
        "description": "Aggregate pitch usage among the selected high-volume starters.",
        "source": "figures/pitch_mix_2025_vs_2026.png",
    },
    {
        "id": "pitcher-accuracy",
        "title": "Prediction Accuracy by Pitcher",
        "description": "Pitcher-level exact-match and top-3 performance.",
        "source": "figures/prediction_accuracy_by_pitcher.png",
    },
    {
        "id": "transition-heatmap",
        "title": "League Pitch Transition Heatmap",
        "description": "Next-pitch probabilities conditional on the previous pitch.",
        "source": "figures/pitch_transition_heatmap_2025.png",
    },
    {
        "id": "count-pitch-mix",
        "title": "Pitch Mix by Count",
        "description": "How count changes the realistic pitch menu.",
        "source": "figures/count_pitch_mix_2025.png",
    },
    {
        "id": "next-pitch-model-sweep-top3",
        "title": "Model Sweep: Top-3 Pitch Menu",
        "description": "Ranked menu accuracy across lookup, logistic, random forest, and boosted approaches.",
        "source": "figures/next_pitch_model_sweep_top3.png",
    },
    {
        "id": "next-pitch-model-sweep-accuracy",
        "title": "Model Sweep: Exact Pitch Accuracy",
        "description": "Exact next-pitch accuracy by model family.",
        "source": "figures/next_pitch_model_sweep_accuracy.png",
    },
    {
        "id": "boosted-predicted-mix",
        "title": "Boosted Model: Actual vs Predicted Pitch Mix",
        "description": "How the boosted model's predicted menu compares with observed pitch usage.",
        "source": "figures/hist_gradient_boosting_arsenal_masked_actual_vs_predicted_pitch_mix.png",
    },
    {
        "id": "state-ols-arsenals",
        "title": "Pitcher Arsenals in State Models",
        "description": "Pitcher-specific repertoires used to constrain realistic pitch choices.",
        "source": "figures/state_ols_pitcher_arsenals.png",
    },
]


PITCH_LABELS = {
    "CH": "changeup (CH)",
    "CU": "curveball (CU)",
    "FC": "cutter (FC)",
    "FF": "4-seam fastball (FF)",
    "FS": "splitter (FS)",
    "KC": "knuckle curve (KC)",
    "SI": "sinker (SI)",
    "SL": "slider (SL)",
    "ST": "sweeper (ST)",
}

PITCH_CODE_PATTERN = re.compile(r"\(([A-Z]+)\)")
PITCH_PROB_PATTERN = re.compile(r"(.+?\([A-Z]+\))\s+([0-9.]+)%")


def cast_value(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        if any(token in value for token in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def project_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else ROOT / path


def output_path(path: str | Path) -> Path:
    return OUTPUTS / path


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    csv_path = project_path(path)
    if not csv_path.exists():
        return []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [{key: cast_value(value) for key, value in row.items()} for row in reader]


def read_json(path: str | Path) -> Any:
    json_path = project_path(path)
    if not json_path.exists():
        return None

    with json_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def copy_figures() -> list[dict[str, str]]:
    PUBLIC_FIGURES.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []

    for figure in FIGURES:
        source = output_path(figure["source"])
        if not source.exists():
            continue
        destination = PUBLIC_FIGURES / source.name
        shutil.copy2(source, destination)
        copied.append(
            {
                "id": figure["id"],
                "title": figure["title"],
                "description": figure["description"],
                "path": f"/figures/{source.name}",
            }
        )

    return copied


def aggregate_pitcher_season_mix() -> list[dict[str, Any]]:
    rows = []
    audit_files = [
        output_path("next_pitch_prediction_audit_2025_holdout.csv"),
        output_path("next_pitch_prediction_audit_2026_to_date.csv"),
    ]

    for audit_file in audit_files:
        audit_rows = read_csv(audit_file)
        counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        labels: dict[str, str] = {}

        for row in audit_rows:
            dataset = row.get("dataset")
            pitcher = row.get("pitcher_name")
            pitch_type = row.get("actual_pitch")
            pitch_label = row.get("actual_pitch_label") or PITCH_LABELS.get(str(pitch_type), str(pitch_type))
            if not dataset or not pitcher or not pitch_type:
                continue
            counts[(str(dataset), str(pitcher))][str(pitch_type)] += 1
            labels[str(pitch_type)] = str(pitch_label)

        for (dataset, pitcher), counter in counts.items():
            total = sum(counter.values())
            for pitch_type, pitches in counter.most_common():
                rows.append(
                    {
                        "dataset": dataset,
                        "pitcher_name": pitcher,
                        "pitch_type": pitch_type,
                        "pitch_label": labels.get(pitch_type, PITCH_LABELS.get(pitch_type, pitch_type)),
                        "pitches": pitches,
                        "share": pitches / total if total else 0,
                        "pitcher_dataset_pitches": total,
                    }
                )

    return rows


def pitch_code_from_label(label: str) -> str:
    match = PITCH_CODE_PATTERN.search(label)
    return match.group(1) if match else label


def parse_prediction_menu(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []

    menu = []
    for label, probability in PITCH_PROB_PATTERN.findall(str(value)):
        label = label.strip(" ,")
        pitch_type = pitch_code_from_label(label)
        menu.append(
            {
                "pitch_type": pitch_type,
                "pitch_label": label,
                "probability": float(probability) / 100,
            }
        )
    return menu


def menu_from_counts(counter: Counter[str], labels: dict[str, str]) -> list[dict[str, Any]]:
    total = sum(counter.values())
    if total == 0:
        return []

    return [
        {
            "pitch_type": pitch_type,
            "pitch_label": labels.get(pitch_type, PITCH_LABELS.get(pitch_type, pitch_type)),
            "probability": count / total,
            "count": count,
        }
        for pitch_type, count in counter.most_common()
    ]


def menu_from_probability_sums(
    probability_sums: dict[str, float],
    rows: int,
    labels: dict[str, str],
) -> list[dict[str, Any]]:
    if rows == 0:
        return []

    menu = [
        {
            "pitch_type": pitch_type,
            "pitch_label": labels.get(pitch_type, PITCH_LABELS.get(pitch_type, pitch_type)),
            "probability": probability_sum / rows,
        }
        for pitch_type, probability_sum in probability_sums.items()
    ]
    return sorted(menu, key=lambda row: row["probability"], reverse=True)


def state_machine_index() -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(output_path("state_machine_top3_transitions.csv")):
        if (row.get("state_total") or 0) < MIN_TRANSITION_STATE_TOTAL:
            continue
        key = (
            str(row.get("pitcher_name")),
            str(row.get("count")),
            str(row.get("prev_pitch_type")),
            str(row.get("stand")),
        )
        index[key].append(
            {
                "pitch_type": str(row.get("next_pitch_type")),
                "pitch_label": str(row.get("next_pitch_label")),
                "probability": row.get("transition_probability") or 0,
                "count": row.get("transition_count"),
                "state_total": row.get("state_total"),
                "rank": row.get("state_rank"),
            }
        )

    for rows in index.values():
        rows.sort(key=lambda row: row.get("rank") or 999)

    return index


def recommendation_index() -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    index = {}
    for row in read_csv(output_path("matchup_recommendations.csv")):
        key = (
            str(row.get("pitcher")),
            str(row.get("batter")),
            str(row.get("count")),
            str(row.get("previous_pitch")),
            str(row.get("previous_result")),
        )
        index[key] = row
    return index


def situation_id(parts: tuple[str, str, str, str, str]) -> str:
    return "__".join(
        re.sub(r"[^a-z0-9]+", "-", part.lower()).strip("-")
        for part in parts
    )


def build_matchup_pitch_menus(
    min_pair_pitches: int = MIN_PAIR_PITCHES,
    min_situation_pair_pitches: int = MIN_SITUATION_PAIR_PITCHES,
) -> dict[str, list[dict[str, Any]]]:
    combined_rows = read_csv(output_path("next_pitch_prediction_audit_combined.csv"))
    boosted_rows = read_csv(output_path("hist_gradient_boosting_arsenal_masked_prediction_audit.csv"))
    pair_counts = Counter(
        (str(row.get("pitcher_name")), str(row.get("batter_name")))
        for row in combined_rows
        if row.get("pitcher_name") and row.get("batter_name")
    )
    eligible_pairs = {pair for pair, count in pair_counts.items() if count >= min_pair_pitches}
    detailed_pairs = {pair for pair, count in pair_counts.items() if count >= min_situation_pair_pitches}
    pair_stand: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    pair_throws: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    pair_latest: dict[tuple[str, str], str] = defaultdict(str)

    groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    pitch_labels: dict[str, str] = {}

    for row in combined_rows:
        pitcher = str(row.get("pitcher_name"))
        batter = str(row.get("batter_name"))
        pair = (pitcher, batter)
        if pair not in eligible_pairs:
            continue

        pair_stand[pair][str(row.get("stand"))] += 1
        pair_throws[pair][str(row.get("p_throws"))] += 1
        pair_latest[pair] = max(pair_latest[pair], str(row.get("game_date") or ""))

        if pair not in detailed_pairs:
            continue

        key = (
            pitcher,
            batter,
            str(row.get("count")),
            str(row.get("prev_pitch_type")),
            str(row.get("prev_description")),
        )
        group = groups.setdefault(
            key,
            {
                "observations": 0,
                "actual": Counter(),
                "stand": Counter(),
                "p_throws": Counter(),
                "datasets": Counter(),
                "latest_game_date": "",
                "logistic_sums": defaultdict(float),
                "logistic_rows": 0,
                "boosted_sums": defaultdict(float),
                "boosted_rows": 0,
                "probability_labels": {},
            },
        )

        actual_pitch = str(row.get("actual_pitch"))
        actual_label = str(row.get("actual_pitch_label") or PITCH_LABELS.get(actual_pitch, actual_pitch))
        group["observations"] += 1
        group["actual"][actual_pitch] += 1
        group["stand"][str(row.get("stand"))] += 1
        group["p_throws"][str(row.get("p_throws"))] += 1
        group["datasets"][str(row.get("dataset"))] += 1
        group["latest_game_date"] = max(str(group["latest_game_date"]), str(row.get("game_date") or ""))
        pitch_labels[actual_pitch] = actual_label

        parsed_menu = parse_prediction_menu(row.get("top_3_predictions"))
        if parsed_menu:
            group["logistic_rows"] += 1
            for item in parsed_menu:
                group["logistic_sums"][item["pitch_type"]] += item["probability"]
                group["probability_labels"][item["pitch_type"]] = item["pitch_label"]

    for row in boosted_rows:
        key = (
            str(row.get("pitcher_name")),
            str(row.get("batter_name")),
            str(row.get("count")),
            str(row.get("prev_pitch_type")),
            str(row.get("prev_description")),
        )
        group = groups.get(key)
        if not group:
            continue

        parsed_menu = parse_prediction_menu(row.get("top_3_predictions"))
        if parsed_menu:
            group["boosted_rows"] += 1
            for item in parsed_menu:
                group["boosted_sums"][item["pitch_type"]] += item["probability"]
                group["probability_labels"][item["pitch_type"]] = item["pitch_label"]

    state_index = state_machine_index()
    recommendations = recommendation_index()
    situation_counts_by_pair: Counter[tuple[str, str]] = Counter()
    latest_by_pair: dict[tuple[str, str], str] = defaultdict(str)
    stand_by_pair: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    throws_by_pair: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    situations: list[dict[str, Any]] = []
    for key, group in groups.items():
        pitcher, batter, count, previous_pitch, previous_result = key
        pair = (pitcher, batter)
        stand = group["stand"].most_common(1)[0][0] if group["stand"] else ""
        p_throws = group["p_throws"].most_common(1)[0][0] if group["p_throws"] else ""
        probability_labels = {**pitch_labels, **group["probability_labels"]}

        observed_menu = menu_from_counts(group["actual"], probability_labels)
        logistic_menu = menu_from_probability_sums(group["logistic_sums"], group["logistic_rows"], probability_labels)
        boosted_menu = menu_from_probability_sums(group["boosted_sums"], group["boosted_rows"], probability_labels)
        state_menu = state_index.get((pitcher, count, previous_pitch, stand), [])
        recommendation = recommendations.get(key)

        method_menus = [
            {
                "id": "boosted",
                "label": "Boosted model",
                "sample_size": group["boosted_rows"],
                "menu": boosted_menu[:6],
            },
            {
                "id": "state_machine",
                "label": "State machine",
                "sample_size": state_menu[0].get("state_total") if state_menu else 0,
                "menu": state_menu[:6],
            },
            {
                "id": "observed_matchup",
                "label": "Observed matchup",
                "sample_size": group["observations"],
                "menu": observed_menu[:6],
            },
            {
                "id": "logistic_baseline",
                "label": "Baseline model",
                "sample_size": group["logistic_rows"],
                "menu": logistic_menu[:6],
            },
        ]

        situation_counts_by_pair[pair] += 1
        latest_by_pair[pair] = max(latest_by_pair[pair], group["latest_game_date"])
        stand_by_pair[pair].update(group["stand"])
        throws_by_pair[pair].update(group["p_throws"])

        situations.append(
            {
                "id": situation_id(key),
                "pitcher": pitcher,
                "batter": batter,
                "count": count,
                "previous_pitch": previous_pitch,
                "previous_result": previous_result,
                "button_label": "First pitch" if previous_pitch == "START" else f"{previous_pitch} {previous_result}",
                "batter_stand": stand,
                "pitcher_throws": p_throws,
                "observations": group["observations"],
                "pair_pitch_count": pair_counts[pair],
                "datasets": sorted(group["datasets"].keys()),
                "latest_game_date": group["latest_game_date"],
                "method_menus": method_menus,
                "recommendation": recommendation,
            }
        )

    pairs = []
    for pair, pair_pitch_count in pair_counts.items():
        if pair not in eligible_pairs:
            continue
        pitcher, batter = pair
        stand_counter = stand_by_pair[pair] if stand_by_pair[pair] else pair_stand[pair]
        throws_counter = throws_by_pair[pair] if throws_by_pair[pair] else pair_throws[pair]
        pairs.append(
            {
                "pitcher": pitcher,
                "batter": batter,
                "pair_pitch_count": pair_pitch_count,
                "situation_count": situation_counts_by_pair[pair],
                "batter_stand": stand_counter.most_common(1)[0][0] if stand_counter else "",
                "pitcher_throws": throws_counter.most_common(1)[0][0] if throws_counter else "",
                "latest_game_date": pair_latest[pair] or latest_by_pair[pair],
            }
        )

    pairs.sort(key=lambda row: (row["pair_pitch_count"], row["situation_count"]), reverse=True)
    situations.sort(key=lambda row: (row["pair_pitch_count"], row["observations"]), reverse=True)

    return {
        "pairs": pairs,
        "situations": situations,
    }


def top_pitcher_rows() -> list[dict[str, Any]]:
    top_100 = output_path("top_100_pitchers_2025.csv")
    if top_100.exists():
        return read_csv(top_100)
    return read_csv(output_path("top_20_pitchers_2025.csv"))


def filtered_state_machine_transitions() -> list[dict[str, Any]]:
    return [
        row
        for row in read_csv(output_path("state_machine_top3_transitions.csv"))
        if (row.get("state_total") or 0) >= MIN_TRANSITION_STATE_TOTAL
    ]


def build_payload() -> dict[str, Any]:
    model_metrics = read_json(output_path("model_metrics.json"))
    temporal_validation = read_json(output_path("temporal_validation_2025_to_2026.json"))
    matchup_pitch_menus = build_matchup_pitch_menus()
    paper_ready_summary = read_json(output_path("paper_ready/paper_ready_summary.json"))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": {
            "title": "What Comes Next?",
            "subtitle": "Modeling MLB pitch sequencing as a state machine and scouting tool.",
            "research_question": (
                "Can we model each pitcher's pitch-selection strategy as a state machine, then compare "
                "how pitcher-specific strategies change by count, previous pitch, batter context, and "
                "lineup handedness?"
            ),
            "data_note": (
                "Static dashboard artifacts were generated locally from pybaseball and Statcast outputs. "
                "The deployed Vercel app reads precomputed boosted-model predictions, state-machine "
                "transitions, matchup menus, JSON summaries, CSV-derived tables, and figures. "
                f"This build uses {OUTPUTS.name}/ with matchup pairs retained at {MIN_PAIR_PITCHES}+ pitches "
                f"and detailed exact-state menus retained at {MIN_SITUATION_PAIR_PITCHES}+ pitches."
            ),
        },
        "modelMetrics": model_metrics,
        "paperReadySummary": paper_ready_summary,
        "temporalValidation": temporal_validation,
        "modelSweepMetrics": read_csv(output_path("next_pitch_model_sweep_metrics.csv")),
        "stateModelMetrics": read_csv(output_path("state_ols_model_comparison_metrics.csv")),
        "topPitchers": top_pitcher_rows(),
        "predictionAccuracyByPitcher": read_csv(output_path("prediction_accuracy_by_pitcher.csv")),
        "pitcherArsenals": read_csv(output_path("pitcher_arsenals_2025_training.csv")),
        "pitcherSeasonPitchMix": aggregate_pitcher_season_mix(),
        "aggregatePitchMix": read_csv(output_path("pitch_mix_2025_vs_2026.csv")),
        "actualVsPredictedPitchMix": read_csv(output_path("hist_gradient_boosting_actual_vs_predicted_pitch_mix.csv")),
        "matchupRecommendations": read_csv(output_path("matchup_recommendations.csv")),
        "matchupPairs": matchup_pitch_menus["pairs"],
        "matchupPitchMenus": matchup_pitch_menus["situations"],
        "stateMachineTopTransitions": filtered_state_machine_transitions(),
        "contextEntropy": read_csv(output_path("pitch_choice_context_entropy.csv")),
        "figures": copy_figures(),
    }


def main() -> None:
    PUBLIC_DATA.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    output_path = PUBLIC_DATA / "site-data.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {output_path.relative_to(ROOT)}")
    print(f"Copied {len(payload['figures'])} figures")
    print(f"Source artifacts: {OUTPUTS.relative_to(ROOT)}")
    print(f"Matchup pair threshold: {MIN_PAIR_PITCHES}+ pitches")


if __name__ == "__main__":
    main()
