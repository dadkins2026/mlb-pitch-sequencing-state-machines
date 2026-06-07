#!/usr/bin/env python3
"""Build a first-pass MLB pitch sequencing dataset, models, and reports."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("PYBASEBALL_CACHE", str(Path(".cache/pybaseball").resolve()))

import numpy as np
import pandas as pd
from pybaseball import (
    cache,
    playerid_reverse_lookup,
    statcast_pitcher,
    statcast_pitcher_arsenal_stats,
)
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "output"
CACHE_DIR = ROOT / ".cache"

PITCH_NAME = {
    "FF": "4-seam fastball",
    "SI": "sinker",
    "FC": "cutter",
    "SL": "slider",
    "ST": "sweeper",
    "SV": "slurve",
    "CH": "changeup",
    "CU": "curveball",
    "KC": "knuckle curve",
    "FS": "splitter",
    "KN": "knuckleball",
    "FO": "forkball",
    "EP": "eephus",
    "SC": "screwball",
}

WHIFF_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
    "foul_tip",
}

CALLED_STRIKE_DESCRIPTIONS = {"called_strike"}

BALL_DESCRIPTIONS = {
    "ball",
    "blocked_ball",
    "pitchout",
    "hit_by_pitch",
    "intent_ball",
    "automatic_ball",
}

POSITIVE_PITCHER_EVENTS = {
    "strikeout": 0.25,
    "field_out": 0.10,
    "force_out": 0.10,
    "grounded_into_double_play": 0.35,
    "double_play": 0.35,
    "strikeout_double_play": 0.40,
}

NEGATIVE_PITCHER_EVENTS = {
    "single": -0.30,
    "double": -0.60,
    "triple": -0.90,
    "home_run": -1.40,
    "walk": -0.25,
    "hit_by_pitch": -0.25,
    "sac_fly": -0.20,
    "field_error": -0.25,
}


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, CACHE_DIR / "matplotlib", CACHE_DIR / "pybaseball"):
        path.mkdir(parents=True, exist_ok=True)


def sample_label(top_n: int) -> str:
    return f"top{top_n}"


def sequence_features_path(season: int, top_n: int) -> Path:
    return PROCESSED_DIR / f"sequence_features_{season}_{sample_label(top_n)}.parquet"


def temporal_sequence_features_path(train_season: int, eval_season: int, top_n: int) -> Path:
    return PROCESSED_DIR / f"sequence_features_{eval_season}_same_{train_season}_{sample_label(top_n)}.parquet"


def temporal_raw_label(train_season: int, top_n: int) -> str:
    return f"statcast_same_{train_season}_{sample_label(top_n)}"


def merge_with_diagnostics(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: list[str],
    *,
    how: str,
    name: str,
) -> pd.DataFrame:
    marker = f"__{name}_matched"
    right = right.copy()
    right[marker] = 1
    duplicate_right_keys = int(right.duplicated(on).sum())
    log(
        f"[merge:{name}] before left_rows={len(left):,}, right_rows={len(right):,}, "
        f"right_duplicate_keys={duplicate_right_keys:,}, keys={on}"
    )
    merged = left.merge(right, on=on, how=how, validate="many_to_one")
    unmatched = int(merged[marker].isna().sum()) if marker in merged.columns else 0
    log(f"[merge:{name}] after rows={len(merged):,}, unmatched_left_rows={unmatched:,}")
    return merged.drop(columns=[marker])


def add_lineup_handedness_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"game_pk", "batter", "stand"}
    if not required <= set(df.columns):
        log("[lineup] skipped lineup handedness features; missing game_pk, batter, or stand")
        return df

    out = df.copy()
    if {"inning_topbot", "home_team", "away_team"} <= set(out.columns):
        out["batting_team"] = np.where(out["inning_topbot"].eq("Top"), out["away_team"], out["home_team"])
        keys = ["game_pk", "batting_team"]
    else:
        out["batting_team"] = "game_unknown"
        keys = ["game_pk", "batting_team"]

    lineup = out[keys + ["batter", "stand"]].dropna(subset=["game_pk", "batter", "stand"]).drop_duplicates()
    if lineup.empty:
        for col in ["lineup_left_share", "lineup_right_share", "lineup_switch_share", "lineup_batter_count"]:
            out[col] = np.nan
        return out

    stand_counts = (
        lineup.assign(
            lineup_left=lambda x: x["stand"].astype(str).eq("L").astype(int),
            lineup_right=lambda x: x["stand"].astype(str).eq("R").astype(int),
            lineup_switch=lambda x: x["stand"].astype(str).eq("S").astype(int),
        )
        .groupby(keys, as_index=False)
        .agg(
            lineup_batter_count=("batter", "nunique"),
            lineup_left_batters=("lineup_left", "sum"),
            lineup_right_batters=("lineup_right", "sum"),
            lineup_switch_batters=("lineup_switch", "sum"),
        )
    )
    denom = stand_counts["lineup_batter_count"].replace(0, np.nan)
    stand_counts["lineup_left_share"] = stand_counts["lineup_left_batters"] / denom
    stand_counts["lineup_right_share"] = stand_counts["lineup_right_batters"] / denom
    stand_counts["lineup_switch_share"] = stand_counts["lineup_switch_batters"] / denom
    stand_counts = stand_counts[
        keys + ["lineup_batter_count", "lineup_left_share", "lineup_right_share", "lineup_switch_share"]
    ]
    return merge_with_diagnostics(out, stand_counts, keys, how="left", name="lineup_handedness")


def to_numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=[])
    return pd.to_numeric(series, errors="coerce")


def display_name(last_first: Any) -> str:
    if pd.isna(last_first):
        return "Unknown"
    text = str(last_first)
    if "," not in text:
        return text
    last, first = text.split(",", 1)
    return f"{first.strip()} {last.strip()}"


def safe_read_parquet(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    return None


def select_pitchers(season: int, top_n: int) -> pd.DataFrame:
    log(f"Fetching {season} Statcast pitcher arsenal leaderboard...")
    arsenal = statcast_pitcher_arsenal_stats(season, minPA=25)
    if arsenal.empty:
        raise RuntimeError("The pitcher arsenal leaderboard returned no rows.")

    required = {"player_id", "last_name, first_name", "pitches"}
    missing = required - set(arsenal.columns)
    if missing:
        raise RuntimeError(f"Pitcher arsenal table missing expected columns: {sorted(missing)}")

    pitchers = (
        arsenal.groupby(["player_id", "last_name, first_name"], as_index=False)["pitches"]
        .sum()
        .sort_values("pitches", ascending=False)
        .head(top_n)
        .rename(columns={"player_id": "pitcher_id", "last_name, first_name": "pitcher_name_raw"})
    )
    pitchers["pitcher_name"] = pitchers["pitcher_name_raw"].map(display_name)
    pitchers["rank_by_arsenal_pitches"] = np.arange(1, len(pitchers) + 1)
    pitchers = pitchers[
        ["rank_by_arsenal_pitches", "pitcher_id", "pitcher_name", "pitcher_name_raw", "pitches"]
    ]
    pitchers.to_csv(OUTPUT_DIR / f"top_{top_n}_pitchers_{season}.csv", index=False)
    return pitchers


def pull_pitcher_data(
    season: int,
    start_date: str,
    end_date: str,
    pitchers: pd.DataFrame,
    raw_label: str = "statcast",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row in pitchers.itertuples(index=False):
        pitcher_id = int(row.pitcher_id)
        name = str(row.pitcher_name)
        path = RAW_DIR / f"{raw_label}_{season}_pitcher_{pitcher_id}.parquet"
        cached = safe_read_parquet(path)
        if cached is not None:
            log(f"Using cached Statcast rows for {name} ({pitcher_id}): {len(cached):,}")
            frames.append(cached)
            continue

        log(f"Pulling Statcast pitch-level rows for {name} ({pitcher_id})...")
        try:
            df = statcast_pitcher(start_date, end_date, pitcher_id)
        except Exception as exc:
            warnings.warn(f"Failed to pull pitcher {name} ({pitcher_id}): {exc}")
            continue

        if df.empty:
            warnings.warn(f"No Statcast rows returned for {name} ({pitcher_id}).")
            continue

        df.to_parquet(path, index=False)
        log(f"  saved {len(df):,} rows to {path.relative_to(ROOT)}")
        frames.append(df)

    if not frames:
        raise RuntimeError("No pitch-level data could be pulled.")

    data = pd.concat(frames, ignore_index=True)
    if "game_type" in data.columns:
        before = len(data)
        data = data.loc[data["game_type"].eq("R")].copy()
        log(f"Filtered to regular season game_type == R: {before:,} -> {len(data):,} rows")
    return data


def add_player_names(
    data: pd.DataFrame,
    pitchers: pd.DataFrame,
    lookup_path: Path | None = None,
) -> pd.DataFrame:
    data = data.copy()
    pitcher_map = pitchers.set_index("pitcher_id")["pitcher_name"].to_dict()
    data["pitcher_name"] = data["pitcher"].map(pitcher_map)

    lookup_path = lookup_path or PROCESSED_DIR / "batter_lookup_2025.csv"
    batter_ids = sorted(data["batter"].dropna().astype(int).unique().tolist())

    lookup: pd.DataFrame | None = None
    if lookup_path.exists():
        lookup = pd.read_csv(lookup_path)
    else:
        log(f"Looking up {len(batter_ids):,} batter names via pybaseball/Chadwick register...")
        try:
            lookup = playerid_reverse_lookup(batter_ids, key_type="mlbam")
            lookup.to_csv(lookup_path, index=False)
        except Exception as exc:
            warnings.warn(f"Could not resolve batter names; keeping MLBAM ids. Error: {exc}")

    if lookup is not None and {"key_mlbam", "name_first", "name_last"} <= set(lookup.columns):
        lookup = lookup.copy()
        lookup["batter_name"] = (
            lookup["name_first"].fillna("").astype(str).str.title().str.strip()
            + " "
            + lookup["name_last"].fillna("").astype(str).str.title().str.strip()
        ).str.strip()
        batter_map = lookup.set_index("key_mlbam")["batter_name"].to_dict()
        data["batter_name"] = data["batter"].map(batter_map)
    else:
        data["batter_name"] = "Batter " + data["batter"].astype("Int64").astype(str)

    data["batter_name"] = data["batter_name"].fillna("Batter " + data["batter"].astype("Int64").astype(str))
    return data


def build_sequence_features(data: pd.DataFrame, output_path: Path | None = None) -> pd.DataFrame:
    df = data.copy()
    df = df.loc[df["pitch_type"].notna()].copy()

    for col in [
        "balls",
        "strikes",
        "outs_when_up",
        "inning",
        "pitch_number",
        "zone",
        "release_speed",
        "release_spin_rate",
        "plate_x",
        "plate_z",
        "pfx_x",
        "pfx_z",
        "launch_speed",
        "launch_angle",
        "estimated_woba_using_speedangle",
        "delta_run_exp",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sort_cols = [col for col in ["game_date", "game_pk", "at_bat_number", "pitch_number"] if col in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    group_cols = [col for col in ["game_pk", "at_bat_number"] if col in df.columns]
    if len(group_cols) != 2:
        raise RuntimeError("Expected game_pk and at_bat_number columns in Statcast data.")
    pa = df.groupby(group_cols, sort=False)

    df["prev_pitch_type"] = pa["pitch_type"].shift(1).fillna("START")
    df["prev2_pitch_type"] = pa["pitch_type"].shift(2).fillna("START")
    df["prev_description"] = pa["description"].shift(1).fillna("START")
    df["prev_zone"] = pa["zone"].shift(1) if "zone" in df.columns else np.nan
    df["prev_release_speed"] = pa["release_speed"].shift(1) if "release_speed" in df.columns else np.nan
    df["prev_plate_x"] = pa["plate_x"].shift(1) if "plate_x" in df.columns else np.nan
    df["prev_plate_z"] = pa["plate_z"].shift(1) if "plate_z" in df.columns else np.nan

    df["count"] = df["balls"].fillna(0).astype(int).astype(str) + "-" + df["strikes"].fillna(0).astype(int).astype(str)
    df["base_state"] = (
        np.where(df.get("on_1b", pd.Series(index=df.index)).notna(), "1", "-")
        + np.where(df.get("on_2b", pd.Series(index=df.index)).notna(), "2", "-")
        + np.where(df.get("on_3b", pd.Series(index=df.index)).notna(), "3", "-")
    )
    df = add_lineup_handedness_features(df)

    if {"bat_score", "fld_score"} <= set(df.columns):
        df["score_diff_batting_team"] = (
            pd.to_numeric(df["bat_score"], errors="coerce") - pd.to_numeric(df["fld_score"], errors="coerce")
        )
    elif {"home_score", "away_score"} <= set(df.columns):
        df["score_diff_batting_team"] = (
            pd.to_numeric(df["home_score"], errors="coerce") - pd.to_numeric(df["away_score"], errors="coerce")
        )
    else:
        df["score_diff_batting_team"] = np.nan

    desc = df["description"].fillna("")
    events = df["events"].fillna("")
    launch_speed = df["launch_speed"] if "launch_speed" in df.columns else pd.Series(np.nan, index=df.index)
    est_woba = (
        df["estimated_woba_using_speedangle"]
        if "estimated_woba_using_speedangle" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    df["whiff"] = desc.isin(WHIFF_DESCRIPTIONS).astype(int)
    df["called_strike"] = desc.isin(CALLED_STRIKE_DESCRIPTIONS).astype(int)
    df["weak_contact"] = (
        df.get("type", pd.Series("", index=df.index)).fillna("").eq("X")
        & ((launch_speed <= 85) | (est_woba <= 0.250))
    ).astype(int)
    df["whiff_or_weak"] = ((df["whiff"] == 1) | (df["weak_contact"] == 1)).astype(int)

    if "delta_run_exp" in df.columns and df["delta_run_exp"].notna().sum() > len(df) * 0.7:
        df["pitcher_run_value"] = -df["delta_run_exp"]
        df["run_value_source"] = "negative_delta_run_exp"
    else:
        reward = pd.Series(0.0, index=df.index)
        reward += desc.isin(CALLED_STRIKE_DESCRIPTIONS).astype(float) * 0.05
        reward += desc.isin(WHIFF_DESCRIPTIONS).astype(float) * 0.12
        reward += df["weak_contact"].astype(float) * 0.08
        reward -= desc.isin(BALL_DESCRIPTIONS).astype(float) * 0.04
        for event, value in POSITIVE_PITCHER_EVENTS.items():
            reward += events.eq(event).astype(float) * value
        for event, value in NEGATIVE_PITCHER_EVENTS.items():
            reward += events.eq(event).astype(float) * value
        df["pitcher_run_value"] = reward
        df["run_value_source"] = "heuristic_pitcher_reward"

    df["pitch_name_clean"] = df["pitch_type"].map(PITCH_NAME).fillna(df.get("pitch_name", df["pitch_type"]))
    output_path = output_path or sequence_features_path(2025, 100)
    df.to_parquet(output_path, index=False)
    return df


def split_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_size: float = 0.20,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    model_df = df.loc[df[target_col].notna(), feature_cols + [target_col]].copy()
    y = model_df[target_col]
    X = model_df[feature_cols]

    stratify_y = None
    if stratify and y.nunique() > 1 and y.value_counts().min() >= 2:
        stratify_y = y

    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify_y)


def make_next_pitch_model(categorical_cols: list[str], numeric_cols: list[str]) -> Pipeline:
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
    pre = ColumnTransformer(
        [("cat", cat_pipe, categorical_cols), ("num", num_pipe, numeric_cols)],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", pre),
            ("model", LogisticRegression(max_iter=700, n_jobs=-1, class_weight="balanced")),
        ]
    )


def make_hgb_classifier(categorical_cols: list[str], numeric_cols: list[str]) -> Pipeline:
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    pre = ColumnTransformer(
        [("cat", cat_pipe, categorical_cols), ("num", num_pipe, numeric_cols)],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", pre),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.07,
                    max_iter=180,
                    max_leaf_nodes=31,
                    l2_regularization=0.03,
                    random_state=42,
                ),
            ),
        ]
    )


def make_hgb_regressor(categorical_cols: list[str], numeric_cols: list[str]) -> Pipeline:
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    pre = ColumnTransformer(
        [("cat", cat_pipe, categorical_cols), ("num", num_pipe, numeric_cols)],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", pre),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.06,
                    max_iter=220,
                    max_leaf_nodes=31,
                    l2_regularization=0.03,
                    random_state=42,
                ),
            ),
        ]
    )


def top_k_accuracy(model: Pipeline, X: pd.DataFrame, y: pd.Series, k: int = 3) -> float:
    probs = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    k = min(k, len(classes))
    top_k = classes[np.argsort(probs, axis=1)[:, -k:]]
    return float(np.mean([actual in row for actual, row in zip(y.to_numpy(), top_k)]))


def train_models(df: pd.DataFrame) -> dict[str, Any]:
    rare_counts = df["pitch_type"].value_counts()
    keep_pitch_types = rare_counts.loc[rare_counts >= 80].index
    model_df = df.loc[df["pitch_type"].isin(keep_pitch_types)].copy()

    next_cat = [
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
    next_num = [
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
    next_features = [c for c in next_cat + next_num if c in model_df.columns]
    next_cat = [c for c in next_cat if c in model_df.columns]
    next_num = [c for c in next_num if c in model_df.columns]

    X_train, X_test, y_train, y_test = split_xy(model_df, next_features, "pitch_type")

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)

    next_model = make_next_pitch_model(next_cat, next_num)
    log("Training multinomial logistic model for next pitch type...")
    next_model.fit(X_train, y_train)
    next_pred = next_model.predict(X_test)
    next_report = classification_report(y_test, next_pred, output_dict=True, zero_division=0)
    pd.DataFrame(next_report).transpose().to_csv(OUTPUT_DIR / "next_pitch_classification_report.csv")

    outcome_cat = next_cat + ["pitch_type"]
    outcome_num = next_num + ["release_speed", "release_spin_rate", "zone", "plate_x", "plate_z", "pfx_x", "pfx_z"]
    outcome_cat = [c for c in outcome_cat if c in model_df.columns]
    outcome_num = [c for c in outcome_num if c in model_df.columns]
    outcome_features = outcome_cat + outcome_num

    Xo_train, Xo_test, yo_train, yo_test = split_xy(model_df, outcome_features, "whiff_or_weak")
    outcome_model = make_hgb_classifier(outcome_cat, outcome_num)
    log("Training gradient-boosted outcome classifier for whiff/weak contact...")
    outcome_model.fit(Xo_train, yo_train.astype(int))
    outcome_prob = outcome_model.predict_proba(Xo_test)[:, 1]
    outcome_pred = (outcome_prob >= 0.5).astype(int)

    Xr_train, Xr_test, yr_train, yr_test = split_xy(
        model_df.dropna(subset=["pitcher_run_value"]),
        outcome_features,
        "pitcher_run_value",
        stratify=False,
    )
    run_value_model = make_hgb_regressor(outcome_cat, outcome_num)
    log("Training gradient-boosted regressor for pitcher run value...")
    run_value_model.fit(Xr_train, yr_train.astype(float))
    rv_pred = run_value_model.predict(Xr_test)

    metrics = {
        "rows_modelled": int(len(model_df)),
        "pitch_types_modelled": sorted(model_df["pitch_type"].unique().tolist()),
        "next_pitch_dummy_accuracy": float(accuracy_score(y_test, dummy_pred)),
        "next_pitch_logistic_accuracy": float(accuracy_score(y_test, next_pred)),
        "next_pitch_logistic_top3_accuracy": top_k_accuracy(next_model, X_test, y_test, k=3),
        "next_pitch_logistic_macro_f1": float(f1_score(y_test, next_pred, average="macro")),
        "outcome_whiff_or_weak_positive_rate": float(model_df["whiff_or_weak"].mean()),
        "outcome_classifier_accuracy": float(accuracy_score(yo_test, outcome_pred)),
        "outcome_classifier_brier": float(brier_score_loss(yo_test, outcome_prob)),
        "outcome_classifier_auc": float(roc_auc_score(yo_test, outcome_prob))
        if yo_test.nunique() == 2
        else None,
        "run_value_rmse": float(mean_squared_error(yr_test, rv_pred) ** 0.5),
        "run_value_r2": float(r2_score(yr_test, rv_pred)),
        "run_value_source": str(model_df["run_value_source"].iloc[0]),
        "next_pitch_features": next_features,
        "outcome_features": outcome_features,
    }

    with open(OUTPUT_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model_df": model_df,
        "metrics": metrics,
        "next_model": next_model,
        "outcome_model": outcome_model,
        "run_value_model": run_value_model,
        "next_features": next_features,
        "outcome_features": outcome_features,
    }


def pitch_label(code: str) -> str:
    return f"{PITCH_NAME.get(code, code)} ({code})"


def pitcher_repertoire(df: pd.DataFrame, pitcher_id: int, min_usage: float = 0.03) -> list[str]:
    sub = df.loc[df["pitcher"].astype(int).eq(int(pitcher_id)), "pitch_type"]
    counts = sub.value_counts()
    if counts.empty:
        return []
    usage = counts / counts.sum()
    candidates = usage.loc[(usage >= min_usage) & (counts >= 20)].head(6).index.tolist()
    return candidates or counts.head(4).index.tolist()


def top_probabilities(model: Pipeline, X: pd.DataFrame, n: int = 3) -> list[tuple[str, float]]:
    probs = model.predict_proba(X)[0]
    classes = np.asarray(model.classes_)
    order = np.argsort(probs)[::-1][:n]
    return [(str(classes[i]), float(probs[i])) for i in order]


def recommend_for_context(
    context: pd.Series,
    repertoire: list[str],
    next_model: Pipeline,
    outcome_model: Pipeline,
    run_value_model: Pipeline,
    next_features: list[str],
    outcome_features: list[str],
) -> dict[str, Any]:
    X_next = pd.DataFrame([context[next_features].to_dict()])
    likely = top_probabilities(next_model, X_next, n=3)

    rows = []
    for pitch_type in repertoire:
        candidate = context.copy()
        candidate["pitch_type"] = pitch_type
        rows.append(candidate[outcome_features].to_dict())

    X_candidates = pd.DataFrame(rows)
    good_prob = outcome_model.predict_proba(X_candidates)[:, 1]
    rv = run_value_model.predict(X_candidates)
    recs = pd.DataFrame(
        {
            "pitch_type": repertoire,
            "pitch_label": [pitch_label(p) for p in repertoire],
            "pred_whiff_or_weak_prob": good_prob,
            "pred_pitcher_run_value": rv,
        }
    ).sort_values(["pred_pitcher_run_value", "pred_whiff_or_weak_prob"], ascending=False)

    return {
        "likely": likely,
        "recommendations": recs,
        "likely_pitch_type": likely[0][0],
        "likely_pitch_probability": likely[0][1],
        "recommended_pitch_type": str(recs.iloc[0]["pitch_type"]),
        "recommended_pitcher_run_value": float(recs.iloc[0]["pred_pitcher_run_value"]),
        "recommended_whiff_or_weak_prob": float(recs.iloc[0]["pred_whiff_or_weak_prob"]),
    }


def choose_report_contexts(df: pd.DataFrame, n_reports: int) -> pd.DataFrame:
    candidates = df.loc[
        df["prev_pitch_type"].ne("START") & df["pitcher_name"].notna() & df["batter_name"].notna()
    ].copy()
    if candidates.empty:
        return df.head(n_reports).copy()

    pair_counts = (
        candidates.groupby(["pitcher", "pitcher_name", "batter", "batter_name"], as_index=False)
        .size()
        .rename(columns={"size": "pair_pitch_count"})
        .sort_values("pair_pitch_count", ascending=False)
    )

    rows = []
    seen_pitchers: set[int] = set()
    for pair in pair_counts.itertuples(index=False):
        pitcher_id = int(pair.pitcher)
        if pitcher_id in seen_pitchers and len(rows) < n_reports // 2:
            continue
        sub = candidates.loc[
            candidates["pitcher"].astype(int).eq(pitcher_id)
            & candidates["batter"].astype(int).eq(int(pair.batter))
        ].sort_values([c for c in ["game_date", "game_pk", "at_bat_number", "pitch_number"] if c in candidates.columns])
        if sub.empty:
            continue
        row = sub.iloc[-1].copy()
        row["pair_pitch_count"] = pair.pair_pitch_count
        rows.append(row)
        seen_pitchers.add(pitcher_id)
        if len(rows) >= n_reports:
            break

    return pd.DataFrame(rows)


def build_matchup_reports(models: dict[str, Any], n_reports: int) -> pd.DataFrame:
    df = models["model_df"]
    contexts = choose_report_contexts(df, n_reports)
    rows = []
    md_lines = ["# 2025 Pitch Sequencing Matchup Reports", ""]

    for _, context in contexts.iterrows():
        pitcher_id = int(context["pitcher"])
        repertoire = pitcher_repertoire(df, pitcher_id)
        if not repertoire:
            continue

        result = recommend_for_context(
            context=context,
            repertoire=repertoire,
            next_model=models["next_model"],
            outcome_model=models["outcome_model"],
            run_value_model=models["run_value_model"],
            next_features=models["next_features"],
            outcome_features=models["outcome_features"],
        )

        top_recs = result["recommendations"].head(3)
        likely_text = ", ".join(f"{pitch_label(code)} {prob:.1%}" for code, prob in result["likely"])
        rec_text = "; ".join(
            f"{row.pitch_label}: RV {row.pred_pitcher_run_value:+.3f}, W/W {row.pred_whiff_or_weak_prob:.1%}"
            for row in top_recs.itertuples(index=False)
        )

        observed_pitch = str(context["pitch_type"])
        row_out = {
            "pitcher": context["pitcher_name"],
            "batter": context["batter_name"],
            "count": context["count"],
            "previous_pitch": context["prev_pitch_type"],
            "previous_result": context["prev_description"],
            "actual_pitch": observed_pitch,
            "actual_pitch_label": pitch_label(observed_pitch),
            "actual_description": context.get("description", ""),
            "likely_pitch": result["likely_pitch_type"],
            "likely_pitch_label": pitch_label(result["likely_pitch_type"]),
            "likely_pitch_probability": result["likely_pitch_probability"],
            "recommended_pitch": result["recommended_pitch_type"],
            "recommended_pitch_label": pitch_label(result["recommended_pitch_type"]),
            "recommended_pitcher_run_value": result["recommended_pitcher_run_value"],
            "recommended_whiff_or_weak_prob": result["recommended_whiff_or_weak_prob"],
            "top_likely_pitches": likely_text,
            "top_recommended_pitches": rec_text,
            "pair_pitch_count_in_sample": int(context.get("pair_pitch_count", 0)),
        }
        rows.append(row_out)

        md_lines.extend(
            [
                f"## {context['pitcher_name']} vs {context['batter_name']}",
                "",
                f"- Context: count {context['count']}, inning {int(context['inning']) if pd.notna(context['inning']) else 'NA'}, "
                f"outs {int(context['outs_when_up']) if pd.notna(context['outs_when_up']) else 'NA'}, "
                f"previous pitch {pitch_label(str(context['prev_pitch_type']))}, previous result `{context['prev_description']}`.",
                f"- Model likely pitch: {pitch_label(result['likely_pitch_type'])} "
                f"({result['likely_pitch_probability']:.1%}). Top likely: {likely_text}.",
                f"- Model recommended pitch: {pitch_label(result['recommended_pitch_type'])}; "
                f"expected pitcher run value {result['recommended_pitcher_run_value']:+.3f}, "
                f"whiff/weak-contact probability {result['recommended_whiff_or_weak_prob']:.1%}.",
                f"- Candidate recommendation table: {rec_text}.",
                f"- Historical pitch in this row: {pitch_label(observed_pitch)}; Statcast description `{context.get('description', '')}`.",
                "",
            ]
        )

    report_df = pd.DataFrame(rows)
    report_df.to_csv(OUTPUT_DIR / "matchup_recommendations.csv", index=False)
    (OUTPUT_DIR / "matchup_reports.md").write_text("\n".join(md_lines), encoding="utf-8")
    return report_df


def print_summary(
    pitchers: pd.DataFrame,
    features: pd.DataFrame,
    models: dict[str, Any],
    reports: pd.DataFrame,
    feature_output_path: Path,
) -> None:
    log(f"\n=== Top {len(pitchers)} high-volume pitchers from 2025 Statcast arsenal table ===")
    print(
        pitchers[["rank_by_arsenal_pitches", "pitcher_name", "pitcher_id", "pitches"]]
        .to_string(index=False),
        flush=True,
    )

    log("\n=== Engineered feature sample ===")
    sample_cols = [
        "game_date",
        "pitcher_name",
        "batter_name",
        "count",
        "prev_pitch_type",
        "prev_description",
        "pitch_type",
        "description",
        "whiff",
        "weak_contact",
        "pitcher_run_value",
    ]
    sample_cols = [c for c in sample_cols if c in features.columns]
    print(features[sample_cols].head(12).to_string(index=False), flush=True)

    log("\n=== Model metrics ===")
    metric_keys = [
        "rows_modelled",
        "pitch_types_modelled",
        "next_pitch_dummy_accuracy",
        "next_pitch_logistic_accuracy",
        "next_pitch_logistic_top3_accuracy",
        "next_pitch_logistic_macro_f1",
        "outcome_whiff_or_weak_positive_rate",
        "outcome_classifier_accuracy",
        "outcome_classifier_auc",
        "outcome_classifier_brier",
        "run_value_rmse",
        "run_value_r2",
        "run_value_source",
    ]
    for key in metric_keys:
        log(f"{key}: {models['metrics'].get(key)}")

    log("\n=== Hitter-specific matchup reports ===")
    display_cols = [
        "pitcher",
        "batter",
        "count",
        "previous_pitch",
        "actual_pitch_label",
        "likely_pitch_label",
        "likely_pitch_probability",
        "recommended_pitch_label",
        "recommended_pitcher_run_value",
        "recommended_whiff_or_weak_prob",
    ]
    if reports.empty:
        log("No matchup reports were generated.")
    else:
        print(reports[display_cols].to_string(index=False), flush=True)

    log("\n=== Files written ===")
    for path in [
        OUTPUT_DIR / f"top_{len(pitchers)}_pitchers_2025.csv",
        feature_output_path,
        OUTPUT_DIR / "model_metrics.json",
        OUTPUT_DIR / "next_pitch_classification_report.csv",
        OUTPUT_DIR / "matchup_recommendations.csv",
        OUTPUT_DIR / "matchup_reports.md",
    ]:
        log(str(path.relative_to(ROOT)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 2025 MLB pitch sequencing project prototype.")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--start-date", default="2025-03-18")
    parser.add_argument("--end-date", default="2025-09-28")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--reports", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    cache.enable()

    log(
        f"Running pitch sequencing pipeline for {args.season} "
        f"({args.start_date} through {args.end_date}), top_n={args.top_n}"
    )
    pitchers = select_pitchers(args.season, args.top_n)
    raw = pull_pitcher_data(args.season, args.start_date, args.end_date, pitchers)
    named = add_player_names(raw, pitchers)
    feature_output_path = sequence_features_path(args.season, args.top_n)
    features = build_sequence_features(named, output_path=feature_output_path)
    log(f"Engineered {len(features):,} pitch rows and {features.shape[1]:,} columns.")
    models = train_models(features)
    reports = build_matchup_reports(models, args.reports)
    print_summary(pitchers, features, models, reports, feature_output_path)


if __name__ == "__main__":
    main()
