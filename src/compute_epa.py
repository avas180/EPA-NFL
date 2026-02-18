"""
Stage 8: Compute Expected Points Added (EPA) for each play.

Grand-scheme purpose:
- EPA quantifies the value of each play as the change in expected points.
- This module uses EP-before and EP-after to calculate EPA while handling
  scoring events and possession changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from xgboost import XGBRegressor

from clean_data import load_and_clean_pbp
from possession import add_possession_columns
from train_ep_model import FEATURE_COLUMNS, prepare_training_frame


@dataclass(frozen=True)
class EPAResult:
    """
    Container for EPA-annotated data and summary stats.
    """

    data: pd.DataFrame
    rows: int
    mean_epa: float


def _train_model(train_years: Iterable[int]) -> XGBRegressor:
    train_data = prepare_training_frame(train_years)
    X = train_data[FEATURE_COLUMNS]
    y = train_data["future_points"]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def _estimate_opponent_start_ep(model: XGBRegressor) -> float:
    """
    Estimate opponent EP at a typical kickoff/drive start state.

    We use a neutral state (1st & 10 at own 25, 15:00 remaining, tie game)
    to approximate the opponent's starting EP for scoring-play adjustments.
    """

    state = pd.DataFrame(
        {
            "down": [1],
            "ydstogo": [10],
            "yardline_100": [75],
            "half_seconds_remaining": [1800],
            "score_differential": [0],
            "goal_to_go": [0],
        }
    )
    return float(model.predict(state)[0])


def compute_epa(train_years: Iterable[int], predict_years: Iterable[int]) -> EPAResult:
    """
    Train an EP model and compute EPA for the requested seasons.
    """

    model = _train_model(train_years)
    opponent_start_ep = _estimate_opponent_start_ep(model)

    _, cleaned = load_and_clean_pbp(predict_years)
    data = add_possession_columns(cleaned.data).data

    data = data.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    data["ep_before"] = model.predict(data[FEATURE_COLUMNS])

    data["next_ep_before"] = data.groupby("game_id")["ep_before"].shift(-1)
    data["next_possession_id"] = data.groupby("game_id")["possession_id"].shift(-1)

    possession_change = data["possession_id"] != data["next_possession_id"]
    possession_change = possession_change.fillna(True)

    data["ep_after"] = data["next_ep_before"]
    data.loc[possession_change, "ep_after"] = -data.loc[possession_change, "next_ep_before"]
    data["ep_after"] = data["ep_after"].fillna(0)

    # Override scoring plays with explicit point values per roadmap guidance.
    if "touchdown" in data.columns:
        td_mask = data["touchdown"] == 1
        data.loc[td_mask, "ep_after"] = 6.0 - opponent_start_ep

    if "field_goal_result" in data.columns:
        fg_mask = data["field_goal_result"].str.lower() == "made"
        data.loc[fg_mask, "ep_after"] = 3.0 - opponent_start_ep

    data["epa"] = data["ep_after"] - data["ep_before"]

    return EPAResult(data=data, rows=len(data), mean_epa=float(data["epa"].mean()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute EPA using an XGBoost EP model."
    )
    parser.add_argument(
        "--train-years",
        nargs="+",
        type=int,
        required=True,
        help="Seasons to train the EP model (e.g., 2018 2019 2020)",
    )
    parser.add_argument(
        "--predict-years",
        nargs="+",
        type=int,
        required=True,
        help="Seasons to compute EPA (e.g., 2025)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write EPA data as parquet",
    )

    args = parser.parse_args()
    result = compute_epa(args.train_years, args.predict_years)

    print("Rows:", result.rows)
    print("Mean EPA:", round(result.mean_epa, 4))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.data.to_parquet(args.output, index=False)
        print("Wrote EPA data to:", args.output)
