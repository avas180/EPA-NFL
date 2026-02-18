"""
Stage 4: Construct the `future_points` target for EP modeling.

Grand-scheme purpose:
- The EP model predicts how many points the offense will score *before it
  loses possession*.
- This module computes that target by summing offensive points from the
  current play through the end of the current possession.
- By keeping this logic in one place, we ensure every model run uses the
  same target definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from possession import PossessionResult, load_clean_and_track_possessions


REQUIRED_COLUMNS = [
    "game_id",
    "play_id",
    "posteam",
    "drive",
    "posteam_score",
    "posteam_score_post",
]


@dataclass(frozen=True)
class TargetResult:
    """
    Container for possession-annotated data with future_points targets.

    The summary fields help sanity check the target before training models.
    """

    data: pd.DataFrame
    total_possessions: int
    total_scoring_plays: int


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise KeyError("Missing required columns for target construction: " + ", ".join(missing))


def _add_points_on_play(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Compute points scored by the offense on each play.

    nflfastR provides posteam_score (pre-play) and posteam_score_post (post-play).
    Their difference is the points the offense scored on that play.
    """

    data = frame.copy()

    data["points_on_play"] = data["posteam_score_post"] - data["posteam_score"]
    # Missing score fields occasionally appear on fringe rows; treat as 0 points.
    data["points_on_play"] = data["points_on_play"].fillna(0)

    return data


def add_future_points(frame: pd.DataFrame) -> TargetResult:
    """
    Add `future_points` to a possession-annotated DataFrame.

    For each play, future_points is the sum of offensive points from the
    current play through the end of the possession.
    """

    _ensure_columns(frame, REQUIRED_COLUMNS + ["possession_id"])

    # Sort by game and play to keep the reverse cumulative sum correct.
    data = frame.sort_values(["game_id", "play_id"]).reset_index(drop=True)

    data = _add_points_on_play(data)

    # Reverse cumulative sum within each possession gives points remaining.
    data["future_points"] = (
        data.groupby("possession_id")["points_on_play"]
        .transform(lambda s: s[::-1].cumsum()[::-1])
    )

    total_possessions = int(data["possession_id"].nunique())
    total_scoring_plays = int((data["points_on_play"] > 0).sum())

    return TargetResult(
        data=data,
        total_possessions=total_possessions,
        total_scoring_plays=total_scoring_plays,
    )


def load_clean_possessions_and_targets(years: Iterable[int]) -> tuple[PossessionResult, TargetResult]:
    """
    End-to-end helper: load, clean, track possessions, then add future_points.

    This is the intended entry point for notebooks and future pipeline scripts.
    """

    cleaned, possessions = load_clean_and_track_possessions(years)
    targets = add_future_points(possessions.data)
    return possessions, targets


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Construct future_points targets for EP modeling."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="Seasons to load (e.g., 2018 2019 2020)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write target data as parquet",
    )

    args = parser.parse_args()
    possessions, targets = load_clean_possessions_and_targets(args.years)

    print("Rows:", len(targets.data))
    print("Possessions:", targets.total_possessions)
    print("Scoring plays:", targets.total_scoring_plays)
    print("Mean future_points:", round(targets.data["future_points"].mean(), 4))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        targets.data.to_parquet(args.output, index=False)
        print("Wrote targets to:", args.output)
