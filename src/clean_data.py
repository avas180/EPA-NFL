"""
Stage 2: Clean raw NFL play-by-play data into a modeling-ready dataset.

Grand-scheme purpose:
- The EP model is only as good as the game states we feed it.
- This module removes non-plays, special teams, and edge cases that distort
  state values, and it enforces the minimal input features required by the
  roadmap.
- Every downstream step (target construction, modeling, EPA) should depend on
  this cleaned output to keep assumptions consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from load_data import DATA_ROOT, LoadResult, load_pbp_years


REQUIRED_STATE_COLUMNS = [
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
    "score_differential",
    "goal_to_go",
]

SPECIAL_TEAMS_PLAY_TYPES = {
    "kickoff",
    "kickoff_return",
    "punt",
    "punt_return",
    "extra_point",
    "on_side_kick",
}


@dataclass(frozen=True)
class CleanResult:
    """
    Container for cleaned data and a small summary of what was removed.

    Tracking these counts makes data preparation auditable and helps debug
    unexpected model behavior later.
    """

    data: pd.DataFrame
    removed_rows: dict[str, int]


def _remove_non_plays(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove obvious non-plays (timeouts, penalties with no play, etc.).

    The nflfastR schema includes a `no_play` flag; we fall back to play_type
    when needed so this logic still works across seasons.
    """

    start = len(frame)

    if "no_play" in frame.columns:
        frame = frame[frame["no_play"] == 0]
    elif "play_type" in frame.columns:
        frame = frame[frame["play_type"] != "no_play"]
    elif "play_type_nfl" in frame.columns:
        frame = frame[frame["play_type_nfl"] != "no_play"]

    removed = start - len(frame)
    return frame, removed


def _remove_special_teams(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove special teams plays for the first EP model iteration.

    Special teams have very different state dynamics, so the roadmap
    explicitly defers them until later. We keep field goal attempts
    because they represent offensive decision states we want the EP
    model to learn.
    """

    start = len(frame)

    if "play_type" in frame.columns:
        frame = frame[~frame["play_type"].isin(SPECIAL_TEAMS_PLAY_TYPES)]
    elif "play_type_nfl" in frame.columns:
        frame = frame[~frame["play_type_nfl"].isin(SPECIAL_TEAMS_PLAY_TYPES)]
    elif "special_teams_play" in frame.columns:
        # Fall back to the broad flag only if play_type is unavailable.
        frame = frame[frame["special_teams_play"] == 0]

    removed = start - len(frame)
    return frame, removed


def _remove_low_clock(frame: pd.DataFrame, min_seconds: int = 30) -> tuple[pd.DataFrame, int]:
    """
    Remove plays with fewer than `min_seconds` remaining in the half.

    End-of-half plays can be extremely context-specific (kneel-downs,
    desperation laterals) and add noise to the EP surface.
    """

    start = len(frame)
    if "half_seconds_remaining" in frame.columns:
        frame = frame[frame["half_seconds_remaining"] >= min_seconds]

    removed = start - len(frame)
    return frame, removed


def _enforce_state_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop rows that are missing any required state features.

    The EP model relies only on these state features, so missing values are
    unusable for training.
    """

    start = len(frame)
    missing_cols = [col for col in REQUIRED_STATE_COLUMNS if col not in frame.columns]
    if missing_cols:
        raise KeyError(
            "Missing required columns for modeling: " + ", ".join(missing_cols)
        )

    frame = frame.dropna(subset=REQUIRED_STATE_COLUMNS)

    # Remove invalid downs (e.g., 0 or >4) to keep the model state space clean.
    frame = frame[frame["down"].between(1, 4)]

    removed = start - len(frame)
    return frame, removed


def _validate_yardline(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Ensure `yardline_100` stays within [0, 100].

    In nflfastR this is already offense-relative distance to the opponent goal
    line. We still guard against invalid values to avoid corrupting the EP
    surface.
    """

    start = len(frame)
    if "yardline_100" in frame.columns:
        frame = frame[frame["yardline_100"].between(0, 100)]

    removed = start - len(frame)
    return frame, removed


def clean_pbp_data(frame: pd.DataFrame) -> CleanResult:
    """
    Apply roadmap cleaning rules to a raw PBP DataFrame.

    Returns a CleanResult so the calling code can log or persist the cleaning
    summary alongside the cleaned dataset.
    """

    removed_rows: dict[str, int] = {}

    frame, removed_rows["non_plays"] = _remove_non_plays(frame)
    frame, removed_rows["special_teams"] = _remove_special_teams(frame)
    frame, removed_rows["low_clock"] = _remove_low_clock(frame)
    frame, removed_rows["missing_state"] = _enforce_state_columns(frame)
    frame, removed_rows["invalid_yardline"] = _validate_yardline(frame)

    # Keep the output indexed clean for downstream merges and modeling steps.
    frame = frame.reset_index(drop=True)

    return CleanResult(data=frame, removed_rows=removed_rows)


def load_and_clean_pbp(
    years: Iterable[int],
    data_dir: Path = DATA_ROOT,
    columns: list[str] | None = None,
) -> tuple[LoadResult, CleanResult]:
    """
    End-to-end helper: load raw PBP seasons and immediately clean them.

    This is the intended entry point for notebooks and future pipeline scripts.
    """

    raw = load_pbp_years(years, data_dir=data_dir, columns=columns)
    cleaned = clean_pbp_data(raw.data)
    return raw, cleaned


def _summarize_removals(removed: dict[str, int]) -> str:
    parts = [f"{key}={value}" for key, value in removed.items()]
    return ", ".join(parts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean NFL PBP data using the roadmap rules and optionally save it."
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
        help="Optional path to write cleaned data as parquet",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_ROOT,
        help="Directory containing play_by_play_YYYY.parquet files",
    )

    args = parser.parse_args()
    raw, cleaned = load_and_clean_pbp(args.years, data_dir=args.data_dir)

    print("Loaded rows:", len(raw.data))
    print("Cleaned rows:", len(cleaned.data))
    print("Removed:", _summarize_removals(cleaned.removed_rows))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cleaned.data.to_parquet(args.output, index=False)
        print("Wrote cleaned data to:", args.output)
