"""
Stage 3: Build possession tracking for cleaned NFL play-by-play data.

Grand-scheme purpose:
- Expected Points targets depend on what happens *until the offense
  loses possession*.
- This module identifies continuous offensive possessions so later
  stages can sum points on a drive and compute future_points accurately.
- By centralizing possession logic here, we keep the downstream target
  construction consistent and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from clean_data import CleanResult, load_and_clean_pbp


REQUIRED_COLUMNS = ["game_id", "play_id", "posteam", "drive"]


@dataclass(frozen=True)
class PossessionResult:
    """
    Container for possession-annotated data and a summary count.

    This mirrors the pattern in cleaning so we can inspect the number of
    possessions created before we move on to target construction.
    """

    data: pd.DataFrame
    total_possessions: int


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise KeyError("Missing required columns for possession tracking: " + ", ".join(missing))


def add_possession_columns(frame: pd.DataFrame) -> PossessionResult:
    """
    Add possession identifiers to a cleaned play-by-play DataFrame.

    Logic:
    - Sort by game and play sequence.
    - Start a new possession when posteam or drive changes.
    - Assign a 1-based possession_index within each game.

    Returns a PossessionResult with the updated DataFrame.
    """

    _ensure_columns(frame, REQUIRED_COLUMNS)

    # Work on a copy so callers keep their original DataFrame unchanged.
    data = frame.copy()

    # Sort by game and play to ensure comparisons are sequential and stable.
    data = data.sort_values(["game_id", "play_id"]).reset_index(drop=True)

    prev_posteam = data.groupby("game_id")["posteam"].shift(1)
    prev_drive = data.groupby("game_id")["drive"].shift(1)

    possession_change = (data["posteam"] != prev_posteam) | (data["drive"] != prev_drive)
    possession_change = possession_change.fillna(True)

    data["possession_index"] = possession_change.groupby(data["game_id"]).cumsum()
    data["possession_id"] = (
        data["game_id"].astype(str) + "-" + data["possession_index"].astype(int).astype(str)
    )

    # A drive_id is a convenient, explicit alias for (game_id, drive).
    data["drive_id"] = data["game_id"].astype(str) + "-" + data["drive"].astype(int).astype(str)

    total_possessions = int(data["possession_id"].nunique())
    return PossessionResult(data=data, total_possessions=total_possessions)


def load_clean_and_track_possessions(years: Iterable[int]) -> tuple[CleanResult, PossessionResult]:
    """
    End-to-end helper: load raw seasons, clean them, then add possession columns.

    This is the intended entry point for notebooks and future pipeline scripts.
    """

    _, cleaned = load_and_clean_pbp(years)
    possessions = add_possession_columns(cleaned.data)
    return cleaned, possessions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add possession tracking columns to cleaned PBP data."
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
        help="Optional path to write possession-annotated data as parquet",
    )

    args = parser.parse_args()
    cleaned, possessions = load_clean_and_track_possessions(args.years)

    print("Cleaned rows:", len(cleaned.data))
    print("Possessions:", possessions.total_possessions)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        possessions.data.to_parquet(args.output, index=False)
        print("Wrote possession data to:", args.output)
