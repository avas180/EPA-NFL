"""
Stage 1: Data loading utilities for NFL play-by-play (PBP) parquet files.

Grand-scheme purpose:
- This module is the front door for all raw data access.
- Every later stage (cleaning, target construction, modeling) should call
  these helpers so we have a single, consistent way to read PBP data.
- Keeping loading logic centralized avoids subtle mismatches in data sources
  across notebooks and scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"


@dataclass(frozen=True)
class LoadResult:
    """
    Container for a loaded dataset plus lightweight metadata.

    This keeps the return value explicit so later stages can track which
    seasons were loaded and where they came from.
    """

    data: pd.DataFrame
    years: tuple[int, ...]
    source_dir: Path


def list_pbp_files(data_dir: Path = DATA_ROOT) -> list[Path]:
    """
    List all play-by-play parquet files available in the data directory.

    We keep this separate so higher-level loaders can validate inputs and
    report what seasons are present before doing any heavy I/O.
    """

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    return sorted(data_dir.glob("play_by_play_*.parquet"))


def _year_from_path(path: Path) -> int:
    """
    Extract the season year from a PBP parquet filename.

    Expected format: play_by_play_YYYY.parquet
    """

    stem = path.stem
    try:
        return int(stem.split("_")[-1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Unexpected PBP filename format: {path.name}") from exc


def available_years(data_dir: Path = DATA_ROOT) -> tuple[int, ...]:
    """
    Return all seasons available in the raw data directory.

    This is useful for validation and for building reproducible pipelines that
    declare exactly which seasons are used to train or analyze models.
    """

    years = [_year_from_path(path) for path in list_pbp_files(data_dir)]
    return tuple(sorted(years))


def load_pbp_years(
    years: Iterable[int],
    data_dir: Path = DATA_ROOT,
    columns: Sequence[str] | None = None,
) -> LoadResult:
    """
    Load one or more seasons of play-by-play data.

    Grand-scheme purpose:
    - This is the starting line for all downstream modeling work.
    - We load raw PBP data without feature engineering or cleaning so later
      stages can apply consistent transformations and keep the lineage clear.

    Parameters
    - years: seasons to load (e.g., [2018, 2019, 2020])
    - data_dir: directory containing play_by_play_YYYY.parquet files
    - columns: optional column subset to reduce memory usage

    Returns
    - LoadResult with the concatenated DataFrame and metadata
    """

    years = tuple(int(y) for y in years)
    if not years:
        raise ValueError("No seasons provided to load_pbp_years")

    files_by_year = {_year_from_path(p): p for p in list_pbp_files(data_dir)}
    missing = sorted(set(years) - set(files_by_year))
    if missing:
        raise FileNotFoundError(
            "Missing parquet files for seasons: " + ", ".join(map(str, missing))
        )

    frames = []
    for year in years:
        path = files_by_year[year]
        # Reading raw PBP data is intentionally done without transforms so
        # cleaning steps remain explicit and auditable in later stages.
        frame = pd.read_parquet(path, columns=list(columns) if columns else None)
        frame["season"] = year
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    return LoadResult(data=data, years=years, source_dir=data_dir)


def load_pbp_range(
    start_year: int,
    end_year: int,
    data_dir: Path = DATA_ROOT,
    columns: Sequence[str] | None = None,
) -> LoadResult:
    """
    Convenience loader for inclusive year ranges.

    This is the typical entry point for building a stable EP model using
    multiple seasons (e.g., 2018-2024) while keeping the call site concise.
    """

    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")

    years = range(start_year, end_year + 1)
    return load_pbp_years(years, data_dir=data_dir, columns=columns)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load NFL PBP parquet files and print a small summary."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="Seasons to load (e.g., 2018 2019 2020)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_ROOT,
        help="Directory containing play_by_play_YYYY.parquet files",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Optional subset of columns to load",
    )

    args = parser.parse_args()
    result = load_pbp_years(args.years, data_dir=args.data_dir, columns=args.columns)

    print("Loaded seasons:", result.years)
    print("Rows:", len(result.data))
    print("Columns:", len(result.data.columns))
    print("Sample columns:", list(result.data.columns[:10]))
