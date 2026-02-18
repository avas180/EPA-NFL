"""
Stage 11: Analyze 2025 teams using EPA outputs.

Grand-scheme purpose:
- Summarize which offenses and defenses performed best by EPA/play.
- Provide a first pass at QB-level EPA for 2025.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from compute_epa import compute_epa


@dataclass(frozen=True)
class TeamAnalysisResult:
    """
    Paths to the generated summary tables.
    """

    offense_path: Path | None
    defense_path: Path | None
    qb_path: Path | None


def analyze_teams(
    train_years: Iterable[int],
    predict_years: Iterable[int],
    output_dir: Path = Path("outputs/tables"),
) -> TeamAnalysisResult:
    """
    Compute team and QB EPA summaries and write them to CSV.
    """

    result = compute_epa(train_years, predict_years)
    data = result.data

    output_dir.mkdir(parents=True, exist_ok=True)

    offense_path = None
    defense_path = None
    qb_path = None

    if "posteam" in data.columns:
        offense = (
            data.groupby("posteam")["epa"]
            .agg(epa_per_play="mean", plays="count")
            .sort_values("epa_per_play", ascending=False)
        )
        offense_path = output_dir / "offense_epa_per_play.csv"
        offense.to_csv(offense_path)

    if "defteam" in data.columns:
        defense = (
            data.groupby("defteam")["epa"]
            .agg(epa_allowed="mean", plays="count")
            .sort_values("epa_allowed")
        )
        # Defense perspective: lower offensive EPA allowed is better.
        defense_path = output_dir / "defense_epa_allowed.csv"
        defense.to_csv(defense_path)

    if "passer_player_name" in data.columns and "pass_attempt" in data.columns:
        qb = data[data["pass_attempt"] == 1]
        qb = (
            qb.groupby("passer_player_name")["epa"]
            .agg(epa_per_play="mean", plays="count")
            .query("plays >= 100")
            .sort_values("epa_per_play", ascending=False)
        )
        qb_path = output_dir / "qb_epa_per_play.csv"
        qb.to_csv(qb_path)

    return TeamAnalysisResult(
        offense_path=offense_path,
        defense_path=defense_path,
        qb_path=qb_path,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute 2025 EPA summaries by team and quarterback."
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
        help="Seasons to analyze (e.g., 2025)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tables"),
        help="Directory to write summary CSVs",
    )

    args = parser.parse_args()
    result = analyze_teams(args.train_years, args.predict_years, args.output_dir)

    if result.offense_path:
        print("Wrote offense summary:", result.offense_path)
    if result.defense_path:
        print("Wrote defense summary:", result.defense_path)
    if result.qb_path:
        print("Wrote QB summary:", result.qb_path)
