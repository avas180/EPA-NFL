"""
Stage 9: Run sanity checks on EP and EPA outputs.

Grand-scheme purpose:
- These checks validate that the EP/EPA pipeline behaves like football.
- If these do not pass, the target construction or EPA logic likely needs
  revision before moving to visualization and analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from compute_epa import compute_epa


@dataclass(frozen=True)
class SanityReport:
    """
    Container for sanity check metrics.
    """

    ep_own_goal_mean: float
    ep_opp_goal_mean: float
    ep_3rd_long_mean: float
    ep_2nd_short_mean: float
    epa_interception_mean: float | None
    epa_4th_conv_mean: float | None


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.mean())


def run_sanity_checks(train_years: Iterable[int], predict_years: Iterable[int]) -> SanityReport:
    result = compute_epa(train_years, predict_years)
    data = result.data

    ep_own_goal_mean = _safe_mean(data.loc[data["yardline_100"] >= 90, "ep_before"])
    ep_opp_goal_mean = _safe_mean(data.loc[data["yardline_100"] <= 10, "ep_before"])

    ep_3rd_long_mean = _safe_mean(
        data.loc[(data["down"] == 3) & (data["ydstogo"] >= 10), "ep_before"]
    )
    ep_2nd_short_mean = _safe_mean(
        data.loc[(data["down"] == 2) & (data["ydstogo"] <= 5), "ep_before"]
    )

    epa_interception_mean = None
    if "interception" in data.columns:
        epa_interception_mean = _safe_mean(data.loc[data["interception"] == 1, "epa"])

    epa_4th_conv_mean = None
    if "first_down" in data.columns:
        epa_4th_conv_mean = _safe_mean(
            data.loc[(data["down"] == 4) & (data["first_down"] == 1), "epa"]
        )

    return SanityReport(
        ep_own_goal_mean=ep_own_goal_mean,
        ep_opp_goal_mean=ep_opp_goal_mean,
        ep_3rd_long_mean=ep_3rd_long_mean,
        ep_2nd_short_mean=ep_2nd_short_mean,
        epa_interception_mean=epa_interception_mean,
        epa_4th_conv_mean=epa_4th_conv_mean,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run EP/EPA sanity checks to validate pipeline behavior."
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
        help="Seasons to validate (e.g., 2025)",
    )

    args = parser.parse_args()
    report = run_sanity_checks(args.train_years, args.predict_years)

    print("EP own goal line (yardline_100>=90):", round(report.ep_own_goal_mean, 4))
    print("EP opponent goal line (yardline_100<=10):", round(report.ep_opp_goal_mean, 4))
    print("EP 3rd & long (ydstogo>=10):", round(report.ep_3rd_long_mean, 4))
    print("EP 2nd & short (ydstogo<=5):", round(report.ep_2nd_short_mean, 4))

    if report.epa_interception_mean is not None:
        print("EPA on interceptions:", round(report.epa_interception_mean, 4))

    if report.epa_4th_conv_mean is not None:
        print("EPA on 4th down conversions:", round(report.epa_4th_conv_mean, 4))
