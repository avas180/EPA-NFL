"""
Stage 10: Visualize Expected Points field value maps.

Grand-scheme purpose:
- These heatmaps show how EP varies by field position and down/distance.
- They provide a quick visual sanity check and a foundation for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import os

_mpl_dir = Path("outputs/.mplconfig").resolve()
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from generate_ep import generate_ep


@dataclass(frozen=True)
class HeatmapResult:
    """
    Container for output paths produced by the heatmap generator.
    """

    paths: list[Path]


def _bin_fields(data: pd.DataFrame) -> pd.DataFrame:
    """
    Bin yardline and yards-to-go for stable heatmap aggregation.
    """

    df = data.copy()
    df["yardline_bin"] = (df["yardline_100"] // 5) * 5
    df["ydstogo_bin"] = df["ydstogo"].clip(0, 20)
    return df


def plot_field_value_heatmaps(
    train_years: Iterable[int],
    predict_years: Iterable[int],
    output_dir: Path = Path("outputs/figures"),
) -> HeatmapResult:
    """
    Generate EP heatmaps by down and write them to disk.
    """

    result = generate_ep(train_years, predict_years)
    data = _bin_fields(result.data)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for down in [1, 2, 3, 4]:
        subset = data[data["down"] == down]
        pivot = (
            subset.pivot_table(
                index="ydstogo_bin",
                columns="yardline_bin",
                values="ep_before",
                aggfunc="mean",
            )
            .sort_index(ascending=False)
        )

        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, cmap="viridis", cbar_kws={"label": "Expected Points"})
        plt.title(f"Expected Points by Field Position (Down {down})")
        plt.xlabel("Yards from Opponent End Zone (binned)")
        plt.ylabel("Yards to Go (binned)")

        path = output_dir / f"ep_heatmap_down_{down}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        paths.append(path)

    return HeatmapResult(paths=paths)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate EP field value heatmaps by down."
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
        help="Seasons to plot (e.g., 2025)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory to write heatmap images",
    )

    args = parser.parse_args()
    result = plot_field_value_heatmaps(args.train_years, args.predict_years, args.output_dir)

    print("Wrote heatmaps:")
    for path in result.paths:
        print("-", path)
