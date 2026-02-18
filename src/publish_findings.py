"""
Stage 12: Publish findings for the selected season(s).

Grand-scheme purpose:
- Produce a concise, shareable summary of key EPA findings.
- Bundle the generated tables and heatmaps into an easy-to-read report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from team_analysis import analyze_teams
from visualization import plot_field_value_heatmaps


@dataclass(frozen=True)
class PublishResult:
    """
    Paths to the generated report assets.
    """

    report_path: Path
    table_dir: Path
    figure_dir: Path


def _top_n(path: Path, n: int = 5) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df.head(n)


def _to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(No data available)"
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def publish_findings(
    train_years: Iterable[int],
    predict_years: Iterable[int],
    output_dir: Path = Path("outputs"),
) -> PublishResult:
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"

    analysis = analyze_teams(train_years, predict_years, output_dir=table_dir)
    plot_field_value_heatmaps(train_years, predict_years, output_dir=figure_dir)

    report_path = output_dir / "summary.md"

    offense_top = _top_n(analysis.offense_path) if analysis.offense_path else pd.DataFrame()
    defense_top = _top_n(analysis.defense_path) if analysis.defense_path else pd.DataFrame()
    qb_top = _top_n(analysis.qb_path) if analysis.qb_path else pd.DataFrame()

    report_lines: list[str] = []
    report_lines.append("# EPA Findings Summary")
    report_lines.append("")
    report_lines.append("Generated outputs:")
    report_lines.append(f"- Tables: {table_dir}")
    report_lines.append(f"- Figures: {figure_dir}")
    report_lines.append("")

    def _append_table(title: str, df: pd.DataFrame) -> None:
        report_lines.append(f"## {title}")
        report_lines.append("")
        report_lines.append(_to_markdown_table(df))
        report_lines.append("")

    _append_table("Top Offenses by EPA/Play", offense_top)
    _append_table("Top Defenses by EPA Allowed", defense_top)
    _append_table("Top QBs by EPA/Play (min 100 attempts)", qb_top)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines))

    return PublishResult(report_path=report_path, table_dir=table_dir, figure_dir=figure_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a markdown summary of EPA findings."
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
        default=Path("outputs"),
        help="Directory to write report assets",
    )

    args = parser.parse_args()
    result = publish_findings(args.train_years, args.predict_years, args.output_dir)

    print("Wrote report:", result.report_path)
