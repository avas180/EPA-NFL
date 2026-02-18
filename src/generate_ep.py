"""
Stage 7: Generate Expected Points (EP) values for each play state.

Grand-scheme purpose:
- This stage assigns EP to every play state using the trained model.
- Those EP values are the foundation for EPA in the next stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from xgboost import XGBRegressor

from clean_data import load_and_clean_pbp
from train_ep_model import FEATURE_COLUMNS, TARGET_COLUMN, prepare_training_frame


@dataclass(frozen=True)
class EPResult:
    """
    Container for EP-annotated data and summary stats.
    """

    data: pd.DataFrame
    rows: int
    ep_mean: float


def _train_model(train_years: Iterable[int]) -> XGBRegressor:
    """
    Train an XGBoost EP model on the provided seasons.

    For reproducibility and consistency, the feature set matches the
    roadmap-defined state variables.
    """

    train_data = prepare_training_frame(train_years)
    X = train_data[FEATURE_COLUMNS]
    y = train_data[TARGET_COLUMN]

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


def generate_ep(
    train_years: Iterable[int],
    predict_years: Iterable[int],
) -> EPResult:
    """
    Train an EP model and apply it to a set of seasons.

    Returns a DataFrame with an added `ep_before` column.
    """

    model = _train_model(train_years)
    _, cleaned = load_and_clean_pbp(predict_years)

    data = cleaned.data.copy()
    data["ep_before"] = model.predict(data[FEATURE_COLUMNS])

    return EPResult(
        data=data,
        rows=len(data),
        ep_mean=float(data["ep_before"].mean()),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate EP values for play states using a trained model."
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
        help="Seasons to assign EP values (e.g., 2025)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write EP-annotated data as parquet",
    )

    args = parser.parse_args()
    result = generate_ep(args.train_years, args.predict_years)

    print("Rows:", result.rows)
    print("Mean EP:", round(result.ep_mean, 4))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.data.to_parquet(args.output, index=False)
        print("Wrote EP data to:", args.output)
