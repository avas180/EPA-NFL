"""
Stage 5: Train a baseline Expected Points (EP) model.

Grand-scheme purpose:
- Before introducing more complex models, we establish a transparent baseline
  that verifies the pipeline and target construction are working.
- A simple linear regression should still recover intuitive EP patterns and
  acts as a sanity check for later modeling stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

from target_construction import load_clean_possessions_and_targets


FEATURE_COLUMNS = [
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
    "score_differential",
    "goal_to_go",
]

TARGET_COLUMN = "future_points"


@dataclass(frozen=True)
class BaselineResult:
    """
    Container for baseline model outputs and evaluation metrics.
    """

    model: LinearRegression
    rmse: float
    mae: float
    rows: int


def _prepare_training_frame(years: Iterable[int]) -> pd.DataFrame:
    """
    Load data, construct targets, and return a modeling-ready DataFrame.

    This function enforces that only pre-play state features are used for
    training, which keeps the EP model aligned with the roadmap.
    """

    _, targets = load_clean_possessions_and_targets(years)
    data = targets.data

    # Keep only features and target to avoid accidental leakage.
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise KeyError("Missing required columns for training: " + ", ".join(missing))

    return data[required].copy()


def train_baseline_model(
    years: Iterable[int],
    test_size: float = 0.2,
    random_state: int = 42,
) -> BaselineResult:
    """
    Train and evaluate a linear regression baseline EP model.
    """

    data = _prepare_training_frame(years)
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    return BaselineResult(model=model, rmse=rmse, mae=mae, rows=len(data))


def _save_model(model: LinearRegression, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a baseline linear regression EP model."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="Seasons to use for training (e.g., 2018 2019 2020)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for testing",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Optional path to save the trained model",
    )

    args = parser.parse_args()
    result = train_baseline_model(
        args.years, test_size=args.test_size, random_state=args.random_state
    )

    print("Rows:", result.rows)
    print("RMSE:", round(result.rmse, 4))
    print("MAE:", round(result.mae, 4))

    if args.model_out:
        _save_model(result.model, args.model_out)
        print("Saved model to:", args.model_out)
