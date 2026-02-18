"""
Stage 6: Train the main EP model using XGBoost regression.

Grand-scheme purpose:
- This model is the workhorse for EP predictions.
- It should outperform the linear baseline while still using only
  pre-play state variables to avoid leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

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
class XGBoostResult:
    """
    Container for model outputs and evaluation metrics.
    """

    model: XGBRegressor
    rmse: float
    mae: float
    rows: int


def prepare_training_frame(years: Iterable[int]) -> pd.DataFrame:
    """
    Load data, construct targets, and return a modeling-ready DataFrame.

    This function enforces that only pre-play state features are used for
    training, which keeps the EP model aligned with the roadmap.
    """

    _, targets = load_clean_possessions_and_targets(years)
    data = targets.data

    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise KeyError("Missing required columns for training: " + ", ".join(missing))

    return data[required].copy()


def train_xgboost_model(
    years: Iterable[int],
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.08,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
) -> XGBoostResult:
    """
    Train and evaluate an XGBoost EP regression model.
    """

    data = prepare_training_frame(years)
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    return XGBoostResult(model=model, rmse=rmse, mae=mae, rows=len(data))


def _save_model(model: XGBRegressor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the main XGBoost EP model."
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
        "--n-estimators",
        type=int,
        default=300,
        help="Number of boosting rounds",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Tree depth",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.08,
        help="Boosting learning rate",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Row subsample ratio",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="Column subsample ratio",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Optional path to save the trained model",
    )

    args = parser.parse_args()
    result = train_xgboost_model(
        args.years,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
    )

    print("Rows:", result.rows)
    print("RMSE:", round(result.rmse, 4))
    print("MAE:", round(result.mae, 4))

    if args.model_out:
        _save_model(result.model, args.model_out)
        print("Saved model to:", args.model_out)
