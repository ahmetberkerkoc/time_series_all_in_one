from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from preprocess_delhi_climate import build_time_series_features, split_train_test


def ensure_train_test_files(
    input_path: Path, train_path: Path, test_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train/test files, or create them if they do not exist."""
    if train_path.exists() and test_path.exists():
        train_df = pd.read_csv(train_path, parse_dates=["date"])
        test_df = pd.read_csv(test_path, parse_dates=["date"])
        return train_df, test_df

    featured_df = build_time_series_features(input_path)
    train_df, test_df = split_train_test(featured_df)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_df, test_df


def main() -> None:
    input_path = Path("DailyDelhiClimate.csv")
    train_path = Path("DailyDelhiClimate_train.csv")
    test_path = Path("DailyDelhiClimate_test.csv")
    predictions_path = Path("DailyDelhiClimate_linear_regression_predictions.csv")

    train_df, test_df = ensure_train_test_files(input_path, train_path, test_path)

    feature_cols = [
        col for col in train_df.columns if col not in {"date", "target", "meantemp"}
    ]
    x_train = train_df[feature_cols]
    y_train = train_df["target"]
    x_test = test_df[feature_cols]
    y_test = test_df["target"]

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    results_df = test_df[["date", "target"]].copy()
    results_df["prediction"] = predictions

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(((y_test - predictions) ** 2).mean())
    results_df.to_csv(predictions_path, index=False)

    print("Created:", predictions_path)
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(results_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
