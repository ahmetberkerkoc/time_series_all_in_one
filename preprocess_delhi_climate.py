from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_time_series_features(csv_path: str | Path) -> pd.DataFrame:
    """Load the Delhi climate dataset and add calendar-based features."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["target"] = df["meantemp"]
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    for lag in range(1, 8):
        df[f"lag_{lag}"] = df["target"].shift(lag)

    # Shift by one first so the rolling window only uses past observations.
    df["rolling_mean_4"] = df["target"].shift(1).rolling(window=4).mean()
    df["rolling_std_4"] = df["target"].shift(1).rolling(window=4).std()

    df = df.dropna()    
    return df


def split_train_test(
    df: pd.DataFrame, test_size: float = 0.3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a chronological train/test split for forecasting."""
    cleaned_df = df.dropna().reset_index(drop=True)
    split_index = int(len(cleaned_df) * (1 - test_size))
    train_df = cleaned_df.iloc[:split_index].copy()
    test_df = cleaned_df.iloc[split_index:].copy()
    return train_df, test_df


if __name__ == "__main__":
    input_path = Path("DailyDelhiClimate.csv")
    features_output_path = Path("DailyDelhiClimate_features.csv")
    train_output_path = Path("DailyDelhiClimate_train.csv")
    test_output_path = Path("DailyDelhiClimate_test.csv")

    featured_df = build_time_series_features(input_path)
    train_df, test_df = split_train_test(featured_df)

    featured_df.to_csv(features_output_path, index=False)
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print("Created:", features_output_path)
    print("Created:", train_output_path)
    print("Created:", test_output_path)
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print(train_df.head().to_string(index=False))
