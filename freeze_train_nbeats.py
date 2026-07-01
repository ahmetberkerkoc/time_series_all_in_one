from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from sklearn.metrics import mean_absolute_error

from preprocess_delhi_climate import build_time_series_features, split_train_test


DATE_EXOG_COLS = [
    "year",
    "quarter",
    "month",
    "week_of_year",
    "day",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
]

DEFAULT_PARAMS = {
    "input_size": 90,
    "stack_types": ["identity"],
    "n_blocks": [1],
    "mlp_units": [[512, 512]],
    "max_steps": 500,
    "learning_rate": 5e-4,
    "batch_size": 32,
    "scaler_type": "standard",
}

TUNING_CANDIDATES = [
    {
        "input_size": 60,
        "stack_types": ["identity"],
        "n_blocks": [1],
        "mlp_units": [[256, 256]],
        "max_steps": 300,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "scaler_type": "standard",
    },
    {
        "input_size": 90,
        "stack_types": ["identity"],
        "n_blocks": [1],
        "mlp_units": [[256, 256]],
        "max_steps": 500,
        "learning_rate": 5e-4,
        "batch_size": 32,
        "scaler_type": "standard",
    },
    {
        "input_size": 90,
        "stack_types": ["identity"],
        "n_blocks": [2],
        "mlp_units": [[512, 512]],
        "max_steps": 500,
        "learning_rate": 5e-4,
        "batch_size": 16,
        "scaler_type": "standard",
    },
    {
        "input_size": 120,
        "stack_types": ["identity"],
        "n_blocks": [2],
        "mlp_units": [[512, 512]],
        "max_steps": 700,
        "learning_rate": 3e-4,
        "batch_size": 16,
        "scaler_type": "standard",
    },
]


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


def to_neuralforecast_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the dataframe to NeuralForecast format with future-known exogenous data."""
    nf_df = pd.DataFrame(
        {
            "unique_id": "delhi_climate",
            "ds": pd.to_datetime(df["date"]),
            "y": df["target"].astype(float),
        }
    )
    for col in DATE_EXOG_COLS:
        nf_df[col] = df[col].to_numpy()
    return nf_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run a bounded parameter search before fitting the final model.",
    )
    parser.add_argument(
        "--use-saved-params",
        action="store_true",
        help="Load parameters from nbeatsx_best_params.txt if it exists.",
    )
    parser.add_argument(
        "--tune-limit",
        type=int,
        default=len(TUNING_CANDIDATES),
        help="Maximum number of candidate parameter sets to evaluate during tuning.",
    )
    return parser.parse_args()


def build_model(train_nf: pd.DataFrame, model_params: dict) -> NBEATSx:
    return NBEATSx(
        h=1,
        input_size=min(model_params["input_size"], len(train_nf) // 2),
        futr_exog_list=DATE_EXOG_COLS,
        stack_types=model_params["stack_types"],
        n_blocks=model_params["n_blocks"],
        mlp_units=model_params["mlp_units"],
        max_steps=model_params["max_steps"],
        learning_rate=model_params["learning_rate"],
        batch_size=model_params["batch_size"],
        scaler_type=model_params["scaler_type"],
        random_seed=42,
        alias="NBEATSx",
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )


def recursive_predict(
    nf: NeuralForecast, history_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """Forecast one step ahead recursively across the full test set."""
    history_nf = history_df.copy()
    predictions: list[float] = []

    for _, row in test_df.iterrows():
        futr_df = nf.make_future_dataframe(df=history_nf)
        for col in DATE_EXOG_COLS:
            futr_df[col] = row[col]
        forecast_df = nf.predict(df=history_nf, futr_df=futr_df)
        prediction = float(forecast_df["NBEATSx"].iloc[0])
        predictions.append(prediction)

        next_history_row = futr_df.copy()
        next_history_row["y"] = prediction
        history_nf = pd.concat([history_nf, next_history_row], ignore_index=True)

    results_df = test_df[["date", "target"]].copy()
    results_df["prediction"] = predictions
    return results_df


def save_params_txt(params_path: Path, params: dict, mae: float | None = None) -> None:
    payload = {"params": params}
    if mae is not None:
        payload["validation_mae"] = mae
    params_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_params_txt(params_path: Path) -> dict:
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    return payload["params"]


def split_train_validation(
    train_df: pd.DataFrame, validation_size: int = 120
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_size = min(validation_size, max(30, len(train_df) // 5))
    train_core_df = train_df.iloc[:-validation_size].copy()
    validation_df = train_df.iloc[-validation_size:].copy()
    return train_core_df, validation_df


def evaluate_candidate(
    train_core_df: pd.DataFrame, validation_df: pd.DataFrame, model_params: dict
) -> float:
    train_core_nf = to_neuralforecast_format(train_core_df)
    model = build_model(train_core_nf, model_params)
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=train_core_nf, val_size=0)
    validation_predictions = recursive_predict(nf, train_core_nf, validation_df)
    return mean_absolute_error(
        validation_predictions["target"], validation_predictions["prediction"]
    )


def tune_parameters(train_df: pd.DataFrame, tune_limit: int) -> tuple[dict, float]:
    train_core_df, validation_df = split_train_validation(train_df)
    best_params = DEFAULT_PARAMS.copy()
    best_mae = float("inf")

    for candidate in TUNING_CANDIDATES[: max(1, tune_limit)]:
        candidate_mae = evaluate_candidate(train_core_df, validation_df, candidate)
        print(f"Tuning candidate MAE: {candidate_mae:.4f} | {candidate}")
        if candidate_mae < best_mae:
            best_mae = candidate_mae
            best_params = candidate.copy()

    return best_params, best_mae


def main() -> None:
    args = parse_args()
    input_path = Path("DailyDelhiClimate.csv")
    train_path = Path("DailyDelhiClimate_train.csv")
    test_path = Path("DailyDelhiClimate_test.csv")
    predictions_path = Path("DailyDelhiClimate_nbeats_predictions.csv")
    params_path = Path("nbeatsx_best_params.txt")

    train_df, test_df = ensure_train_test_files(input_path, train_path, test_path)
    train_nf = to_neuralforecast_format(train_df)

    selected_params = DEFAULT_PARAMS.copy()
    if args.use_saved_params and params_path.exists():
        selected_params = load_params_txt(params_path)
        print("Loaded parameters from:", params_path)

    if args.tune:
        selected_params, best_validation_mae = tune_parameters(train_df, args.tune_limit)
        save_params_txt(params_path, selected_params, best_validation_mae)
        print("Created:", params_path)
        print(f"Best validation MAE: {best_validation_mae:.4f}")

    model = build_model(train_nf, selected_params)

    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=train_nf, val_size=0)
    results_df = recursive_predict(nf, train_nf, test_df)

    mae = mean_absolute_error(results_df["target"], results_df["prediction"])
    rmse = np.sqrt(((results_df["target"] - results_df["prediction"]) ** 2).mean())
    results_df.to_csv(predictions_path, index=False)

    print("Created:", predictions_path)
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Selected params:", selected_params)
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    print(results_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
