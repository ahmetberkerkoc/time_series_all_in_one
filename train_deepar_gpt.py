from __future__ import annotations

# Parse optional command-line flags such as --tune.
import argparse
# Save and reload tuned parameters as plain text JSON.
import json
# Configure a writable matplotlib cache directory for this environment.
import os
# Work with file paths in an OS-independent way.
from pathlib import Path

# Avoid matplotlib permission issues in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Numerical utilities for RMSE computation.
import numpy as np
# Tabular data loading and manipulation.
import pandas as pd
# Forecasting wrapper used to fit and predict with neural models.
from neuralforecast import NeuralForecast
# DeepAR probabilistic autoregressive RNN model.
from neuralforecast.models import DeepAR
# Metric used to score forecasts.
from sklearn.metrics import mean_absolute_error
# DeepAR requires a probabilistic distribution loss.
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss

# Reuse the existing feature-building and chronological split helpers.
from preprocess_delhi_climate import build_time_series_features, split_train_test


# Future-known calendar features passed into DeepAR at prediction time.
# These are safe because they are known before the forecast date.
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

# DeepAR in NeuralForecast is probabilistic and uses Monte Carlo inference.
# According to the NeuralForecast documentation, historic exogenous variables
# are not available for DeepAR during this inference procedure. Therefore,
# humidity, wind_speed, pressure, and rolling y-based features are not passed
# through hist_exog_list here. The autoregressive target history is still used
# through input_size, so lag-1 is included.
FUTR_EXOG_COLS = DATE_EXOG_COLS

# Default configuration used when tuning is skipped.
DEFAULT_PARAMS = {
    # Use the last 7 real target values: y[t-7], ..., y[t-1].
    # Therefore lag-1 is included in the DeepAR autoregressive input.
    "input_size": 7,
    "lstm_n_layers": 2,
    "lstm_hidden_size": 128,
    "lstm_dropout": 0.1,
    "decoder_hidden_layers": 0,
    "decoder_hidden_size": 0,
    "trajectory_samples": 100,
    "max_steps": 1000,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "windows_batch_size": 256,
    "scaler_type": "standard",
}

# Small bounded search space so tuning stays optional and reasonably fast.
TUNING_CANDIDATES = [
    {
        "input_size": 7,
        "lstm_n_layers": 1,
        "lstm_hidden_size": 64,
        "lstm_dropout": 0.1,
        "decoder_hidden_layers": 0,
        "decoder_hidden_size": 0,
        "trajectory_samples": 100,
        "max_steps": 300,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "windows_batch_size": 256,
        "scaler_type": "standard",
    },
    {
        "input_size": 7,
        "lstm_n_layers": 2,
        "lstm_hidden_size": 128,
        "lstm_dropout": 0.1,
        "decoder_hidden_layers": 0,
        "decoder_hidden_size": 0,
        "trajectory_samples": 100,
        "max_steps": 500,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "windows_batch_size": 256,
        "scaler_type": "standard",
    },
    {
        "input_size": 14,
        "lstm_n_layers": 2,
        "lstm_hidden_size": 128,
        "lstm_dropout": 0.1,
        "decoder_hidden_layers": 1,
        "decoder_hidden_size": 64,
        "trajectory_samples": 100,
        "max_steps": 500,
        "learning_rate": 5e-4,
        "batch_size": 32,
        "windows_batch_size": 256,
        "scaler_type": "standard",
    },
    {
        "input_size": 30,
        "lstm_n_layers": 2,
        "lstm_hidden_size": 256,
        "lstm_dropout": 0.1,
        "decoder_hidden_layers": 1,
        "decoder_hidden_size": 128,
        "trajectory_samples": 100,
        "max_steps": 700,
        "learning_rate": 5e-4,
        "batch_size": 16,
        "windows_batch_size": 256,
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


def clean_train_test_boundary(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort data, remove duplicate dates, and prevent train/test leakage."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["date"] = pd.to_datetime(train_df["date"])
    test_df["date"] = pd.to_datetime(test_df["date"])

    train_df = (
        train_df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    test_df = (
        test_df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    first_test_date = test_df["date"].min()
    leaked_train_rows = train_df["date"] >= first_test_date

    if leaked_train_rows.any():
        removed = int(leaked_train_rows.sum())
        print(
            f"Removed {removed} train rows on/after first test date "
            f"({first_test_date.date()}) to avoid one-step-ahead leakage."
        )
        train_df = train_df.loc[~leaked_train_rows].reset_index(drop=True)

    if train_df.empty:
        raise ValueError(
            "Training data became empty after removing train/test overlap. "
            "Check your split_train_test function or delete cached split CSV files."
        )

    return train_df, test_df


def to_neuralforecast_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the dataframe to NeuralForecast format for DeepAR."""
    nf_df = pd.DataFrame(
        {
            "unique_id": "delhi_climate",
            "ds": pd.to_datetime(df["date"]),
            "y": df["target"].astype(float),
        }
    )

    # Attach only future-known exogenous variables.
    for col in FUTR_EXOG_COLS:
        nf_df[col] = df[col].to_numpy()

    return nf_df


def assert_one_step_history(history_nf: pd.DataFrame, row: pd.Series) -> None:
    """Check that the current row is exactly one step after the available history."""
    expected_next_ds = history_nf["ds"].max() + pd.Timedelta(days=1)
    current_ds = pd.to_datetime(row["date"])
    if current_ds != expected_next_ds:
        raise ValueError(
            f"Expected next date {expected_next_ds.date()}, but got {current_ds.date()}. "
            "Check for missing or non-daily dates before forecasting."
        )


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
        help="Load parameters from deepar_best_params.txt if it exists.",
    )
    parser.add_argument(
        "--tune-limit",
        type=int,
        default=len(TUNING_CANDIDATES),
        help="Maximum number of candidate parameter sets to evaluate during tuning.",
    )
    return parser.parse_args()


def build_model(train_nf: pd.DataFrame, model_params: dict) -> DeepAR:
    """Create a one-step-ahead DeepAR model configured from chosen parameters."""
    return DeepAR(
        h=1,
        input_size=max(1, min(model_params["input_size"], len(train_nf) // 2)),
        futr_exog_list=FUTR_EXOG_COLS,
        exclude_insample_y=False,  # keep autoregressive y history; lag-1 is used
        lstm_n_layers=model_params["lstm_n_layers"],
        lstm_hidden_size=model_params["lstm_hidden_size"],
        lstm_dropout=model_params["lstm_dropout"],
        decoder_hidden_layers=model_params["decoder_hidden_layers"],
        decoder_hidden_size=model_params["decoder_hidden_size"],
        trajectory_samples=model_params["trajectory_samples"],
        max_steps=model_params["max_steps"],
        learning_rate=model_params["learning_rate"],
        batch_size=model_params["batch_size"],
        windows_batch_size=model_params["windows_batch_size"],
        scaler_type=model_params["scaler_type"],
        random_seed=42,
        alias="DeepAR",
        loss=DistributionLoss(
            distribution="StudentT",
            level=[80, 90],
            return_params=False,
        ),
        valid_loss=MQLoss(level=[80, 90]),
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )


def extract_point_forecast(forecast_df: pd.DataFrame) -> float:
    """Extract a scalar point forecast from DeepAR's probabilistic output."""
    preferred_cols = [
        "DeepAR-median",
        "DeepAR",  # fallback for versions that return the point forecast under alias
        "DeepAR-mean",
    ]
    for col in preferred_cols:
        if col in forecast_df.columns:
            return float(forecast_df[col].iloc[0])

    deepar_cols = [col for col in forecast_df.columns if col.startswith("DeepAR")]
    if not deepar_cols:
        raise ValueError(
            f"Could not find a DeepAR prediction column. Columns: {forecast_df.columns.tolist()}"
        )

    # Avoid interval columns when possible.
    non_interval_cols = [
        col for col in deepar_cols
        if "-lo-" not in col and "-hi-" not in col and "-param" not in col
    ]
    if non_interval_cols:
        return float(forecast_df[non_interval_cols[0]].iloc[0])

    raise ValueError(
        "DeepAR returned only interval/parameter columns. "
        f"Columns: {forecast_df.columns.tolist()}"
    )


def walk_forward_one_step_predict(
    nf: NeuralForecast, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """Forecast each test row using only actual observations available up to t-1.

    This is not recursive forecasting. For the prediction at time t, the input
    history is train_df plus the already observed test rows before t. Therefore,
    the model's most recent target value is the real lag-1 value, not a previous
    prediction. With input_size=7, DeepAR receives the last 7 real target values
    in its autoregressive window whenever they are available.
    """
    predictions: list[float] = []

    for step, (_, row) in enumerate(test_df.iterrows()):
        observed_history_df = pd.concat(
            [train_df, test_df.iloc[:step]],
            ignore_index=True,
        )
        history_nf = to_neuralforecast_format(observed_history_df)

        # Guarantees that the last y in history_nf is the true lag-1 target.
        assert_one_step_history(history_nf, row)

        # Build the h=1 future dataframe expected by NeuralForecast.
        futr_df = nf.make_future_dataframe(df=history_nf)

        # Fill future-known calendar features for the date being predicted.
        for col in FUTR_EXOG_COLS:
            futr_df[col] = row[col]

        forecast_df = nf.predict(df=history_nf, futr_df=futr_df)
        predictions.append(extract_point_forecast(forecast_df))

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
    validation_predictions = walk_forward_one_step_predict(
        nf, train_core_df, validation_df
    )
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
    predictions_path = Path("DailyDelhiClimate_deepar_predictions.csv")
    params_path = Path("deepar_best_params.txt")

    train_df, test_df = ensure_train_test_files(input_path, train_path, test_path)
    train_df, test_df = clean_train_test_boundary(train_df, test_df)
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

    # Generate non-recursive rolling one-step-ahead predictions.
    # Each prediction uses actual observations up to t-1, so lag-1 is real.
    results_df = walk_forward_one_step_predict(nf, train_df, test_df)

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
