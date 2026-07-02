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
# NBEATSx model that supports exogenous variables.
from neuralforecast.models import NBEATSx
# Metric used to score forecasts.
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
# Losses for training from neuralforecast
from neuralforecast.losses.pytorch import MAE

# Reuse the existing feature-building and chronological split helpers.
from preprocess_delhi_climate import build_time_series_features, split_train_test


# These are future-known calendar features passed into NBEATSx at prediction time.
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

HIST_EXOG_COLS = [
    "humidity",
    "wind_speed",
    "meanpressure",
    "rolling_mean_4",
    "rolling_std_4",
]

# Default configuration used when tuning is skipped.
DEFAULT_PARAMS = {
    "input_size":  7,#90,
    "stack_types": ["identity"],
    "n_blocks": [1],
    "mlp_units": [[512, 512]],
    "max_steps": 1000,
    "learning_rate": 5e-4,
    "batch_size": 32,
    "scaler_type": "standard",
}

# Small bounded search space so tuning stays optional and reasonably fast.
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
    # Reuse previously generated split files when available.
    if train_path.exists() and test_path.exists():
        train_df = pd.read_csv(train_path, parse_dates=["date"])
        test_df = pd.read_csv(test_path, parse_dates=["date"])
        return train_df, test_df

    # Otherwise, rebuild features from the raw source file.
    featured_df = build_time_series_features(input_path)
    # Split the full feature table into chronological train and test sets.
    train_df, test_df = split_train_test(featured_df)
    # Persist the generated split so later runs can load it directly.
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_df, test_df


def to_neuralforecast_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the dataframe to NeuralForecast format with future-known exogenous data."""
    # NeuralForecast expects a series id, timestamp column, and target named y.
    nf_df = pd.DataFrame(
        {
            "unique_id": "delhi_climate",
            "ds": pd.to_datetime(df["date"]),
            "y": df["target"].astype(float),
        }
    )
    # Attach the date-derived exogenous columns used by NBEATSx.
    for col in DATE_EXOG_COLS:
        nf_df[col] = df[col].to_numpy()

    for col in HIST_EXOG_COLS:
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
    # Build a small CLI so tuning remains optional.
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
    # Create a one-step-ahead NBEATSx model configured from the chosen parameters.
    return NBEATSx(
        # Forecast exactly one future step at a time.
        h=1,
        # Cap the lookback so it never exceeds half the available history.
        input_size=max(1, min(model_params["input_size"], len(train_nf) // 2)),
        # Tell the model which future exogenous columns will be supplied.
        futr_exog_list=DATE_EXOG_COLS,
        hist_exog_list=HIST_EXOG_COLS,
        # Architecture-related hyperparameters.
        stack_types=model_params["stack_types"],
        n_blocks=model_params["n_blocks"],
        mlp_units=model_params["mlp_units"],
        # Training-related hyperparameters.
        max_steps=model_params["max_steps"],
        learning_rate=model_params["learning_rate"],
        batch_size=model_params["batch_size"],
        scaler_type=model_params["scaler_type"],
        # Fix randomness for reproducible runs.
        random_seed=42,
        # Use a predictable model name in forecast outputs.
        alias="NBEATSx",
        loss=MAE(),
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )


def walk_forward_one_step_predict(
    nf: NeuralForecast, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """Forecast each test row using only actual observations available up to t-1.

    This is not recursive forecasting. For the prediction at time t, the input
    history is train_df plus the already observed test rows before t. Therefore,
    the model's most recent target value is the real lag-1 value, not a previous
    prediction. With input_size=7, NBEATSx receives the last 7 real target values
    in its lookback window whenever they are available.
    """
    predictions: list[float] = []

    for step, (_, row) in enumerate(test_df.iterrows()):
        # Use actual history only: full training set + observed test rows before t.
        observed_history_df = pd.concat(
            [train_df, test_df.iloc[:step]],
            ignore_index=True,
        )
        history_nf = to_neuralforecast_format(observed_history_df)

        # Make sure the current row is exactly one step after the history.
        # This guarantees that the last y in history_nf is the lag-1 target.
        assert_one_step_history(history_nf, row)

        # Build the h=1 future dataframe expected by NeuralForecast.
        futr_df = nf.make_future_dataframe(df=history_nf)

        # Fill future-known calendar features for the date being predicted.
        for col in DATE_EXOG_COLS:
            futr_df[col] = row[col]

        # Predict one step ahead. The model uses the y lookback from history_nf,
        # so lag-1, lag-2, ..., lag-7 are included through input_size=7.
        forecast_df = nf.predict(df=history_nf, futr_df=futr_df)
        predictions.append(float(forecast_df["NBEATSx"].iloc[0]))

    results_df = test_df[["date", "target"]].copy()
    results_df["prediction"] = predictions
    return results_df


def save_params_txt(params_path: Path, params: dict, mae: float | None = None) -> None:
    # Save the chosen parameters in a simple JSON payload.
    payload = {"params": params}
    # Include validation MAE when the parameters came from tuning.
    if mae is not None:
        payload["validation_mae"] = mae
    # Write the payload to a text file for later reuse.
    params_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_params_txt(params_path: Path) -> dict:
    # Read the saved parameter file.
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    # Return only the parameter dictionary.
    return payload["params"]


def split_train_validation(
    train_df: pd.DataFrame, validation_size: int = 120
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Keep the validation slice reasonably sized relative to the train length.
    validation_size = min(validation_size, max(30, len(train_df) // 5))
    # Use older rows for training during tuning.
    train_core_df = train_df.iloc[:-validation_size].copy()
    # Use the most recent rows as the validation window.
    validation_df = train_df.iloc[-validation_size:].copy()
    return train_core_df, validation_df


def evaluate_candidate(
    train_core_df: pd.DataFrame, validation_df: pd.DataFrame, model_params: dict
) -> float:
    # Convert the tuning-train split into NeuralForecast format.
    train_core_nf = to_neuralforecast_format(train_core_df)
    # Build one model from the candidate hyperparameters.
    model = build_model(train_core_nf, model_params)
    # Wrap the model in NeuralForecast for fitting and predicting.
    nf = NeuralForecast(models=[model], freq="D")
    # Fit using the reduced training portion only.
    nf.fit(df=train_core_nf, val_size=0)
    # Evaluate with the same non-recursive one-step procedure used at test time.
    validation_predictions = walk_forward_one_step_predict(
        nf, train_core_df, validation_df
    )
    # Lower MAE means the candidate is better.
    return mean_absolute_error(
        validation_predictions["target"], validation_predictions["prediction"]
    )


def tune_parameters(train_df: pd.DataFrame, tune_limit: int) -> tuple[dict, float]:
    # Hold out the newest part of the training data for tuning validation.
    train_core_df, validation_df = split_train_validation(train_df)
    # Start from defaults in case no candidate improves over them.
    best_params = DEFAULT_PARAMS.copy()
    # Initialize with infinity so the first real score wins.
    best_mae = float("inf")

    # Evaluate only the requested number of candidates.
    for candidate in TUNING_CANDIDATES[: max(1, tune_limit)]:
        candidate_mae = evaluate_candidate(train_core_df, validation_df, candidate)
        print(f"Tuning candidate MAE: {candidate_mae:.4f} | {candidate}")
        # Keep the candidate if it has the lowest MAE so far.
        if candidate_mae < best_mae:
            best_mae = candidate_mae
            best_params = candidate.copy()

    return best_params, best_mae

def clean_train_test_boundary(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

def main() -> None:
    # Read CLI options such as --tune and --use-saved-params.
    args = parse_args()
    # Define all input and output file locations.
    input_path = Path("DailyDelhiClimate.csv")
    train_path = Path("DailyDelhiClimate_train.csv")
    test_path = Path("DailyDelhiClimate_test.csv")
    predictions_path = Path("DailyDelhiClimate_nbeats_predictions.csv")
    params_path = Path("nbeatsx_best_params.txt")

    # Load or create the train/test split used by this script.
    train_df, test_df = ensure_train_test_files(input_path, train_path, test_path)
    train_df, test_df = clean_train_test_boundary(train_df, test_df)
    # Convert the training data into the format expected by NeuralForecast.
    train_nf = to_neuralforecast_format(train_df)

    # Start from default parameters unless the user requests otherwise.
    selected_params = DEFAULT_PARAMS.copy()
    # Optionally load previously tuned parameters from disk.
    if args.use_saved_params and params_path.exists():
        selected_params = load_params_txt(params_path)
        print("Loaded parameters from:", params_path)

    # Only run the expensive tuning loop when the user explicitly asks for it.
    if args.tune:
        selected_params, best_validation_mae = tune_parameters(train_df, args.tune_limit)
        save_params_txt(params_path, selected_params, best_validation_mae)
        print("Created:", params_path)
        print(f"Best validation MAE: {best_validation_mae:.4f}")

    # Build the final model with the selected parameter set.
    model = build_model(train_nf, selected_params)

    # Fit the final model on the full training data.
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=train_nf, val_size=0)
    # Generate non-recursive rolling one-step-ahead predictions for the full test set.
    # Each prediction uses actual observations up to t-1, so lag-1 is real.
    results_df = walk_forward_one_step_predict(nf, train_df, test_df)

    # Compute the main regression error metrics.
    mae = mean_absolute_error(results_df["target"], results_df["prediction"])
    rmse = np.sqrt(((results_df["target"] - results_df["prediction"]) ** 2).mean())
    mape = mean_absolute_percentage_error(results_df["target"], results_df["prediction"])
    # Save the test predictions to disk.
    results_df.to_csv(predictions_path, index=False)

    # Print a concise run summary.
    print("Created:", predictions_path)
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Selected params:", selected_params)
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Show the first few rows as a quick sanity check.
    print(results_df.head().to_string(index=False))


if __name__ == "__main__":
    # Run the training pipeline only when this file is executed directly.
    main()
