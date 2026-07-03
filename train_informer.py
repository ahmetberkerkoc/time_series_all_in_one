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
# Informer transformer model.
from neuralforecast.models import Informer
# Metric used to score forecasts.
from sklearn.metrics import mean_absolute_error
# Loss for point forecasting.
from neuralforecast.losses.pytorch import MAE

# Reuse the existing feature-building and chronological split helpers.
from preprocess_delhi_climate import build_time_series_features, split_train_test


# Future-known calendar features supplied in futr_df at prediction time.
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

# Informer in NeuralForecast does not support hist_exog_list.
# Therefore this version uses only the autoregressive target history through
# input_size and future-known calendar variables through futr_exog_list.

# Default configuration used when tuning is skipped.
DEFAULT_PARAMS = {
    # With h=1 and input_size=7, the model sees y[t-7], ..., y[t-1].
    # Therefore the real lag-1 target is included in every one-step forecast.
    "input_size": 7,
    # Informer architecture hyperparameters.
    "hidden_size": 64,
    "conv_hidden_size": 32,
    "n_head": 2,
    "encoder_layers": 2,
    "decoder_layers": 1,
    "decoder_input_size_multiplier": 0.5,
    "dropout": 0.05,
    "factor": 3,
    "activation": "gelu",
    "distil": True,
    # Training-related hyperparameters.
    "max_steps": 500,
    "learning_rate": 5e-4,
    "batch_size": 32,
    "windows_batch_size": 128,
    "scaler_type": "standard",
}

# Small bounded search space so tuning stays optional and reasonably fast.
TUNING_CANDIDATES = [
    {
        "input_size": 7,
        "hidden_size": 32,
        "conv_hidden_size": 32,
        "n_head": 2,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "decoder_input_size_multiplier": 0.5,
        "dropout": 0.05,
        "factor": 3,
        "activation": "gelu",
        "distil": True,
        "max_steps": 300,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "windows_batch_size": 128,
        "scaler_type": "standard",
    },
    {
        "input_size": 14,
        "hidden_size": 64,
        "conv_hidden_size": 32,
        "n_head": 2,
        "encoder_layers": 2,
        "decoder_layers": 1,
        "decoder_input_size_multiplier": 0.5,
        "dropout": 0.05,
        "factor": 3,
        "activation": "gelu",
        "distil": True,
        "max_steps": 500,
        "learning_rate": 5e-4,
        "batch_size": 32,
        "windows_batch_size": 128,
        "scaler_type": "standard",
    },
    {
        "input_size": 30,
        "hidden_size": 64,
        "conv_hidden_size": 64,
        "n_head": 4,
        "encoder_layers": 2,
        "decoder_layers": 1,
        "decoder_input_size_multiplier": 0.5,
        "dropout": 0.10,
        "factor": 3,
        "activation": "gelu",
        "distil": True,
        "max_steps": 700,
        "learning_rate": 3e-4,
        "batch_size": 16,
        "windows_batch_size": 128,
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
    """Sort data, drop duplicate dates, and remove train/test overlap.

    For a valid one-step-ahead test, the first test date must be strictly after
    the final training date. If cached CSV files overlap, this function removes
    the overlapping rows from the training set to avoid leakage.
    """
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
            "Check split_train_test or delete cached split CSV files."
        )

    return train_df, test_df


def to_neuralforecast_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a dataframe to NeuralForecast format for Informer.

    Informer does not support historical exogenous variables in NeuralForecast,
    so the dataframe contains only unique_id, ds, y, and future-known calendar
    features. Lagged target values are still used through the model input_size.
    """
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


def assert_one_step_history(history_nf: pd.DataFrame, row: pd.Series) -> None:
    """Check that the current row is exactly one step after the history."""
    expected_next_ds = history_nf["ds"].max() + pd.Timedelta(days=1)
    current_ds = pd.to_datetime(row["date"])
    if current_ds != expected_next_ds:
        raise ValueError(
            f"Expected next date {expected_next_ds.date()}, but got {current_ds.date()}. "
            "Check for missing dates, duplicate dates, or train/test overlap."
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
        help="Load parameters from informer_best_params.txt if it exists.",
    )
    parser.add_argument(
        "--tune-limit",
        type=int,
        default=len(TUNING_CANDIDATES),
        help="Maximum number of candidate parameter sets to evaluate during tuning.",
    )
    return parser.parse_args()


def build_model(train_nf: pd.DataFrame, model_params: dict) -> Informer:
    """Create a one-step-ahead Informer model."""
    return Informer(
        h=1,
        input_size=max(1, min(model_params["input_size"], len(train_nf) // 2)),
        futr_exog_list=DATE_EXOG_COLS,
        hidden_size=model_params["hidden_size"],
        conv_hidden_size=model_params["conv_hidden_size"],
        n_head=model_params["n_head"],
        encoder_layers=model_params["encoder_layers"],
        decoder_layers=model_params["decoder_layers"],
        decoder_input_size_multiplier=model_params["decoder_input_size_multiplier"],
        dropout=model_params["dropout"],
        factor=model_params["factor"],
        activation=model_params["activation"],
        distil=model_params["distil"],
        max_steps=model_params["max_steps"],
        learning_rate=model_params["learning_rate"],
        batch_size=model_params["batch_size"],
        windows_batch_size=model_params["windows_batch_size"],
        scaler_type=model_params["scaler_type"],
        random_seed=42,
        alias="Informer",
        loss=MAE(),
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )


def walk_forward_one_step_predict(
    nf: NeuralForecast, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """Forecast each test row using actual observations available up to t-1.

    This is not recursive forecasting. For test time t, history is train_df plus
    previously observed test rows. Therefore lag-1 is the real target value.
    Informer receives lagged y values through input_size, not through manually
    constructed historical exogenous columns.
    """
    predictions: list[float] = []

    for step, (_, row) in enumerate(test_df.iterrows()):
        observed_history_df = pd.concat(
            [train_df, test_df.iloc[:step]],
            ignore_index=True,
        )
        history_nf = to_neuralforecast_format(observed_history_df)
        assert_one_step_history(history_nf, row)

        futr_df = nf.make_future_dataframe(df=history_nf)
        for col in DATE_EXOG_COLS:
            futr_df[col] = row[col]

        forecast_df = nf.predict(df=history_nf, futr_df=futr_df)
        predictions.append(float(forecast_df["Informer"].iloc[0]))

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
    predictions_path = Path("DailyDelhiClimate_informer_predictions.csv")
    params_path = Path("informer_best_params.txt")

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
