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
# PatchTST model.
from neuralforecast.models import PatchTST
# Metric used to score forecasts.
from sklearn.metrics import mean_absolute_error
# Losses for training from neuralforecast.
from neuralforecast.losses.pytorch import MAE

# Reuse the existing feature-building and chronological split helpers.
from preprocess_delhi_climate import build_time_series_features, split_train_test


# PatchTST version in this script is target-only.
# We intentionally do not pass futr_exog_list, hist_exog_list, or futr_df.
# Therefore PatchTST uses only the autoregressive target window y[t-input_size:t].
# With input_size=7, this means the model receives lag-1 through lag-7.

# Default configuration used when tuning is skipped.
DEFAULT_PARAMS = {
    # Use the last 7 real target values: y[t-7], ..., y[t-1].
    # Therefore lag-1 is included in the lookback window.
    "input_size": 7,
    # PatchTST architecture. Keep patch_len <= input_size for short one-step windows.
    "encoder_layers": 2,
    "n_heads": 4,
    "hidden_size": 64,
    "linear_hidden_size": 128,
    "dropout": 0.10,
    "fc_dropout": 0.10,
    "head_dropout": 0.0,
    "attn_dropout": 0.0,
    "patch_len": 4,
    "stride": 1,
    "revin": True,
    "revin_affine": False,
    "revin_subtract_last": True,
    "activation": "gelu",
    "res_attention": True,
    "batch_normalization": False,
    "learn_pos_embed": True,
    # Training-related hyperparameters.
    "max_steps": 1000,
    "learning_rate": 5e-4,
    "batch_size": 32,
    "windows_batch_size": 256,
    "scaler_type": "standard",
}

# Small bounded search space so tuning stays optional and reasonably fast.
TUNING_CANDIDATES = [
    {
        "input_size": 7,
        "encoder_layers": 1,
        "n_heads": 2,
        "hidden_size": 32,
        "linear_hidden_size": 64,
        "dropout": 0.10,
        "fc_dropout": 0.10,
        "head_dropout": 0.0,
        "attn_dropout": 0.0,
        "patch_len": 3,
        "stride": 1,
        "revin": True,
        "revin_affine": False,
        "revin_subtract_last": True,
        "activation": "gelu",
        "res_attention": True,
        "batch_normalization": False,
        "learn_pos_embed": True,
        "max_steps": 300,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "windows_batch_size": 256,
        "scaler_type": "standard",
    },
    {
        "input_size": 7,
        "encoder_layers": 2,
        "n_heads": 4,
        "hidden_size": 64,
        "linear_hidden_size": 128,
        "dropout": 0.10,
        "fc_dropout": 0.10,
        "head_dropout": 0.0,
        "attn_dropout": 0.0,
        "patch_len": 4,
        "stride": 1,
        "revin": True,
        "revin_affine": False,
        "revin_subtract_last": True,
        "activation": "gelu",
        "res_attention": True,
        "batch_normalization": False,
        "learn_pos_embed": True,
        "max_steps": 500,
        "learning_rate": 5e-4,
        "batch_size": 32,
        "windows_batch_size": 256,
        "scaler_type": "standard",
    },
    {
        "input_size": 14,
        "encoder_layers": 2,
        "n_heads": 4,
        "hidden_size": 64,
        "linear_hidden_size": 128,
        "dropout": 0.15,
        "fc_dropout": 0.15,
        "head_dropout": 0.0,
        "attn_dropout": 0.0,
        "patch_len": 7,
        "stride": 2,
        "revin": True,
        "revin_affine": False,
        "revin_subtract_last": True,
        "activation": "gelu",
        "res_attention": True,
        "batch_normalization": False,
        "learn_pos_embed": True,
        "max_steps": 500,
        "learning_rate": 5e-4,
        "batch_size": 32,
        "windows_batch_size": 256,
        "scaler_type": "standard",
    },
    {
        "input_size": 30,
        "encoder_layers": 3,
        "n_heads": 4,
        "hidden_size": 128,
        "linear_hidden_size": 256,
        "dropout": 0.20,
        "fc_dropout": 0.20,
        "head_dropout": 0.0,
        "attn_dropout": 0.0,
        "patch_len": 8,
        "stride": 4,
        "revin": True,
        "revin_affine": False,
        "revin_subtract_last": True,
        "activation": "gelu",
        "res_attention": True,
        "batch_normalization": False,
        "learn_pos_embed": True,
        "max_steps": 700,
        "learning_rate": 3e-4,
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
    """Sort data and remove train/test boundary overlap."""
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
    """Convert the dataframe to target-only NeuralForecast format for PatchTST."""
    return pd.DataFrame(
        {
            "unique_id": "delhi_climate",
            "ds": pd.to_datetime(df["date"]),
            "y": df["target"].astype(float),
        }
    )


def assert_one_step_history(history_nf: pd.DataFrame, row: pd.Series) -> None:
    """Check that the current row is exactly one step after the available history."""
    expected_next_ds = history_nf["ds"].max() + pd.Timedelta(days=1)
    current_ds = pd.to_datetime(row["date"])
    if current_ds != expected_next_ds:
        raise ValueError(
            f"Expected next date {expected_next_ds.date()}, but got {current_ds.date()}. "
            "This usually means train/test dates overlap at the boundary or "
            "there are missing non-daily dates before forecasting."
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
        help="Load parameters from patchtst_best_params.txt if it exists.",
    )
    parser.add_argument(
        "--tune-limit",
        type=int,
        default=len(TUNING_CANDIDATES),
        help="Maximum number of candidate parameter sets to evaluate during tuning.",
    )
    return parser.parse_args()


def build_model(train_nf: pd.DataFrame, model_params: dict) -> PatchTST:
    """Create a one-step-ahead PatchTST model."""
    input_size = max(2, min(model_params["input_size"], len(train_nf) // 2))
    patch_len = min(model_params["patch_len"], input_size)
    stride = max(1, min(model_params["stride"], patch_len))

    return PatchTST(
        h=1,
        input_size=input_size,
        # PatchTST is intentionally target-only here.
        # Do not pass futr_exog_list, hist_exog_list, stat_exog_list, or futr_df.
        encoder_layers=model_params["encoder_layers"],
        n_heads=model_params["n_heads"],
        hidden_size=model_params["hidden_size"],
        linear_hidden_size=model_params["linear_hidden_size"],
        dropout=model_params["dropout"],
        fc_dropout=model_params["fc_dropout"],
        head_dropout=model_params["head_dropout"],
        attn_dropout=model_params["attn_dropout"],
        patch_len=patch_len,
        stride=stride,
        revin=model_params["revin"],
        revin_affine=model_params["revin_affine"],
        revin_subtract_last=model_params["revin_subtract_last"],
        activation=model_params["activation"],
        res_attention=model_params["res_attention"],
        batch_normalization=model_params["batch_normalization"],
        learn_pos_embed=model_params["learn_pos_embed"],
        max_steps=model_params["max_steps"],
        learning_rate=model_params["learning_rate"],
        batch_size=model_params["batch_size"],
        windows_batch_size=model_params["windows_batch_size"],
        scaler_type=model_params["scaler_type"],
        random_seed=42,
        alias="PatchTST",
        loss=MAE(),
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )


def walk_forward_one_step_predict(
    nf: NeuralForecast, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """Forecast each test row using actual observations available up to t-1.

    This is not recursive forecasting. For the prediction at time t, the input
    history is train_df plus the already observed test rows before t. Therefore,
    the model's most recent target value is the real lag-1 value, not a previous
    prediction. With input_size=7, PatchTST receives the last 7 real target values
    in its lookback window whenever they are available.

    PatchTST is called without futr_df and without exogenous lists.
    Therefore the prediction uses only the target lookback y[t-input_size:t].
    """
    predictions: list[float] = []

    for step, (_, row) in enumerate(test_df.iterrows()):
        observed_history_df = pd.concat(
            [train_df, test_df.iloc[:step]],
            ignore_index=True,
        )
        history_nf = to_neuralforecast_format(observed_history_df)

        # This guarantees that the last y in history_nf is the real lag-1 target.
        assert_one_step_history(history_nf, row)

        # No futr_df is used. PatchTST receives only the target history from df.
        forecast_df = nf.predict(df=history_nf)
        predictions.append(float(forecast_df["PatchTST"].iloc[0]))

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
    predictions_path = Path("DailyDelhiClimate_patchtst_predictions.csv")
    params_path = Path("patchtst_best_params.txt")

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

    # Generate non-recursive rolling one-step-ahead predictions for the full test set.
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
