from __future__ import annotations

import argparse
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss, MAE, MQLoss
from neuralforecast.models import DeepAR, Informer, NBEATSx, NHITS, PatchTST, iTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

def mean_absolute_scaled_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
) -> float:
    naive_errors = np.abs(np.diff(y_train))
    denominator = float(np.mean(naive_errors)) if len(naive_errors) else 0.0
    if np.isclose(denominator, 0.0):
        return float("inf")
    return float(np.mean(np.abs(y_true - y_pred)) / denominator)

RANDOM_SEED = 0
TOURISM_SAMPLE_SIZE = 100
SERIES_ID = "series"
M3_HORIZONS = {"monthly": 18, "quarterly": 8, "yearly": 6}
DATE_EXOG_CANDIDATES = [
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
    "hour",
    "day_of_month",
]
M3_SELECTED_FILES = {
    "monthly": [
        "M838.csv",
        "M1257.csv",
        "M1149.csv",
        "M537.csv",
        "M587.csv",
        "M788.csv",
        "M304.csv",
        "M1292.csv",
        "M1323.csv",
        "M385.csv",
    ],
    "quarterly": [
        "Q319.csv",
        "Q667.csv",
        "Q687.csv",
        "Q699.csv",
        "Q341.csv",
        "Q7.csv",
        "Q636.csv",
        "Q555.csv",
        "Q393.csv",
        "Q154.csv",
    ],
    "yearly": [
        "Y135.csv",
        "Y238.csv",
        "Y234.csv",
        "Y467.csv",
        "Y338.csv",
    ],
}
NBEATS_DEFAULT_PARAMS = {
    "input_size": 7,
    "stack_types": ["identity"],
    "n_blocks": [1],
    "mlp_units": [[128, 128]],
    "max_steps": 600,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "scaler_type": "standard",
}
NHITS_DEFAULT_PARAMS = {
    "input_size": 7,
    "n_blocks": [1, 1, 1],
    "mlp_units": [[64, 64], [64, 64], [64, 64]],
    "n_pool_kernel_size": [1, 1, 1],
    "n_freq_downsample": [1, 1, 1],
    "pooling_mode": "MaxPool1d",
    "interpolation_mode": "linear",
    "max_steps": 500,
    "learning_rate":1e-3,
    "batch_size": 1,
    "scaler_type": "standard",
}
PATCHTST_DEFAULT_PARAMS = {
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
    "max_steps": 1000,
    "learning_rate": 5e-4,
    "batch_size": 32,
    "windows_batch_size": 256,
    "scaler_type": "standard",
}
DEEPAR_DEFAULT_PARAMS = {
    "input_size": 7,
    "lstm_n_layers": 3,
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
ITRANSFORMER_DEFAULT_PARAMS = {
    "input_size": 7,
    "hidden_size": 128,
    "n_heads": 2,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 256,
    "factor": 1,
    "dropout": 0.1,
    "use_norm": True,
    "max_steps": 1000,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "windows_batch_size": 32,
    "scaler_type": "standard",
}
INFORMER_DEFAULT_PARAMS = {
    "input_size": 7,
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
}
MODEL_CHOICES = ("nbeats", "nhits", "patchtst", "deepar", "itransformer", "informer")
DATASET_CHOICES = (
    "m3_monthly",
    "m3_quarterly",
    "m3_yearly",
    "ett_h1",
    "ett_h2",
    "ett_m1",
    "ett_m2",
    "tourism",
)
MODEL_ALIASES = {
    "nbeats": "NBEATSx",
    "nhits": "NHITS",
    "patchtst": "PatchTST",
    "deepar": "DeepAR",
    "itransformer": "iTransformer",
    "informer": "Informer",
}


@dataclass(frozen=True)
class DatasetRun:
    dataset_name: str
    file_label: str
    series_name: str
    df: pd.DataFrame
    horizon: int
    freq: str
    include_mase: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate NBEATS, NHITS, and PatchTST on the extracted "
            "M3, ETT, and Tourism datasets."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=list(MODEL_CHOICES),
        help="Subset of deep models to run.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_CHOICES,
        default=list(DATASET_CHOICES),
        help="Subset of dataset groups to run.",
    )
    parser.add_argument(
        "--series-limit",
        type=int,
        default=None,
        help="Optional cap on the number of series processed per dataset group.",
    )
    parser.add_argument(
        "--tourism-sample-size",
        type=int,
        default=TOURISM_SAMPLE_SIZE,
        help="Number of Tourism CSVs to sample, matching tourism.py behavior.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Training steps used for all deep models.",
    )
    return parser.parse_args()


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in df.columns if col not in {"y", "date", "ds"}]
    if not feature_cols:
        return pd.DataFrame(index=df.index)
    return df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def choose_input_size(default_input_size: int, train_len: int) -> int:
    safe_upper = max(2, min(default_input_size, max(2, train_len // 2)))
    if safe_upper < 2:
        raise ValueError("Series is too short for deep-model training.")
    return safe_upper


def add_time_index(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    indexed_df = df.copy()
    if "date" in indexed_df.columns:
        indexed_df["ds"] = pd.to_datetime(indexed_df["date"])
    else:
        indexed_df["ds"] = pd.date_range(
            start="2000-01-01",
            periods=len(indexed_df),
            freq=freq,
        )
    return indexed_df


def to_neuralforecast_format(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    nf_df = pd.DataFrame(index=df.index)
    nf_df["unique_id"] = SERIES_ID
    nf_df["ds"] = pd.Series(
        pd.to_datetime(df["ds"]).array,
        index=df.index,
        dtype="datetime64[ns]",
    )
    nf_df["y"] = pd.to_numeric(df["y"], errors="coerce").astype(float).to_numpy()
    for col in feature_cols:
        nf_df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
    return nf_df


def build_model(
    model_name: str,
    input_size: int,
    feature_cols: list[str],
    max_steps: int,
) -> NBEATSx | NHITS | PatchTST | DeepAR | iTransformer | Informer:
    if model_name == "nbeats":
        params = NBEATS_DEFAULT_PARAMS.copy()
        params["max_steps"] = max_steps
        return NBEATSx(
            futr_exog_list=feature_cols,
            stack_types=params["stack_types"],
            n_blocks=params["n_blocks"],
            mlp_units=params["mlp_units"],
            alias=MODEL_ALIASES[model_name],
            h=1,
            input_size=input_size,
            max_steps=params["max_steps"],
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            scaler_type=params["scaler_type"],
            random_seed=42,
            loss=MAE(),
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
        )

    if model_name == "nhits":
        params = NHITS_DEFAULT_PARAMS.copy()
        params["max_steps"] = max_steps
        return NHITS(
            futr_exog_list=feature_cols,
            n_blocks=params["n_blocks"],
            mlp_units=params["mlp_units"],
            n_pool_kernel_size=params["n_pool_kernel_size"],
            n_freq_downsample=params["n_freq_downsample"],
            pooling_mode=params["pooling_mode"],
            interpolation_mode=params["interpolation_mode"],
            alias=MODEL_ALIASES[model_name],
            h=1,
            input_size=input_size,
            max_steps=params["max_steps"],
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            scaler_type=params["scaler_type"],
            random_seed=42,
            loss=MAE(),
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
        )

    if model_name == "deepar":
        params = DEEPAR_DEFAULT_PARAMS.copy()
        params["max_steps"] = max_steps
        return DeepAR(
            h=1,
            input_size=input_size,
            futr_exog_list=feature_cols,
            exclude_insample_y=False,
            lstm_n_layers=params["lstm_n_layers"],
            lstm_hidden_size=params["lstm_hidden_size"],
            lstm_dropout=params["lstm_dropout"],
            decoder_hidden_layers=params["decoder_hidden_layers"],
            decoder_hidden_size=params["decoder_hidden_size"],
            trajectory_samples=params["trajectory_samples"],
            max_steps=params["max_steps"],
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            windows_batch_size=params["windows_batch_size"],
            scaler_type=params["scaler_type"],
            random_seed=42,
            alias=MODEL_ALIASES[model_name],
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

    if model_name == "itransformer":
        params = ITRANSFORMER_DEFAULT_PARAMS.copy()
        params["max_steps"] = max_steps
        return iTransformer(
            h=1,
            input_size=input_size,
            n_series=1,
            exclude_insample_y=False,
            hidden_size=params["hidden_size"],
            n_heads=params["n_heads"],
            e_layers=params["e_layers"],
            d_layers=params["d_layers"],
            d_ff=params["d_ff"],
            factor=params["factor"],
            dropout=params["dropout"],
            use_norm=params["use_norm"],
            max_steps=params["max_steps"],
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            windows_batch_size=params["windows_batch_size"],
            scaler_type=params["scaler_type"],
            random_seed=42,
            alias=MODEL_ALIASES[model_name],
            loss=MAE(),
            valid_loss=MAE(),
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
        )

    if model_name == "informer":
        params = INFORMER_DEFAULT_PARAMS.copy()
        params["max_steps"] = max_steps
        return Informer(
            h=1,
            input_size=input_size,
            futr_exog_list=feature_cols,
            hidden_size=params["hidden_size"],
            conv_hidden_size=params["conv_hidden_size"],
            n_head=params["n_head"],
            encoder_layers=params["encoder_layers"],
            decoder_layers=params["decoder_layers"],
            decoder_input_size_multiplier=params["decoder_input_size_multiplier"],
            dropout=params["dropout"],
            factor=params["factor"],
            activation=params["activation"],
            distil=params["distil"],
            max_steps=params["max_steps"],
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            windows_batch_size=params["windows_batch_size"],
            scaler_type=params["scaler_type"],
            random_seed=42,
            alias=MODEL_ALIASES[model_name],
            loss=MAE(),
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
        )

    params = PATCHTST_DEFAULT_PARAMS.copy()
    params["max_steps"] = max_steps
    patch_len = min(params["patch_len"], input_size)
    stride = max(1, min(params["stride"], patch_len))
    return PatchTST(
        encoder_layers=params["encoder_layers"],
        n_heads=params["n_heads"],
        hidden_size=params["hidden_size"],
        linear_hidden_size=params["linear_hidden_size"],
        dropout=params["dropout"],
        fc_dropout=params["fc_dropout"],
        head_dropout=params["head_dropout"],
        attn_dropout=params["attn_dropout"],
        patch_len=patch_len,
        stride=stride,
        revin=params["revin"],
        revin_affine=params["revin_affine"],
        revin_subtract_last=params["revin_subtract_last"],
        activation=params["activation"],
        res_attention=params["res_attention"],
        batch_normalization=params["batch_normalization"],
        learn_pos_embed=params["learn_pos_embed"],
        windows_batch_size=params["windows_batch_size"],
        alias=MODEL_ALIASES[model_name],
        h=1,
        input_size=input_size,
        max_steps=params["max_steps"],
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        scaler_type=params["scaler_type"],
        random_seed=42,
        loss=MAE(),
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )


def infer_step_offset(df: pd.DataFrame) -> pd.Timedelta:
    ds = pd.to_datetime(df["ds"]).sort_values().reset_index(drop=True)
    diffs = ds.diff().dropna()
    if diffs.empty:
        return pd.Timedelta(days=1)
    return diffs.mode().iloc[0]


def assert_one_step_history(
    history_nf: pd.DataFrame,
    row: pd.Series,
    step_offset: pd.Timedelta,
) -> None:
    expected_next_ds = history_nf["ds"].max() + step_offset
    current_ds = pd.to_datetime(row["ds"])
    if current_ds != expected_next_ds:
        raise ValueError(
            f"Expected next date {expected_next_ds}, but got {current_ds}. "
            "This usually means the series is not ordered or there are missing timestamps."
        )


def walk_forward_one_step_predict(
    nf: NeuralForecast,
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    predictions: list[float] = []
    step_offset = infer_step_offset(pd.concat([train_df, test_df], ignore_index=True))

    for step in range(len(test_df)):
        history_df = pd.concat([train_df, test_df.iloc[:step]], ignore_index=True)
        history_nf = to_neuralforecast_format(history_df, feature_cols)
        row = test_df.iloc[step]
        assert_one_step_history(history_nf, row, step_offset)

        if model_name in {"nbeats", "nhits", "deepar", "informer"} and feature_cols:
            futr_df = nf.make_future_dataframe(df=history_nf)
            for col in feature_cols:
                futr_df[col] = row[col]
            forecast_df = nf.predict(
                df=history_nf,
                futr_df=futr_df,
            )
        else:
            forecast_df = nf.predict(df=history_nf)

        if model_name == "deepar":
            preferred_cols = ["DeepAR-median", "DeepAR", "DeepAR-mean"]
            pred_col = next((col for col in preferred_cols if col in forecast_df.columns), None)
            if pred_col is None:
                raise ValueError(
                    f"Could not find a DeepAR point forecast column. "
                    f"Columns: {forecast_df.columns.tolist()}"
                )
            predictions.append(float(forecast_df[pred_col].iloc[0]))
        else:
            predictions.append(float(forecast_df[MODEL_ALIASES[model_name]].iloc[0]))

    return np.asarray(predictions, dtype=float)


def compute_metrics(
    y_train: pd.Series,
    y_test: pd.Series,
    predictions: np.ndarray,
    include_mase: bool,
) -> dict[str, float]:
    metrics = {
        "mse": float(mean_squared_error(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mape": float(mean_absolute_percentage_error(y_test, predictions)),
    }
    if include_mase:
        metrics["mase"] = float(
            mean_absolute_scaled_error(
                y_true=y_test.to_numpy(),
                y_pred=predictions,
                y_train=y_train.to_numpy(),
            )
        )
    return metrics


def evaluate_series(
    run: DatasetRun,
    model_name: str,
    max_steps: int,
) -> dict[str, object]:
    indexed_df = add_time_index(run.df, run.freq)
    feature_df = prepare_feature_frame(indexed_df)
    feature_cols = list(feature_df.columns)
    indexed_df = indexed_df.copy()
    for col in feature_cols:
        indexed_df[col] = feature_df[col]

    y = pd.to_numeric(indexed_df["y"], errors="coerce")
    if y.isna().any():
        raise ValueError("Target contains NaNs.")

    if len(indexed_df) <= run.horizon:
        raise ValueError("Not enough rows for the requested forecast horizon.")

    train_df = indexed_df.iloc[:-run.horizon].reset_index(drop=True)
    test_df = indexed_df.iloc[-run.horizon :].reset_index(drop=True)
    y_train = pd.to_numeric(train_df["y"], errors="coerce")
    y_test = pd.to_numeric(test_df["y"], errors="coerce")

    default_input_size = {
        "nbeats": NBEATS_DEFAULT_PARAMS["input_size"],
        "nhits": NHITS_DEFAULT_PARAMS["input_size"],
        "patchtst": PATCHTST_DEFAULT_PARAMS["input_size"],
        "deepar": DEEPAR_DEFAULT_PARAMS["input_size"],
        "itransformer": ITRANSFORMER_DEFAULT_PARAMS["input_size"],
        "informer": INFORMER_DEFAULT_PARAMS["input_size"],
    }[model_name]
    input_size = choose_input_size(default_input_size, len(train_df))
    model_feature_cols = feature_cols
    if model_name in {"patchtst", "itransformer"}:
        model_feature_cols = []
    elif model_name in {"deepar", "informer"}:
        model_feature_cols = [col for col in feature_cols if col in DATE_EXOG_CANDIDATES]

    train_nf = to_neuralforecast_format(train_df, model_feature_cols)
    model = build_model(
        model_name=model_name,
        input_size=input_size,
        feature_cols=model_feature_cols,
        max_steps=max_steps,
    )
    nf = NeuralForecast(models=[model], freq=run.freq)
    nf.fit(df=train_nf, val_size=0)

    predictions = walk_forward_one_step_predict(
        nf=nf,
        model_name=model_name,
        train_df=train_df,
        test_df=test_df,
        feature_cols=model_feature_cols,
    )
    metrics = compute_metrics(
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
        include_mase=run.include_mase,
    )

    return {
        "metrics": metrics,
        "predictions": predictions.tolist(),
        "targets": y_test.to_numpy(dtype=float).tolist(),
        "train_length": int(len(train_df)),
        "test_length": int(len(test_df)),
        "input_size": int(input_size),
        "feature_count": int(len(feature_cols)),
    }


def load_m3_runs(series_limit: int | None) -> list[DatasetRun]:
    runs: list[DatasetRun] = []
    for m3_type, horizon in M3_HORIZONS.items():
        base_dir = Path(f"data/M3/Extracted/{m3_type}")
        csv_list = [base_dir / filename for filename in M3_SELECTED_FILES[m3_type]]
        missing_files = [path.name for path in csv_list if not path.exists()]
        if missing_files:
            raise FileNotFoundError(
                f"Missing selected M3 files for {m3_type}: {', '.join(missing_files)}"
            )
        if series_limit is not None:
            csv_list = csv_list[:series_limit]
        for csv_path in csv_list:
            runs.append(
                DatasetRun(
                    dataset_name=f"m3_{m3_type}",
                    file_label=f"m3_{m3_type}",
                    series_name=f"{m3_type}_{csv_path.name}",
                    df=pd.read_csv(csv_path),
                    horizon=horizon,
                    freq="D",
                    include_mase=False,
                )
            )
    return runs


def load_ett_runs() -> list[DatasetRun]:
    runs: list[DatasetRun] = []
    for ett_type in ["h1", "m1"]:
        csv_path = Path(f"data/ETT/Extracted/{ett_type}/ETT{ett_type}_extracted.csv")
        df = pd.read_csv(csv_path)
        horizon = 10
        runs.append(
            DatasetRun(
                dataset_name=f"ett_{ett_type}",
                file_label=f"ett_{ett_type}",
                series_name=ett_type,
                df=df,
                horizon=horizon,
                freq="h",
                include_mase=True,
            )
        )
    return runs


def load_tourism_runs(
    series_limit: int | None,
    tourism_sample_size: int,
) -> list[DatasetRun]:
    data_dir = Path("data/Tourism/Extracted")
    csv_list = [
        csv_path
        for csv_path in sorted(data_dir.glob("*.csv"))
        if len(pd.read_csv(csv_path)) > 1
    ]
    random.seed(RANDOM_SEED)
    random.shuffle(csv_list)
    csv_list = csv_list[:tourism_sample_size]
    if series_limit is not None:
        csv_list = csv_list[:series_limit]

    runs: list[DatasetRun] = []
    for csv_path in csv_list:
        df = pd.read_csv(csv_path)
        horizon = max(1, int(len(df) * 0.2))
        runs.append(
            DatasetRun(
                dataset_name="tourism",
                file_label="tourism",
                series_name=csv_path.name,
                df=df,
                horizon=horizon,
                freq="D",
                include_mase=True,
            )
        )
    return runs


def load_selected_runs(
    requested_datasets: list[str],
    series_limit: int | None,
    tourism_sample_size: int,
) -> dict[str, list[DatasetRun]]:
    grouped_runs: dict[str, list[DatasetRun]] = {}

    if any(name.startswith("m3_") for name in requested_datasets):
        for run in load_m3_runs(series_limit):
            if run.dataset_name in requested_datasets:
                grouped_runs.setdefault(run.dataset_name, []).append(run)

    if any(name.startswith("ett_") for name in requested_datasets):
        for run in load_ett_runs():
            if run.dataset_name in requested_datasets:
                grouped_runs.setdefault(run.dataset_name, []).append(run)

    if "tourism" in requested_datasets:
        grouped_runs["tourism"] = load_tourism_runs(series_limit, tourism_sample_size)

    return grouped_runs


def result_pickle_path(model_name: str, dataset_name: str) -> Path:
    return Path(f"ALL_RESULT_{model_name}_{dataset_name}.pkl")


def run_group(
    model_name: str,
    dataset_name: str,
    runs: list[DatasetRun],
    max_steps: int,
) -> None:
    payload: dict[str, object] = {
        "model": model_name,
        "dataset": dataset_name,
        "series_results": {},
        "skipped": {},
    }
    output_path = result_pickle_path(model_name, dataset_name)

    print("=" * 80)
    print(f"Model: {model_name} | Dataset: {dataset_name} | Series count: {len(runs)}")

    for counter, run in enumerate(runs, start=1):
        print(f"[{counter}/{len(runs)}] Processing {run.series_name}")
        try:
            result = evaluate_series(run=run, model_name=model_name, max_steps=max_steps)
            payload["series_results"][run.series_name] = result
            metrics = result["metrics"]
            print(
                f"Completed {run.series_name}: "
                f"mse={metrics['mse']:.6f}, "
                f"mae={metrics['mae']:.6f}, "
                f"mape={metrics['mape']:.6f}"
                + (
                    f", mase={metrics['mase']:.6f}"
                    if "mase" in metrics
                    else ""
                )
            )
        except Exception as exc:
            payload["skipped"][run.series_name] = str(exc)
            print(f"Skipped {run.series_name}: {exc}")

        pd.to_pickle(payload, output_path)

    print(f"Saved results to {output_path}")


def main() -> None:
    args = parse_args()

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    grouped_runs = load_selected_runs(
        requested_datasets=args.datasets,
        series_limit=args.series_limit,
        tourism_sample_size=args.tourism_sample_size,
    )

    for model_name in args.models:
        for dataset_name in args.datasets:
            runs = grouped_runs.get(dataset_name, [])
            if not runs:
                print(f"No runs found for {dataset_name}.")
                continue
            run_group(
                model_name=model_name,
                dataset_name=dataset_name,
                runs=runs,
                max_steps=args.max_steps,
            )


if __name__ == "__main__":
    main()
