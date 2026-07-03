# Time Series Project

This repository is now configured as a `uv` project for the forecasting, boosting, and notebook workflows in the root directory.

## Setup

```bash
uv sync
```

## Common commands

Preprocess the dataset:

```bash
uv run python preprocess_delhi_climate.py
```

Train the baseline linear regression model:

```bash
uv run python train_linear_regression.py
```

Train the LightGBM regressor:

```bash
uv run python train_lightgbm.py
```

Train one of the NeuralForecast models:

```bash
uv run python train_nbeats.py
uv run python train_nhits_gpt.py
uv run python train_patchtst_gpt.py
```

## Dependency notes

- Dependencies were collected from both the runnable `.py` scripts and `model_comparision.ipynb`.
- The libraries are installed as normal project dependencies, so you can use them from either notebooks or `.py` files.
- `SDT` is referenced by the notebook but is not present in this repository, so it was not added as a package dependency.
