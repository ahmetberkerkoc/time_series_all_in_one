import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error as mae,
    mean_absolute_percentage_error as mape,
    mean_squared_error as mse,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error as mase
from statsmodels.tsa.statespace.sarimax import SARIMAX

import m3


SAMPLE_SIZE = 100
RANDOM_SEED = 0
DATA_DIR = "data/Tourism/Extracted"
RESULT_DIR = "tourism_results"
RESULT_PICKLE = "ALL_RESULT_tourism.pkl"


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in df.columns if col != "y"]
    return df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def score_predictions(y_true, y_pred, y_train):
    return (
        mse(y_true=y_true, y_pred=y_pred),
        mae(y_true=y_true, y_pred=y_pred),
        mape(y_true=y_true, y_pred=y_pred),
        mase(y_true=y_true, y_pred=y_pred, y_train=y_train),
    )


if __name__ == "__main__":
    all_result_dict = {}
    os.makedirs(RESULT_DIR, exist_ok=True)

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    csv_list = [
        csv_name
        for csv_name in os.listdir(DATA_DIR)
        if csv_name.endswith(".csv")
        and len(pd.read_csv(os.path.join(DATA_DIR, csv_name))) > 1
    ]
    random.shuffle(csv_list)
    csv_list = csv_list[:SAMPLE_SIZE]

    print(f"Selected {len(csv_list)} tourism series from {DATA_DIR}")

    for counter, csv_name in enumerate(csv_list, start=1):
        datapath = os.path.join(DATA_DIR, csv_name)
        print("=" * 80)
        print(f"Processing {counter}/{len(csv_list)}: {csv_name}")

        try:
            score_dict = {}
            df = pd.read_csv(datapath)
            y = pd.to_numeric(df.loc[:, "y"], errors="coerce")
            X = prepare_feature_frame(df)

            if y.isna().any():
                print(f"Skipping {csv_name}: target contains NaNs.")
                continue

            test_len = max(1, int(len(y) * 0.2))
            train_len = len(y) - test_len

            if train_len < 1:
                print(f"Skipping {csv_name}: not enough training rows after 20% split.")
                continue

            y_train = y.iloc[:-test_len]
            y_test = y.iloc[-test_len:]

            m3.UPD_NOISE_STD = 0.1
            m3.MEAS_NOISE_STD = 1

            order = (2, 0, 0)
            X_scaled = pd.DataFrame(
                MinMaxScaler().fit_transform(X),
                index=X.index,
                columns=X.columns,
            )

            print("Running particle filter model: our")
            pf = m3.filter_sgbdtsx(y, sx_order=order, exog=X_scaled, n_particles=len(y))
            print("Running SARIMAX baseline")
            stats_sx = SARIMAX(y, X_scaled, order=order).fit(disp=False)

            our_preds = pd.Series(pf.predictions, index=y.index)
            stats_preds = pd.Series(stats_sx.predict(), index=y.index)

            if our_preds.iloc[-test_len:].isna().any():
                print(f"Skipping {csv_name}: particle filter produced NaNs.")
                continue

            ax = y_test.plot(label="true")
            our_preds.iloc[-test_len:].plot(figsize=(14, 5), ls="--", label="pred, ours", ax=ax)
            stats_preds.iloc[-test_len:].plot(ls="--", label="pred, statsmodels", ax=ax)

            score_dict["our"] = score_predictions(
                y_true=y_test,
                y_pred=our_preds.iloc[-test_len:],
                y_train=y_train,
            )
            print(
                f"Completed our: mse={score_dict['our'][0]:.6f}, "
                f"mae={score_dict['our'][1]:.6f}, "
                f"mape={score_dict['our'][2]:.6f}, "
                f"mase={score_dict['our'][3]:.6f}"
            )

            score_dict["sarimax"] = score_predictions(
                y_true=y_test,
                y_pred=stats_preds.iloc[-test_len:],
                y_train=y_train,
            )
            print(
                f"Completed sarimax: mse={score_dict['sarimax'][0]:.6f}, "
                f"mae={score_dict['sarimax'][1]:.6f}, "
                f"mape={score_dict['sarimax'][2]:.6f}, "
                f"mase={score_dict['sarimax'][3]:.6f}"
            )

            X_train, X_test = X.iloc[:-test_len], X.iloc[-test_len:]

            print("Training model: Hard_Tree")
            regressor = DecisionTreeRegressor(random_state=RANDOM_SEED)
            regressor.fit(X_train, y_train)
            model_pred = regressor.predict(X_test)
            pd.Series(model_pred, index=range(train_len, train_len + test_len)).plot(
                ls="--", label="pred, HDT", ax=ax
            )
            score_dict["Hard_Tree"] = score_predictions(y_true=y_test, y_pred=model_pred, y_train=y_train)
            print(
                f"Completed Hard_Tree: mse={score_dict['Hard_Tree'][0]:.6f}, "
                f"mae={score_dict['Hard_Tree'][1]:.6f}, "
                f"mape={score_dict['Hard_Tree'][2]:.6f}, "
                f"mase={score_dict['Hard_Tree'][3]:.6f}"
            )

            print("Training model: LGBM")
            regressor = m3.build_lgbm_regressor(
                train_size=len(X_train),
                feature_count=X_train.shape[1],
            )
            regressor.fit(X_train, y_train)
            model_pred = regressor.predict(X_test)
            pd.Series(model_pred, index=range(train_len, train_len + test_len)).plot(
                ls="--", label="pred, LGBM", ax=ax
            )
            score_dict["LGBM"] = score_predictions(y_true=y_test, y_pred=model_pred, y_train=y_train)
            print(
                f"Completed LGBM: mse={score_dict['LGBM'][0]:.6f}, "
                f"mae={score_dict['LGBM'][1]:.6f}, "
                f"mape={score_dict['LGBM'][2]:.6f}, "
                f"mase={score_dict['LGBM'][3]:.6f}"
            )

            print("Training model: XGB")
            regressor = xgb.XGBRegressor(random_state=RANDOM_SEED)
            regressor.fit(X_train, y_train)
            model_pred = regressor.predict(X_test)
            pd.Series(model_pred, index=range(train_len, train_len + test_len)).plot(
                ls="--", label="pred, XGB", ax=ax
            )
            score_dict["XGB"] = score_predictions(y_true=y_test, y_pred=model_pred, y_train=y_train)
            print(
                f"Completed XGB: mse={score_dict['XGB'][0]:.6f}, "
                f"mae={score_dict['XGB'][1]:.6f}, "
                f"mape={score_dict['XGB'][2]:.6f}, "
                f"mase={score_dict['XGB'][3]:.6f}"
            )

            X_train_scaled, X_test_scaled = X_scaled.iloc[:-test_len], X_scaled.iloc[-test_len:]

            print("Training model: MLP")
            regressor = MLPRegressor(random_state=RANDOM_SEED)
            regressor.fit(X_train_scaled, y_train)
            model_pred = regressor.predict(X_test_scaled)
            pd.Series(model_pred, index=range(train_len, train_len + test_len)).plot(
                ls="--", label="pred, MLP", ax=ax
            )
            score_dict["MLP"] = score_predictions(y_true=y_test, y_pred=model_pred, y_train=y_train)
            print(
                f"Completed MLP: mse={score_dict['MLP'][0]:.6f}, "
                f"mae={score_dict['MLP'][1]:.6f}, "
                f"mape={score_dict['MLP'][2]:.6f}, "
                f"mase={score_dict['MLP'][3]:.6f}"
            )

            print("Training model: Linear")
            regressor = LinearRegression()
            regressor.fit(X_train_scaled, y_train)
            model_pred = regressor.predict(X_test_scaled)
            pd.Series(model_pred, index=range(train_len, train_len + test_len)).plot(
                ls="--", label="pred, Linear", ax=ax
            )
            score_dict["Linear"] = score_predictions(y_true=y_test, y_pred=model_pred, y_train=y_train)
            print(
                f"Completed Linear: mse={score_dict['Linear'][0]:.6f}, "
                f"mae={score_dict['Linear'][1]:.6f}, "
                f"mape={score_dict['Linear'][2]:.6f}, "
                f"mase={score_dict['Linear'][3]:.6f}"
            )

            print("Running baseline: Naive")
            model_pred = y.iloc[-test_len - 1 : -1].to_numpy()
            pd.Series(model_pred, index=range(train_len, train_len + test_len)).plot(
                ls="--", label="pred, Naive", ax=ax
            )
            score_dict["Naive"] = score_predictions(y_true=y_test, y_pred=model_pred, y_train=y_train)
            print(
                f"Completed Naive: mse={score_dict['Naive'][0]:.6f}, "
                f"mae={score_dict['Naive'][1]:.6f}, "
                f"mape={score_dict['Naive'][2]:.6f}, "
                f"mase={score_dict['Naive'][3]:.6f}"
            )

           

            all_result_dict[csv_name] = score_dict
            pd.to_pickle(all_result_dict, RESULT_PICKLE)
            print(f"Saved metrics: {RESULT_PICKLE}")
        except Exception as exc:
            print(f"Something went wrong in {csv_name}")
            print(exc)
