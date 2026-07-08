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
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error as mase
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

import m3




def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in df.columns if col != "y"]
    X = df.loc[:, feature_cols].copy()

    if "date" in X.columns:
        X = X.drop(columns=["date"])

    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0)


if __name__ == "__main__":
    all_result_dict = {}
    os.makedirs("ett_results", exist_ok=True)

    np.random.seed(0)
    random.seed(0)

    for ett_type in ["h1", "h2", "m1", "m2"]:
        try:
            datapath = f"data/ETT/Extracted/{ett_type}/ETT{ett_type}_extracted.csv"
            print("=" * 80)
            print("True Training")
            print(f"Processing dataset: {ett_type}")
            print(f"Reading data from: {datapath}")

            df = pd.read_csv(datapath)
            y = df.loc[:, "y"]
            X = prepare_feature_frame(df)
            
            test_len = int(len(y) * 0.2)
            forecast_horizon = test_len
            train_len = len(y) - test_len
            score_dict = {}
            print(
                f"Loaded {ett_type} with {len(df)} rows, {X.shape[1]} features, "
                f"train_len={train_len}, test_len={test_len}"
            )

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

            our_preds = pd.Series(pf.predictions)
            stats_preds = stats_sx.predict()

            if our_preds.iloc[-test_len:].isna().any():
                print(f"Skipping {ett_type}: particle filter produced NaNs.")
                continue

            ax = y.iloc[-test_len:].plot(label="true")
            our_preds.iloc[-test_len:].plot(figsize=(14, 5), ls="--", label="pred, ours", ax=ax)
            stats_preds.iloc[-test_len:].plot(ls="--", label="pred, statsmodels", ax=ax)

            test_mse = mse(y_pred=our_preds.iloc[-test_len:], y_true=y.iloc[-test_len:])
            test_mae = mae(y_pred=our_preds.iloc[-test_len:], y_true=y.iloc[-test_len:])
            test_mape = mape(y_pred=our_preds.iloc[-test_len:], y_true=y.iloc[-test_len:])
            test_mase = mase( y_true=y.iloc[-test_len:], y_pred=our_preds.iloc[-test_len:], y_train=y.iloc[:-test_len])
            
            score_dict["our"] = (test_mse, test_mae, test_mape, test_mase)
            print(f"Completed our: mse={test_mse:.6f}, mae={test_mae:.6f}, mape={test_mape:.6f}")

            sarimax_test_mse = mse(y_pred=stats_preds.iloc[-test_len:], y_true=y.iloc[-test_len:])
            sarimax_test_mae = mae(y_pred=stats_preds.iloc[-test_len:], y_true=y.iloc[-test_len:])
            sarimax_test_mape = mape(y_pred=stats_preds.iloc[-test_len:], y_true=y.iloc[-test_len:])
            sarimax_test_mase = mase( y_true=y.iloc[-test_len:], y_pred=stats_preds.iloc[-test_len:], y_train=y.iloc[:-test_len])
            
            score_dict["sarimax"] = (sarimax_test_mse, sarimax_test_mae, sarimax_test_mape, sarimax_test_mase)
            print(
                f"Completed sarimax: mse={sarimax_test_mse:.6f}, "
                f"mae={sarimax_test_mae:.6f}, mape={sarimax_test_mape:.6f}"
            )

            X_train, X_test = X.iloc[:-forecast_horizon], X.iloc[-forecast_horizon:]
            y_train, y_test = y.iloc[:-forecast_horizon], y.iloc[-forecast_horizon:]

            print("Training model: Hard_Tree")
            regressor = DecisionTreeRegressor(random_state=0)
            regressor.fit(X_train, y_train)
            model_pred = regressor.predict(X_test)
            pd.Series(model_pred, index=range(train_len, train_len + forecast_horizon)).plot(
                ls="--", label="pred, HDT", ax=ax
            )
            score_dict["Hard_Tree"] = (
                mse(y_pred=model_pred, y_true=y_test),
                mae(y_pred=model_pred, y_true=y_test),
                mape(y_pred=model_pred, y_true=y_test),
                mase(y_true=y_test, y_pred=model_pred, y_train=y_train)
            )
            print(
                "Completed Hard_Tree: "
                f"mse={score_dict['Hard_Tree'][0]:.6f}, "
                f"mae={score_dict['Hard_Tree'][1]:.6f}, "
                f"mape={score_dict['Hard_Tree'][2]:.6f},"
                f"mase={score_dict['Hard_Tree'][3]:.6f}"                
            )

            print("Training model: LGBM")
            regressor = m3.build_lgbm_regressor(
                train_size=len(X_train),
                feature_count=X_train.shape[1],
            )
            regressor.fit(X_train, y_train)
            model_pred = regressor.predict(X_test)
            pd.Series(model_pred, index=range(train_len, train_len + forecast_horizon)).plot(
                ls="--", label="pred, LGBM", ax=ax
            )
            score_dict["LGBM"] = (
                mse(y_pred=model_pred, y_true=y_test),
                mae(y_pred=model_pred, y_true=y_test),
                mape(y_pred=model_pred, y_true=y_test),
                mase(y_true=y_test, y_pred=model_pred, y_train=y_train)
            )
            print(
                f"Completed LGBM: mse={score_dict['LGBM'][0]:.6f}, "
                f"mae={score_dict['LGBM'][1]:.6f}, mape={score_dict['LGBM'][2]:.6f}, "
                f"mase={score_dict['LGBM'][3]:.6f}"
            )

            print("Training model: XGB")
            regressor = xgb.XGBRegressor(random_state=0)
            regressor.fit(X_train, y_train)
            model_pred = regressor.predict(X_test)
            pd.Series(model_pred, index=range(train_len, train_len + forecast_horizon)).plot(
                ls="--", label="pred, XGB", ax=ax
            )
            score_dict["XGB"] = (
                mse(y_pred=model_pred, y_true=y_test),
                mae(y_pred=model_pred, y_true=y_test),
                mape(y_pred=model_pred, y_true=y_test),
                mase(y_true=y_test, y_pred=model_pred, y_train=y_train)
            )
            print(
                f"Completed XGB: mse={score_dict['XGB'][0]:.6f}, "
                f"mae={score_dict['XGB'][1]:.6f}, mape={score_dict['XGB'][2]:.6f}, "
                f"mase={score_dict['XGB'][3]:.6f}"
            )

            X_train_scaled, X_test_scaled = X_scaled.iloc[:-forecast_horizon], X_scaled.iloc[-forecast_horizon:]

            print("Training model: MLP")
            regressor = MLPRegressor(random_state=0)
            regressor.fit(X_train_scaled, y_train)
            model_pred = regressor.predict(X_test_scaled)
            pd.Series(model_pred, index=range(train_len, train_len + forecast_horizon)).plot(
                ls="--", label="pred, MLP", ax=ax
            )
            score_dict["MLP"] = (
                mse(y_pred=model_pred, y_true=y_test),
                mae(y_pred=model_pred, y_true=y_test),
                mape(y_pred=model_pred, y_true=y_test),
                mase(y_true=y_test, y_pred=model_pred, y_train=y_train)
            )
            print(
                f"Completed MLP: mse={score_dict['MLP'][0]:.6f}, "
                f"mae={score_dict['MLP'][1]:.6f}, mape={score_dict['MLP'][2]:.6f}"
                f"mase={score_dict['MLP'][3]:.6f}"
            )

            print("Training model: Linear")
            regressor = LinearRegression()
            regressor.fit(X_train_scaled, y_train)
            model_pred = regressor.predict(X_test_scaled)
            pd.Series(model_pred, index=range(train_len, train_len + forecast_horizon)).plot(
                ls="--", label="pred, Linear", ax=ax
            )
            score_dict["Linear"] = (
                mse(y_pred=model_pred, y_true=y_test),
                mae(y_pred=model_pred, y_true=y_test),
                mape(y_pred=model_pred, y_true=y_test),
                mase(y_true=y_test, y_pred=model_pred, y_train=y_train)
            )
            print(
                f"Completed Linear: mse={score_dict['Linear'][0]:.6f}, "
                f"mae={score_dict['Linear'][1]:.6f}, mape={score_dict['Linear'][2]:.6f}"
                f"mase={score_dict['Linear'][3]:.6f}"
            )

            print("Running baseline: Naive")
            model_pred = y.iloc[-test_len - 1 : -1].to_numpy()
            pd.Series(model_pred, index=range(train_len, train_len + forecast_horizon)).plot(
                ls="--", label="pred, Naive", ax=ax
            )
            score_dict["Naive"] = (
                mse(y_pred=model_pred, y_true=y_test),
                mae(y_pred=model_pred, y_true=y_test),
                mape(y_pred=model_pred, y_true=y_test),
                mase(y_true=y_test, y_pred=model_pred, y_train=y_train)
            )
            print(
                f"Completed Naive: mse={score_dict['Naive'][0]:.6f}, "
                f"mae={score_dict['Naive'][1]:.6f}, mape={score_dict['Naive'][2]:.6f}"
                f"mase={score_dict['Naive'][3]:.6f}"
            )

            ax.legend()
            ax.set_title(
                f"{ett_type} Sequence\n"
                f"MSEs: ours={test_mse:.4f}, SARIMAX={sarimax_test_mse:.4f}"
            )
            fig = ax.figure
            fig.tight_layout()
            fig.savefig(
                os.path.join("ett_results", f"{ett_type}_results.png"),
                bbox_inches="tight",
            )
            plt.close(fig)
            print(f"Saved plot: ett_results/{ett_type}_results.png")

            all_result_dict[ett_type] = score_dict
            pd.to_pickle(all_result_dict, f"ALL_RESULT_ett_{ett_type}.pkl")
            print(f"Saved metrics: ALL_RESULT_ett_{ett_type}.pkl")
            print(f"Finished dataset: {ett_type}")
        except Exception as exc:
            print(f"Something went wrong in {ett_type}")
            print(exc)
