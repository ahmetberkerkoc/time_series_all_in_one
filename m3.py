import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import tanh
from scipy.special import expit as sigmoid
import warnings

rng = np.random.default_rng()

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
import xgboost as xgb
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.statespace.sarimax import SARIMAX


import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    mean_absolute_percentage_error as mape,
    mean_squared_error as mse,
    mean_absolute_error as mae,
)
from sklearn.preprocessing import MinMaxScaler

# MEAS_NOISE_STD = 0.01
# UPD_NOISE_STD = 0.1


def build_lgbm_regressor(train_size, feature_count):
    # Small M3 splits can have fewer than 10 training rows, so LightGBM's
    # defaults often cannot form a valid split and emit "best gain: -inf".
    min_child_samples = max(1, min(5, train_size // 2))
    num_leaves = max(2, min(31, 2 ** min(feature_count, 4)))

    return lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        min_split_gain=0.0,
        random_state=0,
        verbose=-1,
    )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class TreeNode:
    def __init__(self, x, prob, weight, input):
        self.val = x
        self.prob = prob
        self.weight = weight
        self.input = input
        self.left = None
        self.right = None


def node_operation(weight, input):
    W, b = weight
    b = b.squeeze()
    # out = sigmoid(W*input+b)
    out = sigmoid(np.einsum("BX,BX->B", W, input) + b)
    return (out, 1 - out)


def leaf_prediction(weight, input):
    W, b = weight
    b = b.squeeze()
    # out = W*input+b

    out = np.einsum("BX,BX->B", W, input) + b
    return out


def buildTree(nums, weights, input, max_depth):
    start_index = 2 * (max_depth)
    final_index = 2 ** (max_depth + 1)
    final_prediction_list = []
    if not nums:
        return None
    root = TreeNode(x=nums[0], prob=1, weight=weights[0], input=input)
    q = [root]
    i = 1
    while i < len(nums):
        curr = q.pop(0)
        left_prob, right_prob = node_operation(curr.weight, curr.input)
        left_prob *= curr.prob
        right_prob *= curr.prob
        if i < len(nums):

            curr.left = TreeNode(x=nums[i], prob=left_prob, weight=weights[i], input=input)
            q.append(curr.left)
            if curr.left.val >= start_index and curr.left.val < final_index:
                pred = leaf_prediction(curr.left.weight, input)
                pred *= left_prob
                final_prediction_list.append(pred)

            i += 1

        if i < len(nums):

            curr.right = TreeNode(x=nums[i], prob=right_prob, weight=weights[i], input=input)
            q.append(curr.right)
            if curr.right.val >= start_index and curr.right.val < final_index:
                pred = leaf_prediction(curr.right.weight, input)
                pred *= right_prob
                final_prediction_list.append(pred)
            i += 1

    final_pred = sum(final_prediction_list) / len(final_prediction_list)
    return root, final_pred


class SX_sGBDT:
    def __init__(
        self,
        n_estimator=1,
        depth=1,
        sx_order=(1, 0, 0),
        sx_seas_order=None,
        exog=None,
        n_particles=500,
        resampling_threshold=0.5,
    ):
        self.n_particles = n_particles
        self.resampling_threshold = resampling_threshold
        self._has_sgbdt = True
        # sSGDT part
        if exog is None:
            warnings.warn("No exogenous variable passed; sGBDT will be inactive")
            self._has_sgbdt = False
            self.exog = None
        else:
            self.exog = np.asarray(exog)
            assert self.exog.ndim == 2, "non-2D exog"
            self.n_estimators = n_estimator
            self.depth = depth
            self.n_node = 2 ** (self.depth + 1) - 1
            self.leaf_node = 2**self.depth

        # SARIMAX
        self.sx_p, self.sx_d, self.sx_q = sx_order

        # Seasonal Part
        if sx_seas_order is not None:
            self.sx_P, self.sx_D, self.sx_Q, self.sx_m = sx_seas_order
        else:
            self.sx_P, self.sx_D, self.sx_Q, self.sx_m = 0, 0, 0, None

        state_dim = self.sx_p + self.sx_q + self.sx_P + self.sx_Q

        if self._has_sgbdt:
            state_dim += exog.shape[1]
            for i in range(self.n_estimators):
                state_dim += self.n_node * (exog.shape[1] * 1)  # weight
                state_dim += self.n_node * 1  # bias

        self.particles = rng.normal(size=(n_particles, state_dim))
        self.weights = rng.dirichlet([1] * n_particles)
        assert np.isclose(self.weights.sum(), 1), "dirichlet broken"
        #assert np.alen(self.particles) == np.alen(self.weights), "p <!-> w"

        self._measurements = []
        self._errors = []
        self.predictions = []

    def _sgbdt_state_transition(self):
        n_particles = self.n_particles
        n_estimators, depth, n_node, n_input = self.n_estimators, self.depth, self.n_node, self.exog.shape[1]
        particles = self.particles
        W_list = [0] * n_node * n_estimators
        b_list = [0] * n_node * n_estimators

        # sSGDT-related state elements start after those of SARIMAX end
        sx_offset = self.sx_p + self.sx_q + self.sx_P + self.sx_Q + n_input
        for i in range(n_estimators):
            estimators_offset = i * n_estimators
            for j in range(n_node):
                W_list[j + i * n_node] = particles[
                    :, sx_offset + estimators_offset + j * n_input : sx_offset + estimators_offset + (j + 1) * n_input
                ]

        soft_tree_weight_ofset = n_estimators * n_node * n_input
        for i in range(n_estimators):
            estimators_offset = i * n_estimators
            for j in range(n_node):
                b_list[i] = particles[
                    :,
                    sx_offset
                    + soft_tree_weight_ofset
                    + estimators_offset
                    + j * 1 : sx_offset
                    + soft_tree_weight_ofset
                    + estimators_offset
                    + (j + 1) * 1,
                ]

        particles[:, sx_offset : sx_offset + n_estimators * n_node * (n_input + 1)] += rng.normal(
            scale=UPD_NOISE_STD, size=(n_particles, n_estimators * n_node * (n_input + 1))
        )

    def take_prediction(self, W_list, b_list, max_depth, input):

        weigths = list(zip(W_list, b_list))
        nums = list(range(1, 2 ** (max_depth + 1)))
        root, final_pred = buildTree(nums, weigths, input, max_depth)
        return final_pred

    def _sgbdt_prediction(self):
        n_particles = self.n_particles
        n_estimators, depth, n_node, leaf_node, n_feature = (
            self.n_estimators,
            self.depth,
            self.n_node,
            self.leaf_node,
            self.exog.shape[1],
        )
        particles = self.particles

        W_list = [0] * n_node * n_estimators
        b_list = [0] * n_node * n_estimators

        sx_offset = self.sx_p + self.sx_q + self.sx_P + self.sx_Q + n_feature
        for i in range(n_estimators):
            estimators_offset = i * n_node
            for j in range(n_node):
                W_list[j + i * n_node] = particles[
                    :,
                    sx_offset + estimators_offset + j * n_feature : sx_offset + estimators_offset + (j + 1) * n_feature,
                ]

        soft_tree_weight_ofset = n_estimators * n_node * n_feature
        for i in range(n_estimators):
            estimators_offset = i * n_node
            for j in range(n_node):
                b_list[j + i * n_node] = particles[
                    :,
                    sx_offset
                    + soft_tree_weight_ofset
                    + estimators_offset
                    + j * 1 : sx_offset
                    + soft_tree_weight_ofset
                    + estimators_offset
                    + (j + 1) * 1,
                ]

        pred = self.take_prediction(W_list, b_list, depth, self.exog)
        return pred

    def _sx_state_transition(self):
        # SARIMAX-related state components are at the beginning
        sx_size = self.sx_p + self.sx_q + self.sx_P + self.sx_Q
        if self.exog is not None:
            sx_size += self.exog.shape[1]
        self.particles[:, :sx_size] += rng.normal(scale=UPD_NOISE_STD, size=(self.n_particles, sx_size))

    def _sx_prediction(self, t):
        # Form r_t := [y_{t-1}, ..., y_{t-p},
        #              y_{t-m}, ..., y_{t-mP},
        #              e_{t-1}, ..., e_{t-q},
        #              e_{t-m}, ..., e_{t-mQ}]
        r_t = []

        # Check if we are at the start yet; fill with 0 if so. Otherwise,
        # last p/q items are taken.
        n_meas, n_errs = len(self._measurements), len(self._errors)

        # AR part
        if n_meas < self.sx_p:
            r_t.extend(self._measurements[::-1] + [0] * (self.sx_p - n_meas))
        elif self.sx_p:  # can say `else` as well but this saves a bit of time
            r_t.extend(self._measurements[: -1 - self.sx_p : -1])

        # Seasonal AR part
        _sx_has_seasonal_part = self.sx_P != 0 or self.sx_Q != 0
        if _sx_has_seasonal_part:
            # say m = 12, P = 3
            # so, need -12, -24, -36th values
            if n_meas < self.sx_m * self.sx_P:
                r_t.extend(
                    self._measurements[-self.sx_m :: -self.sx_m]
                    + [0] * np.ceil(self.sx_P - n_meas / self.sx_m).astype(int)
                )
                assert self.sx_P - n_meas / self.sx_m > 0, "mP versus n_meas gone wrong"
            elif self.sx_P:
                r_t.extend(self._measurements[-self.sx_m : -1 - self.sx_m * self.sx_P : -self.sx_m])

        # MA part
        if n_errs < self.sx_q:
            r_t.extend(self._errors[::-1] + [0] * (self.sx_q - n_errs))
        elif self.sx_q:
            r_t.extend(self._errors[-1 : -1 - self.sx_q : -1])

        # Seasonal MA part
        if _sx_has_seasonal_part:
            if n_errs < self.sx_m * self.sx_Q:
                r_t.extend(
                    self._errors[-self.sx_m :: -self.sx_m] + [0] * np.ceil(self.sx_Q - n_errs / self.sx_m).astype(int)
                )
                assert self.sx_Q - n_errs / self.sx_m > 0, "mQ versus n_errs gone wrong"
            elif self.sx_Q:
                r_t.extend(self._errors[-self.sx_m : -1 - self.sx_m * self.sx_Q : -self.sx_m])

        # Exogenous part
        if self.exog is not None:
            r_t.extend(self.exog[t])

        # Get SARIMAX's particles' predictions
        sx_size = self.sx_p + self.sx_q + self.sx_P + self.sx_Q
        if self.exog is not None:
            sx_size += self.exog.shape[1]
        sx_particles = self.particles[:, :sx_size]
        sx_pred = sx_particles @ r_t
        return sx_pred

    def _normalize_weights(self):
        """
        Assure the weights behave like a probability distribution by normalizing
        them with their sum. If sum is near 0, reinitialize the weights uniformly,
        i.e., w_i = 1 / N for all i in [1, N].
        """
        Z = self.weights.sum()
        if not np.isclose(Z, 0):
            self.weights /= Z
        else:
            warnings.warn("Sum of weights is nearly 0, can't normalize; will uniformize")
            self.weights = np.full_like(self.weights, fill_value=1 / self.n_particles)

    def _compute_likelihood(self, t, y_t):
        """
        Measure how likely each particle is, i.e., p(y_t | \vec{s}_t^{(i)}).
        Assume Gaussian for {y_t}, centered around the specific measurement; so
        compute the prediction and measure the likelihood with it.
        """
        # Record the measurement
        self._measurements.append(y_t)

        # Gather base predictions and combine them to get predictions of each particle
        sx_y_hats_t = self._sx_prediction(t)
        sgbdt_y_hats_t = self._sgbdt_prediction() if self._has_sgbdt else 0
        # print("SX preds:", sx_y_hats_t)
        # print("\nsgbdt preds:", sgbdt_y_hats_t)
        y_hats_t = sx_y_hats_t + sgbdt_y_hats_t

        # Error and likelihood computation
        errs_t = y_t - y_hats_t
        likelihood = np.exp(-(errs_t**2) / (2 * MEAS_NOISE_STD**2)) / MEAS_NOISE_STD / np.sqrt(2 * np.pi)
        # print("max likelihood:", likelihood.max())

        # Before leaving, get a collective estimate out of particles & record
        y_hat_t = self.weights @ y_hats_t
        self.predictions.append(y_hat_t)
        self._errors.append(y_t - y_hat_t)

        return likelihood

    def update_weights(self, time_idx, measurement):
        """
        w_t^{(i)} = w_{t-1}^{(i)} p(y_t | \vec{s}_t^{(i)})
        Then normalize.
        """
        self.weights *= self._compute_likelihood(time_idx, measurement)
        self._normalize_weights()

    def update_particles(self, t):
        """
        Perform the state transition equation, i.e.,
        \vec{s}_t = \vec{s}_{t-1} + \vec{e}_t
        """
        self._sx_state_transition()
        if self._has_sgbdt:
            self._sgbdt_state_transition()

    def resample(self):
        effective_size = 1 / (self.weights**2).sum()
        if effective_size < self.resampling_threshold * self.n_particles:
            # Weighted sampling with replacement
            self.particles = rng.choice(self.particles, size=self.n_particles, replace=True, p=self.weights, axis=0)
            # Uniformize weights
            self.weights = np.full_like(self.weights, fill_value=1 / self.n_particles)


def filter_sgbdtsx(
    ys, sgbdt_hidden_size=20, sx_order=(1, 0, 0), sx_seas_order=None, exog=None, n_particles=100, resampling_thre=0.5
):

    if exog is not None:
        exog = np.asarray(exog)
        assert exog.ndim == 2, "Exogenous non-2D"

    pf = SX_sGBDT(1, 3, sx_order, sx_seas_order, exog, n_particles, resampling_thre)

    for t, y_t in enumerate(ys):
        pf.update_particles(t)
        pf.update_weights(t, y_t)
        pf.resample()
    return pf




import random
import os
if __name__ == "__main__":

    ALL_RESULT_DICT = {}
    os.makedirs("m3_results", exist_ok=True)
    
    for m3_type in ["monthly", "quarterly", "yearly"]:
        
        try:
            csv_list = os.listdir(f"data/M3/Extracted/{m3_type}")
            print(m3_type)

            forecast_dic = {"monthly": 18, "quarterly": 8, "yearly": 6}
            np.random.seed(0)
            random.seed(0)
            random.shuffle(csv_list)

            forecast_horizion = forecast_dic[m3_type]

            #feature_start_index = {"Daily": 7, "Weekly": 8, "Hourly": 7}
            
            counter = 0 
            deleted_data = 0
            for i in csv_list:
                datapath = f"data/M3/Extracted/{m3_type}/{i}"
                    
                print(counter)
                counter +=1
                
                score_dict = {}

                df = pd.read_csv(datapath)
                #df = df.drop("Unnamed: 0", axis=1)
                
                y = df.loc[:, "y"]
                X = df.loc[:, [col for col in df.columns if col != "y"]] #df.iloc[:, feature_start_index[m3_type]:]
                
                UPD_NOISE_STD = 0.1
                MEAS_NOISE_STD = 1

                order = 2, 0, 0
                X = pd.DataFrame(MinMaxScaler().fit_transform(X), index=X.index, columns=X.columns)

                pf = filter_sgbdtsx(y, sx_order=order, exog=X, n_particles=len(y))
                stats_sx = SARIMAX(y, X,order=order).fit()

                # In sample predictions
                our_preds = pd.Series(pf.predictions)
                stats_preds = stats_sx.predict()
                
                test_len = forecast_horizion
                train_len = len(y) - test_len
                
                if our_preds[-test_len:].isna().any():
                    deleted_data +=1
                    print(deleted_data) 
                    continue
                
                ax = y.iloc[-test_len:].plot(label="true")
                our_preds.iloc[-test_len:].plot(figsize=(14, 5), ls="--", label="pred, ours", ax=ax)
                stats_preds.iloc[-test_len:].plot(ls="--", label="pred, statsmodels", ax=ax)
                
                
                test_mse = mse(y_pred=our_preds[-test_len:], y_true=y[-test_len:])
                test_mae = mae(y_pred=our_preds[-test_len:], y_true=y[-test_len:])
                test_mape = mape(y_pred=our_preds[-test_len:], y_true=y[-test_len:])
                score_dict["our"] = (test_mse, test_mae, test_mape)

                sarimax_test_mse = mse(y_pred=stats_preds[-test_len:], y_true=y[-test_len:])
                sarimax_test_mae = mae(y_pred=stats_preds[-test_len:], y_true=y[-test_len:])
                sarimax_test_mape = mape(y_pred=stats_preds[-test_len:], y_true=y[-test_len:])

                score_dict["sarimax"] = (sarimax_test_mse, sarimax_test_mae, sarimax_test_mape)

                
                y = df.loc[:, "y"]
                X = X = df.loc[:, [col for col in df.columns if col != "y"]]#df.iloc[:,  feature_start_index[m3_type]:]
                #X = pd.DataFrame(MinMaxScaler().fit_transform(X), index=X.index, columns=X.columns)

                X_train, X_test = X.iloc[:-forecast_horizion], X.iloc[-forecast_horizion:]
                y_train, y_test = y.iloc[:-forecast_horizion], y.iloc[-forecast_horizion:]

                # HARD DECISON TREE

                regressor = DecisionTreeRegressor(random_state=0)
                regressor.fit(X_train, y_train)
                model_pred = regressor.predict(X_test)
                model_pred_series = pd.Series(model_pred,index=range(train_len,train_len + forecast_horizion))
                
                model_pred_series.plot(ls="--", label="pred, HDT", ax=ax)
                
                model_test_mse = mse(y_pred=model_pred, y_true=y[-test_len:])
                model_test_mae = mae(y_pred=model_pred, y_true=y[-test_len:])
                model_test_mape = mape(y_pred=model_pred, y_true=y[-test_len:])

                score_dict["Hard_Tree"] = (model_test_mse, model_test_mae, model_test_mape)

                
                
                

                # LGBM
                regressor = build_lgbm_regressor(
                    train_size=len(X_train),
                    feature_count=X_train.shape[1],
                )
                regressor.fit(X_train, y_train)
                model_pred = regressor.predict(X_test)
                model_pred_series = pd.Series(model_pred,index=range(train_len,train_len + forecast_horizion))
                model_pred_series.plot(ls="--", label="pred, LGBM", ax=ax)
                
                model_test_mse = mse(y_pred=model_pred, y_true=y[-test_len:])
                model_test_mae = mae(y_pred=model_pred, y_true=y[-test_len:])
                model_test_mape = mape(y_pred=model_pred, y_true=y[-test_len:])

                score_dict["LGBM"] = (model_test_mse, model_test_mae, model_test_mape)
                
                # XGB
                regressor = xgb.XGBRegressor(random_state=0)
                regressor.fit(X_train, y_train)
                model_pred = regressor.predict(X_test)
                model_pred_series = pd.Series(model_pred,index=range(train_len,train_len + forecast_horizion))
                model_pred_series.plot(ls="--", label="pred, XGB", ax=ax)
                
                model_test_mse = mse(y_pred=model_pred, y_true=y[-test_len:])
                model_test_mae = mae(y_pred=model_pred, y_true=y[-test_len:])
                model_test_mape = mape(y_pred=model_pred, y_true=y[-test_len:])

                score_dict["XGB"] = (model_test_mse, model_test_mae, model_test_mape)
                
                X = pd.DataFrame(MinMaxScaler().fit_transform(X), index=X.index, columns=X.columns)

                X_train, X_test = X.iloc[:-forecast_horizion], X.iloc[-forecast_horizion:]
                y_train, y_test = y.iloc[:-forecast_horizion], y.iloc[-forecast_horizion:]
                
                # MLP
                regressor = MLPRegressor(random_state=0)
                regressor.fit(X_train, y_train)
                model_pred = regressor.predict(X_test)
                model_pred_series = pd.Series(model_pred,index=range(train_len,train_len + forecast_horizion))
                model_pred_series.plot(ls="--", label="pred, MLP", ax=ax)
                
                model_test_mse = mse(y_pred=model_pred, y_true=y[-test_len:])
                model_test_mae = mae(y_pred=model_pred, y_true=y[-test_len:])
                model_test_mape = mape(y_pred=model_pred, y_true=y[-test_len:])

                score_dict["MLP"] = (model_test_mse, model_test_mae, model_test_mape)
                
                # Linear
                regressor = LinearRegression()
                regressor.fit(X_train, y_train)
                model_pred = regressor.predict(X_test)
                model_pred_series = pd.Series(model_pred,index=range(train_len,train_len + forecast_horizion))
                model_pred_series.plot(ls="--", label="pred, Linear", ax=ax)

                model_test_mse = mse(y_pred=model_pred, y_true=y[-test_len:])
                model_test_mae = mae(y_pred=model_pred, y_true=y[-test_len:])
                model_test_mape = mape(y_pred=model_pred, y_true=y[-test_len:])

                score_dict["Linear"] = (model_test_mse, model_test_mae, model_test_mape)

                # NAIVE
                model_pred = y[-test_len - 1 : -1]
                model_pred_series = pd.Series(model_pred,index=range(train_len,train_len + forecast_horizion))
                model_pred_series.plot(ls="--", label="pred, Naive", ax=ax)
                
                model_test_mse = mse(y_pred=model_pred, y_true=y[-test_len:])
                model_test_mae = mae(y_pred=model_pred, y_true=y[-test_len:])
                model_test_mape = mape(y_pred=model_pred, y_true=y[-test_len:])

                score_dict["Naive"] = (model_test_mse, model_test_mae, model_test_mape)
                    
                ax.legend()
                ax.set_title(
                    f"{m3_type}_{i}'th Sequence\n"
                    f"MSEs: ours={test_mse:.4f}, SARIMAX={sarimax_test_mse:.4f}"
                )
                fig = ax.figure
                fig.tight_layout()
                fig.savefig(
                    os.path.join("m3_results", f"{m3_type}_{i}_v2.png"),
                    bbox_inches="tight",
                )
                plt.close(fig)
                
                ALL_RESULT_DICT[f"{m3_type}_{i}"] = score_dict

                pd.to_pickle(ALL_RESULT_DICT,f"ALL_RESULT_m3_{m3_type}.pkl")
        
        except Exception as e:
            print(f"Something went wrong in {counter}")
            print(e)
            continue
