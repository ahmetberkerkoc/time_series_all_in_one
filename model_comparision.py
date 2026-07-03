#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import itertools

#models
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.arima.model import ARIMA


# # Metrics

# In[2]:


from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

def MAPE(y_test, pred):
    mape = np.mean(np.abs((y_test - pred) / y_test))
    return mape


# # Parameters

# In[3]:


label_name = "y"
test_size = 0.3
date_column_name = "date"
model_name = "lightgbm"


# # Read Data

# In[4]:


data_path = "DailyDelhiClimate.csv"


# In[38]:


df = pd.read_csv(data_path)
df = df.drop("Unnamed: 0",axis=1)

if date_column_name is not None:
    df = df.drop(date_column_name, axis=1)
col_list = list(df.columns)
col_list.remove(label_name)
col_list.insert(0, label_name)

df = df[col_list]

y = df.loc[:,"y"]
X = df.iloc[:,1:]

data_len = len(X)
forecast_horizion = int(data_len * test_size)

X_train, X_test = X.iloc[:-forecast_horizion], X.iloc[-forecast_horizion:]
y_train, y_test = y.iloc[:-forecast_horizion], y.iloc[-forecast_horizion:]


# # Model

# In[39]:


model_dic = {
    "lightgbm": lgb.LGBMRegressor(),
    "xgboost": xgb.XGBRegressor(),
    "catboost": cb.CatBoostRegressor(),
    "decision_tree": DecisionTreeRegressor()
 }

search_param_dic = {
    "lightgbm": {
            "n_estimators": [100, 250, 500, 750, 1000],
            "learning_rate": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            "num_leaves": [15, 31, 63, 127, 255],
            "max_depth": [4, 6, 7, 8, 10],
            "subsample": [0.4, 0.6, 0.7, 0.9],
            "subsample_freq": [1, 5, 10, 20, 50],
            "colsample_bytree": [0.4, 0.6, 0.7, 0.9],
            "reg_alpha": [0, 0.01, 0.05, 0.5, 1, 10],
            "reg_lambda": [0, 0.01, 0.05, 0.5, 1, 10],
            "max_bin": [15, 31, 63, 127, 255],
            "random_state": [0],
            "verbose": [-1],
        },
    "xgboost": {
            "n_estimators": [100, 250, 500, 750, 1000],
            "learning_rate": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            "num_leaves": [15, 31, 63, 127, 255],
            "max_depth": [4, 6, 7, 8, 10],
            "subsample": [0.4, 0.6, 0.7, 0.9],
            "subsample_freq": [1, 5, 10, 20, 50],
            "colsample_bytree": [0.4, 0.6, 0.7, 0.9],
            "reg_alpha": [0, 0.01, 0.05, 0.5, 1, 10],
            "reg_lambda": [0, 0.01, 0.05, 0.5, 1, 10],
            "max_bin": [15, 31, 63, 127, 255],
            "random_state": [0],
            "verbose": [-1],
        },
    "catboost": {
        "iterations": [100, 300, 500, 1000],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "depth": [4, 6, 8, 10],
        "l2_leaf_reg": [1, 3, 5, 7, 9],
        "border_count": [32, 64, 128, 254],
        "random_strength": [0, 0.1, 0.5, 1],
        "random_seed": [0],
    }
,
    "decision_tree": {
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 3, 4, 5, 10, 12],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 10],
}
 }


model = model_dic[model_name]
searching_params = search_param_dic[model_name]


# # Parameter Search

# In[41]:


# Hold-out Validation
xv_cls = RandomizedSearchCV
val_size = forecast_horizion
train_val_indexes = np.zeros_like(y_train)
train_val_indexes[:-val_size] = -1
fold_size = PredefinedSplit(test_fold=train_val_indexes)

xv = xv_cls(estimator=model, param_distributions=searching_params, n_iter=10000, scoring="neg_mean_absolute_error", n_jobs=-1,
                    cv=fold_size, verbose=-1, refit=False)

xv.fit(X_train, y_train)

best_params = xv.best_params_


# # Fit & Predict

# In[ ]:


X_train, X_test = X.iloc[:-forecast_horizion], X.iloc[-forecast_horizion:]
y_train, y_test = y.iloc[:-forecast_horizion], y.iloc[-forecast_horizion:]

best_model = type(model)(**best_params).fit(X_train, y_train)
preds = best_model.predict(X_train)
fores = best_model.predict(X_test)


# # Score Calculation

# In[ ]:


train_mse_score = mse(y_train,preds)
train_mae_score = mae(y_train,preds)
train_mape_score = MAPE(y_train,preds)

test_mse_score = mse(y_test,fores)
test_mae_score = mae(y_test,fores)
test_mape_score = MAPE(y_test,fores)


# # PLOT

# In[ ]:


plt.plot(y_train,label = "target")
plt.plot(preds,label="preds")
plt.legend()
plt.title("Train Targets vs Train Preds")
plt.show()

plt.plot(y_test,label = "target")
plt.plot(fores,label="fores")
plt.legend()
plt.title("Test Labels vs Test Forecasts")
plt.show()


# #######################################

# # SARIMAX

# In[16]:


df = pd.read_csv(data_path)
df = df.drop("Unnamed: 0",axis=1)

if date_column_name is not None:
    df = df.drop(date_column_name, axis=1)
col_list = list(df.columns)
col_list.remove(label_name)
col_list.insert(0, label_name)

df = df[col_list]

y = df.loc[:,"y"]
X = df.iloc[:,1:]

data_len = len(X)
forecast_horizion = int(data_len * test_size)

value = y
train_value = X.iloc[:-forecast_horizion]
test_value = X.iloc[-forecast_horizion:]


# In[ ]:


# Define the parameter space for grid search
p = d = q = range(0, 3)  # Example range, adjust as necessary
P = D = Q = range(0, 3)  # Example range, adjust as necessary
s = [12,24]  # Seasonal period, adjust based on your data's seasonality


best_mse = float("inf")
for param in [(x[0], x[1], x[2]) for x in itertools.product(p, d, q)]:
    for seasonal_param in [(x[0], x[1], x[2], x[3]) for x in itertools.product(P, D, Q, s)]:

        arima_model = ARIMA(
                        value[:-2*forecast_horizion],
                        order=param,
                        exog=train_value[:-forecast_horizion],
                        seasonal_order=seasonal_param
                    )
        model = arima_model.fit()

        start_index = len(train_value)-forecast_horizion
        end_index = start_index + forecast_horizion - 1

        forecast = model.predict(start=start_index, end=end_index, exog=train_value[-forecast_horizion:])

        y_test_np = value[-2*forecast_horizion:-forecast_horizion].to_numpy()
        forecast = forecast.to_numpy()
        
        mse_score = mse(y_test_np, forecast)
        mae_score = mae(y_test_np, forecast)
        mape_score = MAPE(y_test_np, forecast)
        
        if mse_score<best_mse:
            best_mse = mse_score
            best_order = param
            best_seasonal_order = seasonal_param
        
        print(mse_score)
        print(mae_score)
        print(mape_score)


# In[ ]:


best_model = ARIMA(
                        value[:-forecast_horizion],
                        order=best_order,
                        exog=train_value,
                        seasonal_order=best_seasonal_order
                    )
model = best_model.fit()

start_index = len(train_value)
end_index = start_index + forecast_horizion - 1

forecast = model.predict(start=start_index, end=end_index, exog=test_value)

y_test_np = value[-forecast_horizion:].to_numpy()
forecast = forecast.to_numpy()

mse_score = mse(y_test_np, forecast)
mae_score = mae(y_test_np, forecast)
mape_score = MAPE(y_test_np, forecast)

print(mse_score)
print(mae_score)
print(mape_score)


# # LSTM

# In[20]:


df = pd.read_csv(data_path)
df = df.drop("Unnamed: 0",axis=1)

if date_column_name is not None:
    df = df.drop(date_column_name, axis=1)
col_list = list(df.columns)
col_list.remove(label_name)
col_list.insert(0, label_name)

df = df[col_list]

y = df.loc[:,"y"]
X = df.iloc[:,1:]

y = y.values
X = X.values

data_len = len(X)
forecast_horizion = int(data_len * test_size)

y_train_orig = y[:-forecast_horizion]
y_test_orig = y[-forecast_horizion:]

feature_shape = X.shape[1]


# In[15]:


# Normalize the data
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Create sequences for input and output
def create_sequences(data, target, n_steps):
    X_seq, y_seq = [], []
    for i in range(len(data) - n_steps):
        X_seq.append(data[i : i + n_steps])
        y_seq.append(target[i + n_steps])
    return np.array(X_seq), np.array(y_seq)

n_steps = 10  # number of time steps to look back
X_seq, y_seq = create_sequences(X_scaled, y_scaled, n_steps)

# Split the data into training and testing sets

X_train, X_test = X_seq[:-forecast_horizion], X_seq[-forecast_horizion:]
y_train, y_test = y_seq[:-forecast_horizion], y_seq[-forecast_horizion:]


# In[ ]:


# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, activation="relu", input_shape=(n_steps, feature_shape)))
model.add(Dense(units=1))
model.compile(optimizer="adam", loss="mean_absolute_error")

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)

print(f"Training Loss: {train_loss}")
print(f"Test Loss: {test_loss}")

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# In[19]:


# Inverse transform the predictions to the original scale
y_train_pred_inv = scaler_y.inverse_transform(y_train_pred).reshape(-1)
y_test_pred_inv = scaler_y.inverse_transform(y_test_pred).reshape(-1)

y_test = y_test_orig.copy()

test_mse_score = mse(y_test, y_test_pred_inv)
test_mae_score = mae(y_test, y_test_pred_inv)
test_mape_score = MAPE(y_test, y_test_pred_inv)


# # MLP - ARMA

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import pandas as pd
import numpy as np

# from SDT import SDT
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import random
from torch.utils.data import TensorDataset, DataLoader
import os
import matplotlib.pyplot as plt
from SDT import MLPModel, MLPModel_deep, MLPModel_residual


def mape(y_test, pred):
    mape = np.mean(np.abs((y_test - pred) / y_test))
    return mape


exp_name = "experiment"


model_type = "mlp_model" # mlp_model, deep_mlp_model, #mlp_model_residual_connections

lr = 1e-2  # learning rate
epochs = 30 # the number of training epochs

if not os.path.exists("Results"):
    os.makedirs("Results")

result_folder = f"Results/sdt_arm_{exp_name}"
result_txt_file = f"sdt_arma_{exp_name}.txt"

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# Parameters
# the number of input dimensions
output_dim = 1  # the number of outputs (i.e., # classes on MNIST)
batch_size = 1  # batch size
use_cuda = torch.cuda.is_available()  # whether to use GPU

device = torch.device("cuda" if use_cuda else "cpu")

# reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Load data

##########################

mu, sigma = 0, 0.1

df = pd.read_csv(data_path)
df = df.drop("Unnamed: 0", axis=1)
if date_column_name is not None:
    df = df.drop(date_column_name, axis=1)
col_list = list(df.columns)
col_list.remove(label_name)
col_list.insert(0, label_name)

df = df[col_list]

y = df.loc[:, label_name]
X = df.iloc[:, 1:]
data_len = len(X)

forecast_horizon = int(data_len * test_size)
e_t = np.random.normal(mu, sigma, len(X))

X["e"] = e_t
X["e-1"] = 0
X["e-2"] = 0
X["e-3"] = 0

X_train, X_test = X.iloc[:-forecast_horizon].to_numpy().astype(np.float32), X.iloc[
    -forecast_horizon:
].to_numpy().astype(np.float32)
y_train, y_test = y.iloc[:-forecast_horizon].to_numpy().astype(np.float32), y.iloc[
    -forecast_horizon:
].to_numpy().astype(np.float32)

scaler = MinMaxScaler()
X_train_arr = scaler.fit_transform(X_train)
X_test_arr = scaler.transform(X_test)

y_train_arr = scaler.fit_transform(y_train.reshape(-1, 1))
y_test_arr = scaler.transform(y_test.reshape(-1, 1))

train_features = torch.Tensor(X_train_arr).to(device)
train_targets = torch.Tensor(y_train_arr).to(device)
test_features = torch.Tensor(X_test_arr).to(device)
test_targets = torch.Tensor(y_test_arr).to(device)

train = TensorDataset(train_features, train_targets)
test = TensorDataset(test_features, test_targets)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

input_dim = train_features.shape[1]
print(f"input_dim: {input_dim}")
##########################

# Model and Optimizer
if model_name == "mlp_model":
    model = MLPModel(input_dim) 
elif model_name == "deep_mlp_model":
    model = MLPModel_deep(input_dim)
elif model_name == "mlp_model_residual_connections":
    model = MLPModel_residual(input_dim)
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

criterion = nn.L1Loss()

train_losses = []
test_losses = []

for epoch in range(epochs):
    print("###############")
    print(f"epoch: {epoch}")
    # Training
    model.train()
    train_target_list = []
    train_output_list = []
    e_1 = 0
    e_2 = 0
    e_3 = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data[0][-3] = e_1
        data[0][-2] = e_2
        data[0][-1] = e_3

        batch_size = data.size()[0]
        data, target = data.to(device), target.to(device)
        # target_onehot = onehot_coding(target, device, output_dim)

        output = model.forward(data)

        e_3 = e_2
        e_2 = e_1
        e_1 = target.item() - output.item()

        train_output_list.append(output.cpu().detach().numpy())
        train_target_list.append(target.cpu().detach().numpy())

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_target_list = np.array(train_target_list).ravel()
    train_output_list = np.array(train_output_list).ravel()

    train_target_list = scaler.inverse_transform(train_target_list.reshape(-1, 1))
    train_output_list = scaler.inverse_transform(train_output_list.reshape(-1, 1))

    train_mse = mse(train_target_list, train_output_list)
    train_mae = mae(train_target_list, train_output_list)
    train_mape = mape(train_target_list, train_output_list)
    print(f"traim mse {train_mse}")
    print(f"train mae {train_mae}")
    print(f"train mape {train_mape}")
    print("--------")

    train_losses.append((train_mse, train_mae, train_mape))

    # Evaluating
    model.eval()
    correct = 0.0
    output_list = []
    target_list = []

    for batch_idx, (data, target) in enumerate(test_loader):
        data[0][-3] = e_1
        data[0][-2] = e_2
        data[0][-1] = e_3

        batch_size = data.size()[0]
        data, target = data.to(device), target.to(device)

        output = model.forward(data)

        e_3 = e_2
        e_2 = e_1
        e_1 = target.item() - output.item()

        output_list.append(output.cpu().detach().numpy())
        target_list.append(target.cpu().detach().numpy())

    target_list = np.array(target_list).ravel()
    output_list = np.array(output_list).ravel()

    target_list = scaler.inverse_transform(target_list.reshape(-1, 1))
    output_list = scaler.inverse_transform(output_list.reshape(-1, 1))

    test_mse = mse(target_list, output_list)
    test_mae = mae(target_list, output_list)
    test_mape = mape(target_list, output_list)
    print(f"test mse {test_mse}")
    print(f"test mae {test_mae}")
    print(f"test mape {test_mape}")

    test_losses.append((test_mse, test_mae, test_mape))

    print("###############")

error = (target_list - output_list) ** 2
cum = np.cumsum(error) / (1 + np.arange(len(error)))

# print(str(cum))

plt.plot(target_list)
plt.plot(output_list)
plt.savefig(f"{result_folder}/predictions_{exp_name}.png")
plt.cla()
f = open(f"{result_folder}/{result_txt_file}", "a+")
f.write("########\n")
f.write(f"Data Type: {exp_name}\n")

f.write("Train Losses")
f.write(str(train_losses) + "\n\n")

f.write("Test Losses")
f.write(str(test_losses) + "\n\n")

f.write("Cum Losses" + "\n\n")
f.write((str(cum)))

f.write("########\n")
f.close()

test_losses = [l[0] for l in test_losses]
train_losses = [l[0] for l in train_losses]

plt.plot(test_losses, label="test")
plt.plot(train_losses, label="train")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE Losses")
plt.savefig(f"{result_folder}/MLP_result_{exp_name}.png")
plt.cla()


# # MLP AR

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import pandas as pd
import numpy as np

# from SDT import SDT
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import random
from torch.utils.data import TensorDataset, DataLoader
import os
import matplotlib.pyplot as plt
from SDT import MLPModel, MLPModel_deep, MLPModel_residual


def mape(y_test, pred):
    mape = np.mean(np.abs((y_test - pred) / y_test))
    return mape




    

exp_name = "experiment"



model_type = "mlp_model" # mlp_model, deep_mlp_model, #mlp_model_residual_connections

lr = 1e-2  # learning rate
epochs = 30 # the number of training epochs

if not os.path.exists("Results"):
    os.makedirs("Results")

result_folder = f"Results/sdt_arm_{exp_name}"
result_txt_file = f"sdt_arma_{exp_name}.txt"

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# Parameters
# the number of input dimensions
output_dim = 1  # the number of outputs (i.e., # classes on MNIST)
batch_size = 1  # batch size
use_cuda = torch.cuda.is_available()  # whether to use GPU

device = torch.device("cuda" if use_cuda else "cpu")

# reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Load data

##########################

mu, sigma = 0, 0.1

df = pd.read_csv(data_path)
df = df.drop("Unnamed: 0", axis=1)
if date_column_name is not None:
    df = df.drop(date_column_name, axis=1)
col_list = list(df.columns)
col_list.remove(label_name)
col_list.insert(0, label_name)

df = df[col_list]

y = df.loc[:, label_name]
X = df.iloc[:, 1:]
data_len = len(X)

forecast_horizon = int(data_len * test_size)
e_t = np.random.normal(mu, sigma, len(X))

X["e"] = e_t
X["e-1"] = 0
X["e-2"] = 0
X["e-3"] = 0

X_train, X_test = X.iloc[:-forecast_horizon].to_numpy().astype(np.float32), X.iloc[
    -forecast_horizon:
].to_numpy().astype(np.float32)
y_train, y_test = y.iloc[:-forecast_horizon].to_numpy().astype(np.float32), y.iloc[
    -forecast_horizon:
].to_numpy().astype(np.float32)

scaler = MinMaxScaler()
X_train_arr = scaler.fit_transform(X_train)
X_test_arr = scaler.transform(X_test)

y_train_arr = scaler.fit_transform(y_train.reshape(-1, 1))
y_test_arr = scaler.transform(y_test.reshape(-1, 1))

train_features = torch.Tensor(X_train_arr).to(device)
train_targets = torch.Tensor(y_train_arr).to(device)
test_features = torch.Tensor(X_test_arr).to(device)
test_targets = torch.Tensor(y_test_arr).to(device)

train = TensorDataset(train_features, train_targets)
test = TensorDataset(test_features, test_targets)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

input_dim = train_features.shape[1]
print(f"input_dim: {input_dim}")
##########################

# Model and Optimizer

if model_name == "mlp_model":
    model = MLPModel(input_dim) 
elif model_name == "deep_mlp_model":
    model = MLPModel_deep(input_dim)
elif model_name == "mlp_model_residual_connections":
    model = MLPModel_residual(input_dim)
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

criterion = nn.L1Loss()

train_losses = []
test_losses = []

for epoch in range(epochs):
    print("###############")
    print(f"epoch: {epoch}")
    # Training
    model.train()
    train_target_list = []
    train_output_list = []

    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size = data.size()[0]
        data, target = data.to(device), target.to(device)
        # target_onehot = onehot_coding(target, device, output_dim)

        output = model.forward(data)

        train_output_list.append(output.cpu().detach().numpy())
        train_target_list.append(target.cpu().detach().numpy())

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_target_list = np.array(train_target_list).ravel()
    train_output_list = np.array(train_output_list).ravel()

    train_target_list = scaler.inverse_transform(train_target_list.reshape(-1, 1))
    train_output_list = scaler.inverse_transform(train_output_list.reshape(-1, 1))

    train_mse = mse(train_target_list, train_output_list)
    train_mae = mae(train_target_list, train_output_list)
    train_mape = mape(train_target_list, train_output_list)
    print(f"traim mse {train_mse}")
    print(f"train mae {train_mae}")
    print(f"train mape {train_mape}")
    print("--------")

    train_losses.append((train_mse, train_mae, train_mape))

    # Print training status
    # if batch_idx % log_interval == 0:
    #     pred = output.data.max(1)[1]
    #     correct = pred.eq(target.view(-1).data).sum()

    # msg = "Epoch: {:02d} | Batch: {:03d} | Loss: {:.5f} |" " Correct: {:03d}/{:03d}"
    # print(msg.format(epoch, batch_idx, loss, correct, batch_size))
    # training_loss_list.append(loss.cpu().data.numpy())

    # Evaluating
    model.eval()
    correct = 0.0
    output_list = []
    target_list = []

    for batch_idx, (data, target) in enumerate(test_loader):
        batch_size = data.size()[0]
        data, target = data.to(device), target.to(device)

        output = model.forward(data)

        output_list.append(output.cpu().detach().numpy())
        target_list.append(target.cpu().detach().numpy())

    target_list = np.array(target_list).ravel()
    output_list = np.array(output_list).ravel()

    target_list = scaler.inverse_transform(target_list.reshape(-1, 1))
    output_list = scaler.inverse_transform(output_list.reshape(-1, 1))

    test_mse = mse(target_list, output_list)
    test_mae = mae(target_list, output_list)
    test_mape = mape(target_list, output_list)
    print(f"test mse {test_mse}")
    print(f"test mae {test_mae}")
    print(f"test mape {test_mape}")

    test_losses.append((test_mse, test_mae, test_mape))

    print("###############")

error = (target_list - output_list) ** 2
cum = np.cumsum(error) / (1 + np.arange(len(error)))

error_train = (target_list - output_list) ** 2
cum_train = np.cumsum(error_train) / (1 + np.arange(len(error_train)))

# print(str(cum_train))

plt.plot(target_list)
plt.plot(output_list)
plt.savefig(f"{result_folder}/predictions_{exp_name}.png")
plt.cla()
f = open(f"{result_folder}/{result_txt_file}", "a+")
f.write("########\n")
f.write(f"M4 index: {exp_name}\n")

f.write("Train Losses")
f.write(str(train_losses) + "\n\n")

f.write("Test Losses")
f.write(str(test_losses) + "\n\n")

f.write("Cum Losses" + "\n\n")
f.write((str(cum)))

f.write("########\n")
f.close()

test_losses = [l[0] for l in test_losses]
train_losses = [l[0] for l in train_losses]

plt.plot(test_losses, label="test")
plt.plot(train_losses, label="train")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE Losses")
plt.savefig(f"{result_folder}/MLP_result_{exp_name}.png")
plt.cla()


# # SOFT AR

# In[ ]:


import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from SDT import SDT
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import random
from torch.utils.data import TensorDataset, DataLoader
import os
import matplotlib.pyplot as plt
import argparse


def mape(y_test, pred):
    mape = np.mean(np.abs((y_test - pred) / y_test))
    return mape


if __name__ == "__main__":

    

    exp_name = "soft_ar_experiment"
    data_path = data_path
    label_name = label_name
    test_size = test_size
    date_column_name = date_column_name

    depth = 3  # tree depth
    lamda = 1e-3  # coefficient of the regularization term
    lr = 1e-2   # learning rate
    epochs = 30  # the number of training epochs

    if not os.path.exists("Results"):
        os.makedirs("Results")

    result_folder = f"Results/sdt_arm_{exp_name}"
    result_txt_file = f"sdt_arma_{exp_name}.txt"

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Parameters
    # the number of input dimensions
    output_dim = 1  # the number of outputs (i.e., # classes on MNIST)
    batch_size = 1  # batch size
    use_cuda = torch.cuda.is_available()  # whether to use GPU

    device = torch.device("cuda" if use_cuda else "cpu")

    # reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # Load data

    ##########################

    mu, sigma = 0, 0.1

    df = pd.read_csv(data_path)
    df = df.drop("Unnamed: 0", axis=1)
    if date_column_name is not None:
        df = df.drop(date_column_name, axis=1)
    col_list = list(df.columns)
    col_list.remove(label_name)
    col_list.insert(0, label_name)

    df = df[col_list]

    y = df.loc[:, label_name]
    X = df.iloc[:, 1:]
    data_len = len(X)

    forecast_horizon = int(data_len * test_size)
    e_t = np.random.normal(mu, sigma, len(X))

    X["e"] = e_t
    X["e-1"] = 0
    X["e-2"] = 0
    X["e-3"] = 0

    X_train, X_test = X.iloc[:-forecast_horizon].to_numpy().astype(np.float32), X.iloc[
        -forecast_horizon:
    ].to_numpy().astype(np.float32)
    y_train, y_test = y.iloc[:-forecast_horizon].to_numpy().astype(np.float32), y.iloc[
        -forecast_horizon:
    ].to_numpy().astype(np.float32)

    scaler = MinMaxScaler()
    X_train_arr = scaler.fit_transform(X_train)
    X_test_arr = scaler.transform(X_test)

    y_train_arr = scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_arr = scaler.transform(y_test.reshape(-1, 1))

    train_features = torch.Tensor(X_train_arr).to(device)
    train_targets = torch.Tensor(y_train_arr).to(device)
    test_features = torch.Tensor(X_test_arr).to(device)
    test_targets = torch.Tensor(y_test_arr).to(device)

    train = TensorDataset(train_features, train_targets)
    test = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    input_dim = train_features.shape[1]
    print(f"input_dim: {input_dim}")
    ##########################

    # Model and Optimizer
    tree = SDT(input_dim, output_dim, depth, lamda, use_cuda)
    tree = tree.to(device)

    optimizer = torch.optim.SGD(tree.parameters(), lr=lr)

    criterion = nn.L1Loss()

    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        time_start = time.time()
        print("###############")
        print(f"epoch: {epoch}")
        # Training
        tree.train()
        train_target_list = []
        train_output_list = []

        for batch_idx, (data, target) in enumerate(train_loader):
            batch_size = data.size()[0]
            data, target = data.to(device), target.to(device)
            # target_onehot = onehot_coding(target, device, output_dim)

            output, penalty = tree.forward(data, is_training_data=True)

            train_output_list.append(output.cpu().detach().numpy())
            train_target_list.append(target.cpu().detach().numpy())

            loss = criterion(output, target)
            loss += penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_target_list = np.array(train_target_list).ravel()
        train_output_list = np.array(train_output_list).ravel()
        time_end = time.time()
        print(f"Train Time {time_end-time_start}")
        train_target_list = scaler.inverse_transform(train_target_list.reshape(-1, 1))
        train_output_list = scaler.inverse_transform(train_output_list.reshape(-1, 1))

        train_mse = mse(train_target_list, train_output_list)
        train_mae = mae(train_target_list, train_output_list)
        train_mape = mape(train_target_list, train_output_list)
        print(f"traim mse {train_mse}")
        print(f"train mae {train_mae}")
        print(f"train mape {train_mape}")
        print("--------")

        train_losses.append((train_mse, train_mae, train_mape))
    

        # Evaluating
        tree.eval()
        correct = 0.0
        output_list = []
        target_list = []

        for batch_idx, (data, target) in enumerate(test_loader):
            time_start_test = time.time()
            batch_size = data.size()[0]
            data, target = data.to(device), target.to(device)

            output = tree.forward(data)

            output_list.append(output.cpu().detach().numpy())
            target_list.append(target.cpu().detach().numpy())
            time_end_test  =time.time()
        target_list = np.array(target_list).ravel()
        output_list = np.array(output_list).ravel()

        target_list = scaler.inverse_transform(target_list.reshape(-1, 1))
        output_list = scaler.inverse_transform(output_list.reshape(-1, 1))

        test_mse = mse(target_list, output_list)
        test_mae = mae(target_list, output_list)
        test_mape = mape(target_list, output_list)
        print(f"test mse {test_mse}")
        print(f"test mae {test_mae}")
        print(f"test mape {test_mape}")

        test_losses.append((test_mse, test_mae, test_mape))

        print("###############")

    error = (target_list - output_list) ** 2
    cum = np.cumsum(error) / (1 + np.arange(len(error)))

    plt.plot(target_list)
    plt.plot(output_list)
    plt.savefig(f"{result_folder}/predictions_{exp_name}.png")
    plt.cla()
    f = open(f"{result_folder}/{result_txt_file}", "a+")
    f.write("########\n")
    f.write(f"M4 index: {exp_name}\n")

    f.write("Train Losses")
    f.write(str(train_losses) + "\n\n")

    f.write("Test Losses")
    f.write(str(test_losses) + "\n\n")

    f.write("Cum Losses" + "\n\n")
    f.write((str(cum)))

    f.write("########\n")
    f.close()

    test_losses = [l[0] for l in test_losses]
    train_losses = [l[0] for l in train_losses]

    plt.plot(test_losses, label="test")
    plt.plot(train_losses, label="train")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE Losses")
    plt.savefig(f"{result_folder}/SDT_result_{exp_name}.png")
    plt.cla()

    

