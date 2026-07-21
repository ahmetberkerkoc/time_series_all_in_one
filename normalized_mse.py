import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_percentage_error as mape,
    mean_squared_error as mse,
    mean_absolute_error as mae,
)
from sklearn.preprocessing import MinMaxScaler

data_type = "m3_yearly"
#data_names = ['monthly_M838.csv', 'monthly_M1257.csv', 'monthly_M1149.csv', 'monthly_M537.csv', 'monthly_M587.csv', 'monthly_M788.csv', 'monthly_M304.csv', 'monthly_M1292.csv', 'monthly_M1323.csv', 'monthly_M385.csv']
# data_names = [
#     "monthly_M583.csv",
#     "monthly_M927.csv",
#     "monthly_M442.csv",
#     "monthly_M1061.csv",
#     "monthly_M1370.csv",
#     "monthly_M748.csv",
#     "monthly_M275.csv",
#     "monthly_M1337.csv",
#     "monthly_M238.csv",
#     "monthly_M310.csv"
# ]
data_names = [
#    "quarterly_Q270.csv",
#     "quarterly_Q43.csv",
#     "quarterly_Q470.csv",
#     "quarterly_Q732.csv",
#     "quarterly_Q436.csv",
#     "quarterly_Q697.csv",
#     "quarterly_Q281.csv",
#     "quarterly_Q273.csv",
#     "quarterly_Q79.csv",
#     "quarterly_Q697.csv"
]


data_names = [
    #"yearly_Y194.csv",
    # "yearly_Y216.csv",
    # "yearly_Y247.csv",
    # "yearly_Y369.csv",
    # "yearly_Y411.csv",
     "yearly_Y374.csv",
    # "yearly_Y199.csv",
    # "yearly_Y547.csv",
    # "yearly_Y365.csv",
    # "yearly_Y364.csv",
    # "yearly_Y238.csv"
]
models = ["deepar", "informer", "itransformer", "nbeats", "nhits", "patchtst"]
TARGET_DICT = {}

for model in models:
    normalized_mse_list = []
    average_mape_list = {}
    for data_name in data_names:
        data = pd.read_pickle(f"ALL_RESULT_{model}_{data_type}_normalized_mse.pkl")
        
        target_lists = data["series_results"][data_name]["targets"]
        prediction_list = data["series_results"][data_name]["predictions"]
        if model not in average_mape_list.keys():
            average_mape_list[model] = data["series_results"][data_name]["metrics"]["mape"]/10
        else:
            average_mape_list[model] += data["series_results"][data_name]["metrics"]["mape"]/10

        target_lists = np.asarray(target_lists).reshape(-1, 1)
        prediction_list = np.asarray(prediction_list).reshape(-1, 1)

        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(target_lists)
        TARGET_DICT[data_name] = abs((max(target_lists) - min(target_lists))).item()
        pred_scaled = scaler.transform(prediction_list)
        normalized_mse = mae(y_scaled, pred_scaled)
        normalized_mse_list.append(normalized_mse)
    #print(len(normalized_mse_list))
    print(f"Model: {model}")
    print(sum(normalized_mse_list)/len(normalized_mse_list))
    
    #print("Mape")
    #print(average_mape_list[model])
    
our_data = pd.read_pickle(f"ALL_RESULT_{data_type}.pkl")
filtered_data = {}
for i in our_data.keys():
    if i in data_names:
        filtered_data[i] = our_data[i]
        



average_mape = {}
average_mse = {}
for data_name in filtered_data.keys():
    for key, value in filtered_data[data_name].items():
        if key not in average_mape.keys():
            average_mape[key] = value[2]
            average_mse[key] = value[1]/TARGET_DICT[data_name]
        else:
            average_mape[key] += value[2]
            average_mse[key] += value[1]/TARGET_DICT[data_name]

for key in average_mape.keys():
    print(f"Model: {key}")
    #print(f"{average_mape[key] / len(filtered_data)}")
    print(f"{average_mse[key] / len(filtered_data)}")