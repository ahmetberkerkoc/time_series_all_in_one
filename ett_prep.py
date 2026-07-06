import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import random


reproduciblity = True
if reproduciblity:
    random.seed(10)
    
    
import numpy as np
import pandas as pd


def feature_extractor_ETT(data, time_resolution):
    """
    Feature extraction for ETT competition datasets.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing at least column ['y'] and optionally a ['date'] column.
    time_resolution : str
        One of: "h1", "h2", "m1", "m2", "hourly", "minute"

    Returns
    -------
    pd.DataFrame
        DataFrame with lag, rolling, and calendar features.
    """

    data = data.copy()
    resolution = str(time_resolution).lower()

    timestamp_values = None
    if "date" in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        timestamp_values = data.loc[:, "date"]
    elif isinstance(data.index, pd.DatetimeIndex):
        timestamp_values = pd.Series(data.index, index=data.index)

    # Basic autoregressive lags
    data.loc[:, "y-1"] = data.loc[:, "y"].shift(1)
    data.loc[:, "y-2"] = data.loc[:, "y"].shift(2)
    data.loc[:, "y-3"] = data.loc[:, "y"].shift(3)

    if resolution in {"h1", "h2", "hourly", "hour"}:
        # Hourly ETT data shows strong daily and weekly seasonality.
        data.loc[:, "y-24"] = data.loc[:, "y"].shift(24)
        data.loc[:, "y-48"] = data.loc[:, "y"].shift(48)
        data.loc[:, "y-168"] = data.loc[:, "y"].shift(168)

        for window in (3, 6, 12, 24, 168):
            data.loc[:, f"y-1_rolling_mean_{window}"] = data.loc[:, "y-1"].rolling(window).mean()
            data.loc[:, f"y-1_rolling_std_{window}"] = data.loc[:, "y-1"].rolling(window).std()

        for window in (24, 168):
            data.loc[:, f"y-24_rolling_mean_{window}"] = data.loc[:, "y-24"].rolling(window).mean()
            data.loc[:, f"y-24_rolling_std_{window}"] = data.loc[:, "y-24"].rolling(window).std()

        if timestamp_values is not None:
            data.loc[:, "hour"] = timestamp_values.dt.hour
            data.loc[:, "day_of_week"] = timestamp_values.dt.dayofweek
            data.loc[:, "day_of_month"] = timestamp_values.dt.day
            data.loc[:, "day_of_year"] = timestamp_values.dt.dayofyear
            data.loc[:, "month"] = timestamp_values.dt.month

    elif resolution in {"m1", "m2", "minute", "minutely"}:
        # Minute-level ETT data is sampled every 15 minutes.
        data.loc[:, "y-4"] = data.loc[:, "y"].shift(4)
        data.loc[:, "y-8"] = data.loc[:, "y"].shift(8)
        data.loc[:, "y-96"] = data.loc[:, "y"].shift(96)
        data.loc[:, "y-192"] = data.loc[:, "y"].shift(192)
        data.loc[:, "y-672"] = data.loc[:, "y"].shift(672)

        for window in (4, 8, 12, 24, 96, 672):
            data.loc[:, f"y-1_rolling_mean_{window}"] = data.loc[:, "y-1"].rolling(window).mean()
            data.loc[:, f"y-1_rolling_std_{window}"] = data.loc[:, "y-1"].rolling(window).std()

        for window in (4, 96):
            data.loc[:, f"y-96_rolling_mean_{window}"] = data.loc[:, "y-96"].rolling(window).mean()
            data.loc[:, f"y-96_rolling_std_{window}"] = data.loc[:, "y-96"].rolling(window).std()

        if timestamp_values is not None:
            data.loc[:, "minute"] = timestamp_values.dt.minute
            data.loc[:, "hour"] = timestamp_values.dt.hour
            data.loc[:, "day_of_week"] = timestamp_values.dt.dayofweek
            data.loc[:, "day_of_month"] = timestamp_values.dt.day
            data.loc[:, "day_of_year"] = timestamp_values.dt.dayofyear
            data.loc[:, "month"] = timestamp_values.dt.month

    else:
        raise ValueError(
            "time_resolution must be one of: 'h1', 'h2', 'm1', 'm2', 'hourly', 'minute'"
        )

    # Remove rows with missing values caused by shift and rolling operations
    data = data.dropna()

    # Reset index like your ETT code
    data.index = np.arange(len(data))

    return data



if __name__ == "__main__":
    
    ETTs = {"h1": 10, "h2": 10, "m1": 10, "m2": 10}
    for i in range(4):
        ETT_TYPE = list(ETTs.keys())[i]  # Change this to "quarterly" or "yearly" as needed
        df = pd.read_csv(f"data/ETT/ETT{ETT_TYPE}.csv")
        os.makedirs(f"data/ETT/Extracted/{ETT_TYPE}/", exist_ok=True)
        df.rename(columns={"OT": "y"}, inplace=True)
        df = feature_extractor_ETT(df, ETT_TYPE)
        df.to_csv(f"data/ETT/Extracted/{ETT_TYPE}/ETT{ETT_TYPE}_extracted.csv", index=False)