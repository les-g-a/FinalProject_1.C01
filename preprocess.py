import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dropout, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.regularizers import l2
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator
import datetime as dt
from datetime import timedelta
# Check current directory
import os
print(os.getcwd())


# Separate function to create a DT object that is 00:15, :30, :45 or 1:00 for HE 1
# Separated out because the imported load forecasts already have a DT index, incorporated into method below

def interval_fix(intervaldt, interval):
    return intervaldt + pd.Timedelta(minutes=-15*(4-interval))

def create_time_intervals(date, hour, interval):
        dt_date = pd.to_datetime(date)
        if hour == "24:00":
            hour = "00:00"
            dt_date = dt_date + timedelta(days=1)
        hour_datetime = pd.to_datetime(hour, format='%H:%M')
        interval_time = interval_fix(dt_date,interval) + timedelta(hours=hour_datetime.hour)
        return interval_time

#def load_merge_DAMRTM_forecasts():

# Load DAM and RTM .csv
DAM = pd.read_csv('Data_Clean/DAM.csv')
RTM = pd.read_csv('Data_Clean/RTM.csv')
loadfc_df = pd.read_pickle('Data_Clean/merged_fcload.pkl')

# From RTM take Delivert Date Delivery Hour Delivery Interval and Settlement Point Price columns
RTM = RTM[['Delivery Date', 'Delivery Hour', 'Delivery Interval', 'Settlement Point Price']]

# From DAM take Delivery Date Hour Ending and Settlement Point Price columns
DAM = DAM[['Delivery Date', 'Hour Ending', 'Settlement Point Price']]

loadfc_df = loadfc_df[['interval_end_utc', 'houston_load', 'houston_diff']]
# next - add the one day timedelta below and set up the DFs to merge on datetime after expanding the DAM dataset


def expand_hourly(df):
    dam_intervals = pd.DataFrame({"Delivery Interval": range(1, 5)})
    df['key'] = 1
    dam_intervals['key'] = 1
    df = pd.merge(df, dam_intervals, on='key').drop('key', axis=1)
    return df

DAM = expand_hourly(DAM)
loadfc_df = expand_hourly(loadfc_df)

# Apply only the interval_fix to the existing DT index
loadfc_df['interval_end_utc'] = loadfc_df.apply(lambda x: interval_fix(x['interval_end_utc'], x['Delivery Interval']), axis=1)



# Create DT index
DAM['dt'] = DAM.apply(lambda x: create_time_intervals(x['Delivery Date'], x['Hour Ending'], x['Delivery Interval']), axis=1)

# Prepare the 'Hour Ending' column for RTM to format hours correctly and create DT index
RTM['Hour Ending'] = RTM['Delivery Hour'].astype(int).apply(lambda x: f"{x:02d}:00")
RTM['dt'] = RTM.apply(lambda x: create_time_intervals(x['Delivery Date'], x['Hour Ending'], x['Delivery Interval']), axis=1)


def prep_one_hot(df):
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day_of_week
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    return df

# Combine DAM and RTM on Interval Time.
df = pd.merge(DAM, RTM, on=['dt'], how = 'outer', suffixes=('_DAM', '_RTM'))
loadfc_df['interval_end_utc'] = loadfc_df['interval_end_utc'].dt.tz_localize(None)
df = pd.merge(df, loadfc_df, left_on='dt', right_on='interval_end_utc', how='outer')
df.set_index('dt', inplace=True, drop = True)
prep_one_hot(df)
#  Only keep Interval Time Delivery Date and Settlement Point Price, day, month, year
df['houston_diff'] = df['houston_diff'].fillna(0)
df['houston_load'] = df['houston_load'].fillna(0)
#df = df[['Delivery Hour', 'Settlement Point Price_DAM', 'Settlement Point Price_RTM', 'Day_DAM', 'Month_DAM', 'Year_DAM', 'Day_RTM', 'Month_RTM', 'Year_RTM']]

df = df[['Settlement Point Price_DAM', 'Settlement Point Price_RTM', 'houston_load', 'houston_diff', 'Day', 'Month', 'Year', 'Hour', 'Minute']]

df.rename(columns={'Settlement Point Price_DAM': 'DAM', 'Settlement Point Price_RTM': 'RTM', 'houston_load' : 'load_fc', 'houston_diff' : 'load_diff'}, inplace=True)

df.to_pickle('Data_Clean/merged_clean_df.pkl')
