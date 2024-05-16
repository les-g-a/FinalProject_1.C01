import pandas as pd
import numpy as np
import argparse
import os
import datetime
import time

from epftoolbox.data import read_data
from epftoolbox.evaluation import MAE, sMAPE
from epftoolbox.models import DNN

# ------------------------------ EXTERNAL PARAMETERS ------------------------------------#
nlayers = 2
dataset = "ERCOT_7"
years_test = 1
shuffle_train = 1
data_augmentation = 0
new_recalibration = 1
calibration_window = 4
experiment_id = 1
begin_test_date = "01/01/2023 00:00"
end_test_date = "31/12/2023 23:00"
hyperparameter_evals = 10

## Get the current directory
current_directory = os.getcwd()

# Define paths for datasets and inputs/outputs
path_datasets_folder = os.path.join(current_directory, 'examples', 'datasets')
path_recalibration_folder = os.path.join(current_directory, 'examples', 'experimental_files')
path_hyperparameter_folder = os.path.join(current_directory, 'examples', 'experimental_files')

# Defining train and testing data
df_train, df_test = read_data(dataset=dataset, years_test=years_test, path=path_datasets_folder,
                              begin_test_date=begin_test_date, end_test_date=end_test_date)

# Defining unique name to save the forecast
forecast_file_name = 'DNN_' + str(nlayers) + '_layers' + \
                    '_SF' + str(shuffle_train) + \
                   '_' + str(experiment_id) + '.csv'

forecast_file_path = os.path.join(path_recalibration_folder, forecast_file_name)


# Defining empty forecast array and the real values to be predicted in a more friendly format
forecast = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])
real_values = df_test.loc[:, ['Price']].values.reshape(-1, 24)
real_values = pd.DataFrame(real_values, index=forecast.index, columns=forecast.columns)

# If we are not starting a new recalibration but re-starting an old one, we import the
# existing files and print metrics 
if not new_recalibration:
    # Import existinf forecasting file
    forecast = pd.read_csv(forecast_file_path, index_col=0)
    forecast.index = pd.to_datetime(forecast.index)

    # Reading dates to still be forecasted by checking NaN values
    forecast_dates = forecast[forecast.isna().any(axis=1)].index

    # If all the dates to be forecasted have already been forecast, we print information
    # and exit the script
    if len(forecast_dates) == 0:

        mae = np.mean(MAE(forecast.values.squeeze(), real_values.values))
        smape = np.mean(sMAPE(forecast.values.squeeze(), real_values.values)) * 100
        print('{} - sMAPE: {:.2f}%  |  MAE: {:.3f}'.format('Final metrics', smape, mae))

else:
    forecast_dates = forecast.index

## Optimize hyperparameters
from epftoolbox.models import _dnn_hyperopt

## Start counting time
start_time = time.time()

# # Hyperparameter optimizer
_dnn_hyperopt.hyperparameter_optimizer(path_datasets_folder=path_datasets_folder,
                                       path_hyperparameters_folder=path_hyperparameter_folder,
                                       new_hyperopt=1,max_evals=hyperparameter_evals,
                                       nlayers=nlayers,dataset=dataset,years_test=1,
                                       calibration_window=calibration_window,
                                       shuffle_train=shuffle_train,data_augmentation=data_augmentation,
                                       experiment_id=experiment_id,
                                       begin_test_date=begin_test_date,
                                       end_test_date=end_test_date)


model = DNN(
    experiment_id=experiment_id, path_hyperparameter_folder=path_hyperparameter_folder, nlayers=nlayers, 
    dataset=dataset, years_test=years_test, shuffle_train=shuffle_train, data_augmentation=data_augmentation,
    calibration_window=calibration_window)

## Create a dataframe for the errors to be stored for each date
error_df = pd.DataFrame(columns=['Date', 'sMAPE', 'MAE'])

# For loop over the recalibration dates
for date in forecast_dates:

    # For simulation purposes, we assume that the available data is
    # the data up to current date where the prices of current date are not known
    data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)

    # We extract real prices for current date and set them to NaN in the dataframe of available data
    data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'] = np.NaN

    # Recalibrating the model with the most up-to-date available data and making a prediction
    # for the next day
    Yp = model.recalibrate_and_forecast_next_day(df=data_available, next_day_date=date)

    # Saving the current prediction
    forecast.loc[date, :] = Yp

    # Computing metrics up-to-current-date
    mae = np.mean(MAE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) 
    smape = np.mean(sMAPE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) * 100
    
    # Change sMAPE and MAE into this - {} - sMAPE: {:.2f}%  |  MAE: {:.3f}'
    error_df.loc[len(error_df)] = [date, smape, mae]
    # Pringint information
    # print('{} - sMAPE: {:.2f}%  |  MAE: {:.3f}'.format(str(date)[:10], smape, mae))

    # Saving forecast
    forecast.to_csv(forecast_file_path)

## Get end time
end_time = time.time()

# Get runtime
runtime = end_time - start_time

# Print out runtime
print(f"The runtime of a DNN model with shuffle of {shuffle_train} and {hyperparameter_evals} hyp evals is: {runtime}")
    
# Get the current datetime
current_datetime = datetime.datetime.now()

# Save the error dataframe
error_df.to_csv(os.path.join(path_recalibration_folder, f"DNN_Error_run_{shuffle_train}_{nlayers}_{current_datetime.strftime('%Y-%m-%d_%H-%M-%S')}.csv"), index=False)
