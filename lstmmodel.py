
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
from sklearn.model_selection import TimeSeriesSplit


# Check current directory
import os
print(os.getcwd())

import dill
# load the DF from pickle to speed pre-processing
def preprocess(chop, lookback, forecast, scaler, n_splits=3, val_size=0.2, test_size=0.1, name = None):
    pkl_name = "merged_clean_nonzero"
    
    df = pd.read_pickle(f'Data_Clean/{pkl_name}.pkl')

    ###########
    # Set params
    ###########

    # set to >1 to chop off 1/n of the data
    # chop = 1
    # lookback = 96
    # forecast = 96
    # epochs = 8
    # test_size = 0.2

    # set to 'minmax' or 'standard'. A dict after the functions are defined routes the function call


     ###########
    ###########

    # Shorten dataset for quicker testing
    df = df.iloc[-int(len(df)/chop):]
    sequence_length = lookback + forecast

    # Lag all variables that are what we're trying to predict in the forecast period by the forecast period
    df['RTMlag'] = df['RTM'].shift(forecast)
    df['load_diff'] = df['load_diff'].shift(forecast)
    df.dropna(inplace=True)
    X_vars = list(df.columns)
    X_vars.remove('RTM')
    X_vars


    # Continuous features
    X_continuous = df[['DAM', 'RTMlag', 'load_fc', 'load_diff']].values
    # Categorical features
    enc = OneHotEncoder(handle_unknown='ignore')
    X_categorical = df[['Day', 'Month', 'Hour']]
    X_categorical = enc.fit_transform(X_categorical).toarray()

    # print(df.shape)
    # print(X_continuous.shape)
    # print(X_categorical.shape)
    # print(df['RTM'].shape)

    import gc
    gc.collect()
    
    #### We are splitting the data into test/train before we create the split sequences to input into
    #### LSTM. We have already onehotencoded the data and have lagged it
    
    def non_seq_split(X_continuous, X_categorical, Y, test_size=0.2, val_size=0.1):
        X_cont_train = X_continuous[:int(len(X_continuous) * (1 - test_size))]
        X_cont_test = X_continuous[int(len(X_continuous) * (1 - test_size)):]
        X_cat_train = X_categorical[:int(len(X_categorical) * (1 - test_size))]
        X_cat_test = X_categorical[int(len(X_categorical) * (1 - test_size)):]
        Y_train = Y[:int(len(Y) * (1 - test_size))]
        Y_test = Y[int(len(Y) * (1 - test_size)):]
        return X_cont_train, X_cat_train, X_cont_test, X_cat_test, Y_train, Y_test
        
        
    
    def time_series_cv(X_continuous, X_categorical, Y, n_splits=3, val_fraction=0.1):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        
        for train_idx, test_idx in tscv.split(X_continuous):
            # Assuming that the test set size can be used to define the validation set size
            val_size = int(len(train_idx) * val_fraction)
            train_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]

            # Extract continuous features for training, validation, and testing
            X_cont_train, X_cont_val, X_cont_test = X_continuous[train_idx], X_continuous[val_idx], X_continuous[test_idx]
            
            # Extract categorical features for training, validation, and testing
            X_cat_train, X_cat_val, X_cat_test = X_categorical[train_idx], X_categorical[val_idx], X_categorical[test_idx]
            
            # Split target variable
            Y_train, Y_val, Y_test = Y[train_idx], Y[val_idx], Y[test_idx]
            
            splits.append({
                'X_cont_train': X_cont_train, 'X_cont_val': X_cont_val, 'X_cont_test': X_cont_test,
                'X_cat_train': X_cat_train, 'X_cat_val': X_cat_val, 'X_cat_test': X_cat_test,
                'Y_train': Y_train, 'Y_val': Y_val, 'Y_test': Y_test
            })

        return splits


    
    

    def trim(arr, size):
        if isinstance(arr, tuple):
            return tuple(a[size:] for a in arr)
        else:
            return arr[size:]
    # Non time cross validation splot
    #X_cont_train, X_cat_train, X_cont_test, X_cat_test, y_train, y_test = trim(non_seq_split(X_continuous, X_categorical, df['RTM'].values, test_size=test_size, val_size=val_size),lookback)

    def trim(data, lookback):
        # Assuming data is a dictionary with train, validation, and test datasets
        trimmed_data = {
            'X_cont_train': data['X_cont_train'][lookback:],
            'X_cat_train': data['X_cat_train'][lookback:],
            'X_cont_val': data['X_cont_val'][lookback:],  # Adding trimming for validation data
            'X_cat_val': data['X_cat_val'][lookback:],    # Adding trimming for validation data
            'X_cont_test': data['X_cont_test'][lookback:],
            'X_cat_test': data['X_cat_test'][lookback:],
            'Y_train': data['Y_train'][lookback:],
            'Y_val': data['Y_val'][lookback:],            # Adding trimming for validation data
            'Y_test': data['Y_test'][lookback:]
        }
        return trimmed_data


    # commented without trimmed version
    #X_cont_train, X_cat_train, X_cont_test, X_cat_test, y_train, y_test = non_seq_split(X_continuous, X_categorical, df['Settlement Point Price_RTM'].values)


    def non_seq_scaler_minmax(X_cont_train, X_cont_val, X_cont_test, Y_train, Y_val, Y_test):
        Xscaler = MinMaxScaler()
        X_cont_train = Xscaler.fit_transform(X_cont_train)
        X_cont_val = Xscaler.transform(X_cont_val)      # Scale validation data
        X_cont_test = Xscaler.transform(X_cont_test)
        Yscaler = MinMaxScaler()
        Y_train = Y_train.reshape(-1, 1)
        Y_val = Y_val.reshape(-1, 1)                    # Reshape validation target data
        Y_test = Y_test.reshape(-1, 1)
        Y_train = Yscaler.fit_transform(Y_train)
        Y_val = Yscaler.transform(Y_val)                # Scale validation target data
        Y_test = Yscaler.transform(Y_test)
        return X_cont_train, X_cont_val, X_cont_test, Y_train, Y_val, Y_test, Xscaler, Yscaler

    def non_seq_scaler_standardization(X_cont_train, X_cont_val, X_cont_test, Y_train, Y_val, Y_test):
        Xscaler = StandardScaler()
        X_cont_train = Xscaler.fit_transform(X_cont_train)
        X_cont_val = Xscaler.transform(X_cont_val)      # Scale validation data
        X_cont_test = Xscaler.transform(X_cont_test)
        Yscaler = StandardScaler()
        Y_train = Y_train.reshape(-1, 1)
        Y_val = Y_val.reshape(-1, 1)                    # Reshape validation target data
        Y_test = Y_test.reshape(-1, 1)
        Y_train = Yscaler.fit_transform(Y_train)
        Y_val = Yscaler.transform(Y_val)                # Scale validation target data
        Y_test = Yscaler.transform(Y_test)
        return X_cont_train, X_cont_val, X_cont_test, Y_train, Y_val, Y_test, Xscaler, Yscaler


    # if scaler == 'standard':
    #     X_cont_train, X_cont_test, y_train, y_test, Xscaler, Yscaler = non_seq_scaler_standardization(X_cont_train, X_cont_test, y_train, y_test)
    # elif scaler == 'minmax':
    #     X_cont_train, X_cont_test, y_train, y_test, Xscaler, Yscaler = non_seq_scaler_minmax(X_cont_train, X_cont_test, y_train, y_test)
    # elif scaler == 'none':
    #     pass
    # else:
    #     raise ValueError("scaler must be 'minmax' or 'standard'")
    
    
    # X_train = np.concatenate((X_cont_train, X_cat_train), axis=1)
    # X_test = np.concatenate((X_cont_test, X_cat_test), axis=1)

 #   X_train.shape


    def rev_encode_data(X, y, lookback = lookback, forecast=forecast):
        """
        Encode the data for training an LSTM model, including creating sequences and splitting into input and target.

        Parameters:
        X (numpy.ndarray): The input data array.
        y (numpy.ndarray): The target variable array.
        lookback (int): The number of timesteps to look back (48 hours with 15 minute intervals).
        forecast (int): The number of timesteps to forecast (24 hours with 15 minute intervals).

        Returns:
        X_seq (numpy.ndarray): The encoded input data with sequences.
        y_seq (numpy.ndarray): The encoded target variable array with sequences.
        """

        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        l = lookback
        f = forecast
        X_seq = []
        y_seq = []

        for i in range(len(X) - l - f + 1):
            Xtemp = X[i:i+l+f]
            ytemp = y[i+l:i+l+f]
            X_seq.append(Xtemp)
            y_seq.append(ytemp)
            # if i % 500 == 0:
            #     print(i)

        return np.array(X_seq), np.array(y_seq)



    def rev_encode_data(X, y, lookback, forecast):
        """
        Encode the data for training an LSTM model, including creating sequences for input and target.
        The input will include only the 'lookback' period, whereas the target will include only the 'forecast' period.
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        X_seq = []
        y_seq = []

        # Ensure we only create sequences where a full forecast period can also be created
        for i in range(len(X) - lookback - forecast + 1):
            Xtemp = X[i:i + lookback]
            ytemp = y[i + lookback:i + lookback + forecast]
            X_seq.append(Xtemp)
            y_seq.append(ytemp)

        return np.array(X_seq), np.array(y_seq)



        # Assuming necessary imports and helper functions are defined above

    # Perform time series split including validation
    splits = time_series_cv(X_continuous, X_categorical, df['RTM'].values, n_splits=n_splits, val_fraction=0.1)
    
    # Apply trimming to each split
    trimmed_splits = [trim(split, lookback) for split in splits]

    scaled_encoded_splits = []

    for split in trimmed_splits:
        # Apply MinMax scaling
        X_cont_train_scaled, X_cont_val_scaled, X_cont_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, Xscaler, Yscaler = non_seq_scaler_minmax(
            split['X_cont_train'], split['X_cont_val'], split['X_cont_test'],
            split['Y_train'], split['Y_val'], split['Y_test']
        )
        
        # Concatenate the scaled continuous and categorical features
        X_train = np.concatenate((X_cont_train_scaled, split['X_cat_train']), axis=1)
        X_val = np.concatenate((X_cont_val_scaled, split['X_cat_val']), axis=1)
        X_test = np.concatenate((X_cont_test_scaled, split['X_cat_test']), axis=1)
        
        # Encode the scaled and concatenated data for LSTM
        X_train_encoded, y_train_encoded = rev_encode_data(X_train, Y_train_scaled, lookback=lookback, forecast=forecast)
        X_val_encoded, y_val_encoded = rev_encode_data(X_val, Y_val_scaled, lookback=lookback, forecast=forecast)
        X_test_encoded, y_test_encoded = rev_encode_data(X_test, Y_test_scaled, lookback=lookback, forecast=forecast)
        
        # Collect the fully processed splits
        scaled_encoded_splits.append({
            'X_train_encoded': X_train_encoded,
            'y_train_encoded': y_train_encoded,
            'X_val_encoded': X_val_encoded,
            'y_val_encoded': y_val_encoded,
            'X_test_encoded': X_test_encoded,
            'y_test_encoded': y_test_encoded,
            'Xscaler': Xscaler,
            'Yscaler': Yscaler
        })

    # Optional garbage collection
    gc.collect()

    # Return the list of processed splits
    return scaled_encoded_splits 

    # Encode the train and test data
    X_train_encoded, y_train_encoded = rev_encode_data(X_train, y_train, lookback=lookback, forecast=forecast)
    X_test_encoded, y_test_encoded = rev_encode_data(X_test, y_test, lookback=lookback, forecast=forecast)
    print('X_train_encoded shape:', X_train_encoded.shape)
    print('y_train_encoded shape:', y_train_encoded.shape)
    print('X_test_encoded shape:', X_test_encoded.shape)
    print('y_test_encoded shape:', y_test_encoded.shape)
    import gc
    gc.collect()

    return X_train_encoded, y_train_encoded, X_test_encoded, y_test, y_test_encoded, Xscaler, Yscaler




def pickle_ins(X_train, X_test, y_train, y_test, Xscaler, Yscaler, name):
    dill.dump(Xscaler, open('Xscaler'+name+'.pkl', 'wb'))
    dill.dump(Yscaler, open('Yscaler'+name+'.pkl', 'wb'))
    np.save('X_train_encoded'+name+'.npy', X_train)
    np.save('X_test_encoded'+name+'.npy', X_test)
    np.save('y_train_encoded'+name+'.npy', y_train)
    np.save('y_test_encoded'+name+'.npy', y_test)
    return

# %%
# BE AWARE: depeding on size of might want to stratify by months when splitting
#train_test_split(X, Y_reshaped, months_reshaped, test_size=test_size, stratify=months_reshaped, random_state=42)

# %%
def relative_mae(y_true, y_pred):
    """
    Calculates the Relative Mean Absolute Error (Relative MAE). an arguably better metric is the relative MAE (rMAE). 
    Similar to MASE, rMAE normalizes the MAE by the MAE of a naive forecast. 
    However, instead of considering the in-sample dataset, the naive forecast is built based on the out-of-sample dataset.
    
    Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
        
    Returns:
        float: The Relative Mean Absolute Error.
    """
    mae = np.mean(np.abs(y_pred - y_true))  # Calculate the Mean Absolute Error
    mean_actual = np.mean(np.abs(y_true))  # Calculate the mean of absolute values of actual data
    
    return mae / mean_actual

def plot_sequences(X, y, title='Sample Sequences'):

    plt.figure(figsize=(14, 6))
    num_sequences = 3
    for i in range(num_sequences):
        plt.plot(X[i], label=f'Sequence {i} Features')
        plt.plot(y[i], label=f'Sequence {i} Target', linestyle='--')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Scaled Value')
    plt.legend()
    plt.show()


def run_model(X_train_encoded, y_train_encoded, X_val_encoded, y_val_encoded, epochs, run_name, model_name = 'BiLSTM_3layers', regularizer=0.0001, dropout=0.2):
    # Print shapes to verify
    print("Train shapes: X =", X_train_encoded.shape, "y =", y_train_encoded.shape)
    #print("Validation shapes: X =", X_val_encoded.shape, "y =", y_val_encoded.shape)
    print("Test shapes: X =", X_val_encoded.shape, "y =", y_val_encoded.shape)

        # Check for NaN values in X_train_encoded and y_train_encoded
    if np.isnan(X_train_encoded).any() or np.isnan(y_train_encoded).any():
        raise ValueError("NaN values found in X_train_encoded or y_train_encoded")

    # Optionally plot a few sequences
    import matplotlib.pyplot as plt


    batch_size = 32


    input_shape = (batch_size,X_train_encoded.shape[1], X_train_encoded.shape[2])  # (num_timesteps, num_features)
    print(input_shape)
    BiLSTM_3layers = Sequential()
    BiLSTM_3layers.add(Bidirectional(LSTM(64, return_sequences=True, activation='tanh', kernel_regularizer=l2(regularizer))))#, input_shape=input_shape)))
    BiLSTM_3layers.add(Dropout(dropout))
    BiLSTM_3layers.add(Bidirectional(LSTM(32, return_sequences=True, activation='tanh', kernel_regularizer=l2(regularizer))))
    BiLSTM_3layers.add(Dropout(dropout))
    BiLSTM_3layers.add(Bidirectional(LSTM(10, return_sequences=True, activation='tanh', kernel_regularizer=l2(regularizer))))
    BiLSTM_3layers.add(Dense(1))
    BiLSTM_3layers.compile(optimizer='adam', loss='mse', metrics=["mean_absolute_error"])
    # Train model on CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Train model
    history = BiLSTM_3layers.fit(X_train_encoded,y_train_encoded, batch_size=batch_size, epochs=epochs, validation_data=(X_val_encoded, y_val_encoded), verbose=2)

    # Save model
    BiLSTM_3layers.save('Model_Outputs/model_BiLSTM_3layers_'+str(dt.datetime.now().date)+run_name+'.keras')
    return history, BiLSTM_3layers



def predict(X_test_encoded, y_test_encoded, BiLSTM_3layers, Yscaler):
    # Predict and inverse transform to get predictions in original scale
    predictions = BiLSTM_3layers.predict(X_test_encoded)
    predictions = predictions.reshape(-1, 1)  # Flatten predictions to 2D for inverse scaling
    predictions = Yscaler.inverse_transform(predictions)
    predictions = predictions.flatten()  # Flatten to 1D for easy handling

    # Adjust y_test for comparison:
    # If y_test is 3D (batches, timesteps, features), you must flatten it similarly as predictions
    y_test_reshaped = y_test_encoded.reshape(-1, 1)  # Reshape y_test to 2D
    y_test_nonscale = Yscaler.inverse_transform(y_test_reshaped)
    y_test_nonscale = y_test_nonscale.flatten()  # Flatten to 1D

    # Ensure that predictions and actuals are the same length
    min_length = min(len(predictions), len(y_test_nonscale))
    predictions = predictions[:min_length]
    y_test_nonscale = y_test_nonscale[:min_length]

    # Create DataFrame to compare actual vs predicted values
    pred_df = pd.DataFrame({'Actual': y_test_nonscale, 'Predicted': predictions})
    pred_df.to_csv('Model_Outputs/predictions_BiLSTM_3layers_' + dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv')

    # Plotting
    # plt.figure(figsize=(15, 7))
    # plt.plot(pred_df['Actual'], label='Actual', alpha=0.7)
    # plt.plot(pred_df['Predicted'], label='Predicted', alpha=0.7)
    # plt.title('Actual vs Predicted Prices')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Price')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return pred_df


def preprocessandrun(epochs = 8, lookback = 96, forecast = 96, test_size = 0.2, scaler = 'minmax'):
    X_train_encoded, y_train_encoded, X_test_encoded, y_test, y_test_encoded, Xscaler, Yscaler, params = preprocess(1, lookback, forecast, test_size)

    pickle_ins(X_train_encoded, X_test_encoded, y_train_encoded, y_test, Xscaler, Yscaler)
    
    history, model = run_model(X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded, epochs)
    
    preds = predict(X_test_encoded, y_test, model, Yscaler, lookback, forecast)
    
    return history, model, preds

# %% [markdown]
# Troubleshooting:
# 
# Check the variance in your y_train to ensure there is enough variance to predict.
# 
# Try using a different activation function like 'tanh' or 'sigmoid'.
# 
# Experiment with different architectures, reducing the complexity of the model.
# 
# Scale back the L2 regularization (i.e., lower the lambda value).
# 
# Try different batch sizes and number of epochs.
# 
# Plot the training and validation loss to see if the model is learning over time.
# 
