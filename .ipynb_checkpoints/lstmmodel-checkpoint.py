
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

import dill
# load the DF from pickle to speed pre-processing
def preprocess(chop, lookback, forecast, test_size, scaler, name = None):
    df = pd.read_pickle('Data_Clean/merged_clean_df.pkl')

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
    X_categorical = df[['Day', 'Month', 'Year', 'Hour', 'Minute']]
    X_categorical = enc.fit_transform(X_categorical).toarray()

    # print(df.shape)
    # print(X_continuous.shape)
    # print(X_categorical.shape)
    # print(df['RTM'].shape)

    import gc
    gc.collect()
    
    #### We are splitting the data into test/train before we create the split sequences to input into
    #### LSTM. We have already onehotencoded the data and have lagged it
    
    def non_seq_split(X_continuous, X_categorical, Y, test_size=0.2):
        X_cont_train = X_continuous[:int(len(X_continuous) * (1 - test_size))]
        X_cont_test = X_continuous[int(len(X_continuous) * (1 - test_size)):]
        X_cat_train = X_categorical[:int(len(X_categorical) * (1 - test_size))]
        X_cat_test = X_categorical[int(len(X_categorical) * (1 - test_size)):]
        Y_train = Y[:int(len(Y) * (1 - test_size))]
        Y_test = Y[int(len(Y) * (1 - test_size)):]
        return X_cont_train, X_cat_train, X_cont_test, X_cat_test, Y_train, Y_test

    def trim(arr, size):
        if isinstance(arr, tuple):
            return tuple(a[size:] for a in arr)
        else:
            return arr[size:]

    X_cont_train, X_cat_train, X_cont_test, X_cat_test, y_train, y_test = trim(non_seq_split(X_continuous, X_categorical, df['RTM'].values, test_size=test_size),lookback)

    # commented without trimmed version
    #X_cont_train, X_cat_train, X_cont_test, X_cat_test, y_train, y_test = non_seq_split(X_continuous, X_categorical, df['Settlement Point Price_RTM'].values)


    def non_seq_scaler_minmax(X_cont_train, X_cont_test, Y_train, Y_test):
        Xscaler = MinMaxScaler()
        X_cont_train = Xscaler.fit_transform(X_cont_train)
        X_cont_test = Xscaler.transform(X_cont_test)
        Yscaler = MinMaxScaler()
        Y_train = Y_train.reshape(-1, 1)
        Y_test = Y_test.reshape(-1, 1)
        Y_train = Yscaler.fit_transform(Y_train)
        Y_test = Yscaler.transform(Y_test)
        return X_cont_train, X_cont_test, Y_train, Y_test, Xscaler, Yscaler

    def non_seq_scaler_standardization(X_cont_train, X_cont_test, Y_train, Y_test):
        Xscaler = StandardScaler()
        X_cont_train = Xscaler.fit_transform(X_cont_train)
        X_cont_test = Xscaler.transform(X_cont_test)
        Yscaler = StandardScaler()
        Y_train = Y_train.reshape(-1, 1)
        Y_test = Y_test.reshape(-1, 1)
        Y_train = Yscaler.fit_transform(Y_train)
        Y_test = Yscaler.transform(Y_test)
        return X_cont_train, X_cont_test, Y_train, Y_test, Xscaler, Yscaler

    if scaler == 'standard':
        X_cont_train, X_cont_test, y_train, y_test, Xscaler, Yscaler = non_seq_scaler_standardization(X_cont_train, X_cont_test, y_train, y_test)
    elif scaler == 'minmax':
        X_cont_train, X_cont_test, y_train, y_test, Xscaler, Yscaler = non_seq_scaler_minmax(X_cont_train, X_cont_test, y_train, y_test)
    elif scaler == 'none':
        pass
    else:
        raise ValueError("scaler must be 'minmax' or 'standard'")
    
    X_train = np.concatenate((X_cont_train, X_cat_train), axis=1)
    X_test = np.concatenate((X_cont_test, X_cat_test), axis=1)

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


def run_model(X_train_encoded, y_train_encoded, X_test_encoded, y_test_encoded, epochs, run_name, model_name = 'BiLSTM_3layers', regularizer=0.0001, dropout=0.2):
    # Print shapes to verify
    print("Train shapes: X =", X_train_encoded.shape, "y =", y_train_encoded.shape)
    #print("Validation shapes: X =", X_val_encoded.shape, "y =", y_val_encoded.shape)
    print("Test shapes: X =", X_test_encoded.shape, "y =", y_test_encoded.shape)

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
    BiLSTM_3layers.add(Bidirectional(LSTM(10, return_sequences=False, activation='tanh', kernel_regularizer=l2(regularizer))))
    BiLSTM_3layers.add(Dense(1))
    BiLSTM_3layers.compile(optimizer='adam', loss='mse', metrics=["mean_absolute_error"])
    # Train model on CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Train model
    history = BiLSTM_3layers.fit(X_train_encoded,y_train_encoded, batch_size=batch_size, epochs=epochs, validation_data=(X_test_encoded, y_test_encoded), verbose=2)

    # Save model
    BiLSTM_3layers.save('Model_Outputs/model_BiLSTM_3layers_'+str(dt.datetime.now().date)+run_name+'.keras')
    return history, BiLSTM_3layers

# change predict to be encoded?
def predict(X_test_encoded, y_test, BiLSTM_3layers, Yscaler, lookback, forecast):
    predictions = BiLSTM_3layers.predict(X_test_encoded)
    predictions = Yscaler.inverse_transform(predictions)
    predictions = predictions.reshape(-1)
    predictions.reshape(-1)
    y_test_nonscale = Yscaler.inverse_transform(y_test).reshape(-1)

    print(predictions.shape)
    print(y_test_nonscale.shape)
    y_test_nonscale = y_test_nonscale[lookback+forecast-1:]
    print('new shape'+str(y_test_nonscale.shape))

    pred_df = pd.DataFrame({'Actual': y_test_nonscale, 'Predicted': predictions})
    pred_df.to_csv('Model_Outputs/predictions_BiLSTM_3layers_'+str(dt.datetime.now())+'.csv')
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
