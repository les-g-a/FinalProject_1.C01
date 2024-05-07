
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
df = pd.read_pickle('Data_Clean/merged_clean_df.pkl')

###########
# Set params
###########

# set to >1 to chop off 1/n of the data
chop = 3
lookback = 96
forecast = 1
epochs = 3
test_size = 0.2

# set to 'minmax' or 'standard'. A dict after the functions are defined routes the function call
# above not yet implemented
#scaler = 'standard'

# the params are dumped along with the model below
params = {'lookback': lookback, 'forecast': forecast, 'chop': chop, 'epochs': epochs}
###########
###########

# Shorten dataset for quicker testing
df = df.iloc[-int(len(df)/chop):]
sequence_length = lookback + forecast

# Lag all variables that are what we're trying to predict in the forecast period by the forecast period
df['diffsign'] = np.where(df['RTM'].values - df['DAM'].values > 0, 1, 0)
df['RTMlag'] = df['diffsign'].shift(forecast)
df['load_diff'] = df['load_diff'].shift(forecast)
df.dropna(inplace=True)
X_vars = list(df.columns)
X_vars.remove('RTM')
X_vars


# %%

X_continuous = df[['DAM', 'RTMlag', 'load_fc', 'load_diff']].values
# Categorical features
enc = OneHotEncoder(handle_unknown='ignore')
X_categorical = df[['Day', 'Month', 'Year', 'Hour', 'Minute']]
X_categorical = enc.fit_transform(X_categorical).toarray()

# %%
print(df.shape)
print(X_continuous.shape)
print(X_categorical.shape)
print(df['RTM'].shape)

# %%
import gc
gc.collect()

# %%
def non_seq_split(X_continuous, X_categorical,Y, test_size=0.2):
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

X_cont_train, X_cat_train, X_cont_test, X_cat_test, y_train, y_test = trim(non_seq_split(X_continuous, X_categorical, df['diffsign'].values, test_size=test_size),lookback)

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

X_cont_train, X_cont_test, y_train, y_test, Xscaler, Yscaler = non_seq_scaler_standardization(X_cont_train, X_cont_test, y_train, y_test)
X_train = np.concatenate((X_cont_train, X_cat_train), axis=1)
X_test = np.concatenate((X_cont_test, X_cat_test), axis=1)

def pickle_ins(X_train, X_test, y_train, y_test, Xscaler, Yscaler):
    dill.dump(Xscaler, open('Xscaler.pkl', 'wb'))
    dill.dump(Yscaler, open('Yscaler.pkl', 'wb'))
    dill.dump(X_train, open('X_train.pkl', 'wb'))
    dill.dump(X_test, open('X_test.pkl', 'wb'))
    dill.dump(y_train, open('y_train.pkl', 'wb'))
    dill.dump(y_test, open('y_test.pkl', 'wb'))

pickle_ins(X_train, X_test, y_train, y_test, Xscaler, Yscaler)
# %%

X_train.shape

# %%

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
        if i % 500 == 0:
            print(i)

    return np.array(X_seq), np.array(y_seq)


# Encode the train and test data
X_train_encoded, y_train_encoded = rev_encode_data(X_train, y_train, lookback=lookback, forecast=forecast)
X_test_encoded, y_test_encoded = rev_encode_data(X_test, y_test, lookback=lookback, forecast=forecast)

# %%
import gc
gc.collect()

# %%


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



# Print shapes to verify
print("Train shapes: X =", X_train_encoded.shape, "y =", y_train_encoded.shape)
#print("Validation shapes: X =", X_val_encoded.shape, "y =", y_val_encoded.shape)
print("Test shapes: X =", X_test_encoded.shape, "y =", y_test_encoded.shape)

# Check for NaN values in X_train_encoded and y_train_encoded
if np.isnan(X_train_encoded).any() or np.isnan(y_train_encoded).any():
    raise ValueError("NaN values found in X_train_encoded or y_train_encoded")

# Optionally plot a few sequences
import matplotlib.pyplot as plt

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

#plot_sequences(X_train_encoded[:3], y_train_encoded[:3])

batch_size = 32

# Define the LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(units=128, return_sequences=True), input_shape=(X_train_encoded.shape[1], X_train_encoded.shape[2])))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(units=64)))
model.add(Dropout(0.2))
model.add(Dense(units=y_train_encoded.shape[1], activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_encoded, y_train_encoded, batch_size=batch_size, epochs=epochs, validation_data=(X_test_encoded, y_test_encoded))

# Train model on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Train model
history = model.fit(X_train_encoded,y_train_encoded, batch_size=batch_size, epochs=epochs, validation_data=(X_test_encoded, y_test_encoded))

# Save model
model.save('Model_Outputs/model_BiLSTM_3layers_'+str(dt.datetime.now())+'.keras')

# Plot training & validation loss values

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# 

predictions = BiLSTM_3layers.predict(X_test_encoded)
predictions = Yscaler.inverse_transform(predictions)
predictions = predictions.reshape(-1)
predictions.reshape(-1)
y_test_nonscale = Yscaler.inverse_transform(y_test).reshape(-1)



print(X_test.shape)
print(y_test.shape)
# cut off the first of the tests
print(predictions.shape)
print(y_test_nonscale.shape)
y_test_nonscale = y_test_nonscale[lookback+forecast-1:]
print('new shape'+str(y_test_nonscale.shape))

# %%
pred_df = pd.DataFrame({'Actual': y_test_nonscale, 'Predicted': predictions})
pred_df.to_csv('Model_Outputs/predictions_BiLSTM_3layers_'+str(dt.datetime.now())+'.csv')

# TODO save params

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
