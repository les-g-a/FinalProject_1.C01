
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
chop = 1
lookback = 96
forecast = 96
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


def non_seq_scaler_minmax(X_cont_train, X_cont_test):
    Xscaler = MinMaxScaler()
    X_cont_train = Xscaler.fit_transform(X_cont_train)
    X_cont_test = Xscaler.transform(X_cont_test)
    return X_cont_train, X_cont_test, Xscaler

def non_seq_scaler_standardization(X_cont_train, X_cont_test):
    Xscaler = StandardScaler()
    X_cont_train = Xscaler.fit_transform(X_cont_train)
    X_cont_test = Xscaler.transform(X_cont_test)
    
    return X_cont_train, X_cont_test,  Xscaler

X_cont_train, X_cont_test, Xscaler = non_seq_scaler_standardization(X_cont_train, X_cont_test)
X_train = np.concatenate((X_cont_train, X_cat_train), axis=1)
X_test = np.concatenate((X_cont_test, X_cat_test), axis=1)

def pickle_ins(X_train, X_test, y_train, y_test, Xscaler):
    dill.dump(Xscaler, open('Xscaler.pkl', 'wb'))
    dill.dump(X_train, open('X_train.pkl', 'wb'))
    dill.dump(X_test, open('X_test.pkl', 'wb'))
    dill.dump(y_train, open('y_train.pkl', 'wb'))
    dill.dump(y_test, open('y_test.pkl', 'wb'))

pickle_ins(X_train, X_test, y_train, y_test, Xscaler)

print(y_train[:35])
# Define the model

model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1, validation_data=(X_test, y_test))

# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

# Plot the training and validation accuracy and loss at each epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# predict on y test and then plot the comparison
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred = y_pred.astype(int)

plt.scatter(df.index[-len(y_pred):], df['RTM'][-len(y_pred):] - df['DAM'][-len(y_pred):], c=y_pred, cmap='coolwarm')
plt.colorbar()
plt.xlabel('Index')
plt.ylabel('Difference (RTM - DAM)')
plt.title('Prediction of Positive/Negative Difference')
plt.show()

plt.legend()
plt.show()

print(y_pred[:35])
print(y_test[:35])

