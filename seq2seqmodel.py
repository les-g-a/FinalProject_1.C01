## Seq2Seq model code
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
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout
import dill



def create_sequences(X, y, input_length, prediction_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - input_length - prediction_length + 1):
        X_seq.append(X[i:(i + input_length)])
        y_seq.append(y[(i + input_length):(i + input_length + prediction_length)])
    return np.array(X_seq), np.array(y_seq)

X, y = encode_data()

input_seq_length = 48 * 4  # 48 hours with 15-minute intervals
output_seq_length = 24 * 4  # 24 hours with 15-minute intervals

X_sequences, y_sequences = create_sequences(X, y, input_seq_length, output_seq_length)
y_sequences = y_sequences.reshape(y_sequences.shape[0], y_sequences.shape[1], 1)  # Reshaping for model compatibility

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)


### Model 3

# input_seq_length: This should be the number of time steps you want your model to consider for each input sequence. It’s a hyperparameter you set based on your data or problem specifics.

# output_seq_length: This is similar but for the output. For some sequence-to-sequence models, this might be the same as the input sequence length, or it might be different, such as when predicting future values or translating sentences to another language with different sentence lengths.

%%
X, y = encode_data()

input_seq_length = 48 * 4  # 48 hours with 15-minute intervals
output_seq_length = 24 * 4  # 24 hours with 15-minute intervals

X_sequences, y_sequences = create_sequences(X, y, input_seq_length, output_seq_length)
y_sequences = y_sequences.reshape(y_sequences.shape[0], y_sequences.shape[1], 1)  # Reshaping for model compatibility

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

X_train.shape[2]

%%
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

%%
df_trimmed = trim_data(df)
X, y, months = encode_data(df=df_trimmed)
# Use the function to trim the data
#X_trimmed, y_trimmed = trim_data(X, y)
X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, scaler_X, scaler_y = prep_LSTM_data(X=X, y=y, months=months)
input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])  # (num_timesteps, num_features)

# %%

# Define input sequence length and the number of features
input_seq_length = 48 * 4  # 48 hours x 4, 15 minutes intervals 
output_seq_length = 24 * 4  # 24 hours x 4, 15 minutes intervals
num_features = X_train_encoded.shape[2]  # Adjust this to match the second dimension of X_train after preprocessing

## SEQ2SEQ architecture

# Define the input shape for the encoder
encoder_inputs = Input(shape=(input_seq_length, num_features))

# Define the LSTM layers with dropout as you had previously set them up
encoder_lstm1 = LSTM(128, return_sequences=True, return_state=True, dropout=0.2)
encoder_outputs1, state_h1, state_c1 = encoder_lstm1(encoder_inputs)

encoder_lstm2 = LSTM(64, return_sequences=True, return_state=True, dropout=0.2)
encoder_outputs2, state_h2, state_c2 = encoder_lstm2(encoder_outputs1)

encoder_lstm3 = LSTM(32, return_state=True, dropout=0.2)
encoder_outputs3, state_h3, state_c3 = encoder_lstm3(encoder_outputs2)
encoder_states = [state_h3, state_c3]

# Define the decoder input and LSTM layer with dropout using the last encoder states as initial state
decoder_inputs = Input(shape=(output_seq_length, 1))  # Assuming the output feature is 1
decoder_lstm = LSTM(32, return_sequences=True, return_state=False, dropout=0.2)
decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# Add a dropout layer after the LSTM layers
decoder_dropout = Dropout(0.2)(decoder_outputs)

# Define the dense output layer
decoder_dense = Dense(1, activation='linear')
decoder_outputs = decoder_dense(decoder_dropout)

# Define the Seq2Seq model that will turn `encoder_inputs` & `decoder_inputs` into `decoder_outputs`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Summary of the model
model.summary()


# %%
# Create a Zero-Padded Initial Input for the Decoder
decoder_input_train = np.zeros((y_train.shape[0], y_train.shape[1], 1))  # 
decoder_input_test = np.zeros((y_test.shape[0], y_test.shape[1], 1))  # 

## TODO do we train on encoded sequences?
history = model.fit(
    [X_train, decoder_input_train],  # Provide initial state for the decoder
    y_train,                         # Actual target outputs for training
    epochs=10,
    batch_size=64,
    validation_data=([X_test, decoder_input_test], y_test)  # Similar setup for validation
)


# %%
# Create the initial input for the decoder for prediction
decoder_input_test = np.zeros((y_test.shape[0], y_test.shape[1], 1))  # Assuming the same shape as during training

# Make predictions using the model
predictions = model.predict([X_test, decoder_input_test])

# %%
# Flatten the predictions and actual values if they are in three dimensions
predictions = predictions.reshape(-1)
actual = y_test.reshape(-1)


# %%
mse = mean_squared_error(actual, predictions)
mae = mean_absolute_error(actual, predictions)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Plot the overall comparison
plot_comparison(predictions, actual)

# Plot a specific interval for detailed analysis
start_index = 100  # example start index
end_index = 200    # example end index
plot_comparison(predictions, actual, startx=start_index, endx=end_index)
