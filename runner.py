## Use this script to run the model and properly save the params

from lstmmodel import *
import datetime
import pickle

preprocess_inputs = [
{
    'chop' : 1,
    'lookback' : 96,
    'forecast' : 96,
    # 'n_splits' : 5,
    #'val_size': .1,
    'test_size' : .2,
    'scaler' : 'standard',
    'name' : 'standard96newencode',
    # 'lookback': 48,
    # 'forecast': 24
}
]

model_inputs = [
    {
        'epochs' : 10,
        'regularizer' : 0,
        'run_name' : 'noreg005drop_10epoch',
        'dropout' : 0.05
    },
    {
        'epochs' : 10,
        'regularizer' : 0.00001,
        'run_name' : '105reg005 drop_10epoch',
        'dropout' : 0.05
    },
    {
        'epochs' : 10,
        'regularizer' : 0.0001,
        'run_name' : '105reg005 drop_10epocht',
        'dropout' : 0
    },
        {
        'epochs' : 10,
        'regularizer' : 0.00001,
        'run_name' : '105reg005 drop_10epoch',
        'dropout' : 0
    }
]

# model_inputs = [{
#     'epochs' : 10,
#     'regularizer' : .0001,
#     'run_name' : 'lowl2',
#     'dropout' : 0.05,
# }
# ]


history = {}
model = {}
preds = {}


#for train_index, val_index in tscv.split(X_cont_train_val):

split_index = 0  # Initialize split index
for pp_input in preprocess_inputs:
    scaled_encoded_splits = preprocess(**pp_input)

    # Loop through each split returned by the preprocess function
    for split in scaled_encoded_splits:
        split_name = f"{pp_input['name']}_split_{str(split_index)}"  # Unique name for each split

        # Extract the datasets and scalers from each split dictionary
        X_train_encoded = split['X_train_encoded']
        y_train_encoded = split['y_train_encoded']
        X_val_encoded = split['X_val_encoded']
        y_val_encoded = split['y_val_encoded']
        X_test_encoded = split['X_test_encoded']
        y_test_encoded = split['y_test_encoded']
        Xscaler = split['Xscaler']
        Yscaler = split['Yscaler']
        
        # Optional: Serialize the processed data and scalers
        # pickle_out = open(f"{pp_input['preprocess_name']}_data.pkl", "wb")
        # pickle.dump(split, pickle_out)
        # pickle_out.close()

        # Loop through each model input configuration
        for m_input in model_inputs:
            print(m_input)
            run_name = m_input['run_name'] + '_' + str(split_name)
            
            # Include validation data in training process
            # Assuming run_model is updated to accept X_val and y_val as arguments
            if 'X_val_encoded' in split and 'y_val_encoded' in split:
                history[run_name], model[run_name] = run_model(
                    X_train_encoded=X_train_encoded,
                    y_train_encoded=y_train_encoded,
                    X_val_encoded=X_val_encoded,
                    y_val_encoded=y_val_encoded,
                    **m_input
                )
            else:
                # Fallback to train without validation if not available
                history[run_name], model[run_name] = run_model(
                    X_train_encoded=split['X_train_encoded'],
                    y_train_encoded=split['y_train_encoded'],
                    **m_input
                )
            
            # Predict using the trained model
            # Optional: Validate model on validation data before testing
            # if 'X_val_encoded' in split and 'y_val_encoded' in split:
            #     val_preds = predict(split['X_val_encoded'], split['y_val_encoded'], model[run_name], Yscaler, pp_input['lookback'], pp_input['forecast'])
                
            #     print(f"Validation Predictions for {run_name}: {val_preds}")

            preds[run_name] = predict(X_test_encoded = X_test_encoded, y_test_encoded = y_test_encoded, BiLSTM_3layers = model[run_name],
                                      Yscaler=Yscaler,) #lookback=pp_input['lookback'], forecast=pp_input['forecast'])

        split_index += 1
        
# Get the current datetime
current_datetime = datetime.datetime.now()

# Create a filename using the current datetime
filename = f"output_{current_datetime.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"

# Dump the history, model, and preds dictionaries to a .pkl file
with open(filename, 'wb') as file:
    pickle.dump((history, model, preds), file)