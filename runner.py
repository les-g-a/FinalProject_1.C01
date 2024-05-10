## Use this script to run the model and properly save the params

from lstmmodel import *
import datetime
import pickle

preprocess_inputs = [
{
    'chop' : 1,
    'lookback' : 96,
    'forecast' : 96,
    'test_size' : .2,
    'scaler' : 'standard',
    'name' : 'standard96newencode'
}
]

model_inputs = [
    {
        'epochs' : 20,
        'regularizer' : 0,
        'run_name' : 'noreg005drop_newencodingtest',
        'dropout' : 0.05
    },
    {
        'epochs' : 20,
        'regularizer' : 0.00001,
        'run_name' : '105reg005 drop_newencodingtest',
        'dropout' : 0.05
    },
    {
        'epochs' : 20,
        'regularizer' : 0.0001,
        'run_name' : '105reg005 drop_newencodingtest',
        'dropout' : 0
    },
        {
        'epochs' : 20,
        'regularizer' : 0.00001,
        'run_name' : '105reg005 drop_newencodingtest',
        'dropout' : 0
    }
]

# model_inputs = {
#     'epochs' : 10,
#     'regularization' : .0001,
#     'run_name' : 'lowl2'
# }



history = {}
model = {}
preds = {}
for pp_input in preprocess_inputs:
    X_train_encoded, y_train_encoded, X_test_encoded, y_test, y_test_encoded, Xscaler, Yscaler = preprocess(**pp_input)
    #pickle_ins(X_train_encoded, y_train_encoded, X_test_encoded, y_test, y_test_encoded, Xscaler, Yscaler,pp_input['preprocess_name'])
    for m_input in model_inputs:
        name = m_input['run_name']+'_'+pp_input['name']
        history[name], model[name] = run_model(X_train_encoded, y_train_encoded, X_test_encoded, y_test_encoded, **m_input)
        preds[name] = predict(X_test_encoded, y_test, model[name], Yscaler, pp_input['lookback'], pp_input['forecast'])

# Get the current datetime
current_datetime = datetime.datetime.now()

# Create a filename using the current datetime
filename = f"output_{current_datetime.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"

# Dump the history, model, and preds dictionaries to a .pkl file
with open(filename, 'wb') as file:
    pickle.dump((history, model, preds), file)