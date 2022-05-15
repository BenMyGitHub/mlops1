import pandas as pd


import os


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Set path to inputs
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
Xtrain_data_file = 'Xtrain.csv'
ytrain_data_file = 'ytrain.csv'
Xtrain_data_path = os.path.join(PROCESSED_DATA_DIR, Xtrain_data_file)
ytrain_data_path = os.path.join(PROCESSED_DATA_DIR, ytrain_data_file)
# Read data


# Split data into dependent and independent variables
X_train = pd.read_csv(Xtrain_data_path, sep=",")
y_train = pd.read_csv(ytrain_data_path, sep=",")


# Model 
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])


# Set path to output (model)
MODEL_DIR = os.environ["MODEL_DIR"]
model_name = 'action.h5'
model_path = os.path.join(MODEL_DIR, model_name)

# Serialize and save model

model.save(model_name,model_path)



