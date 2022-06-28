import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['head', 'Movement'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 30



label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
# Set path to the outputs
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
Xtrain_path = os.path.join(PROCESSED_DATA_DIR, 'Xtrain.npy')
ytrain_path = os.path.join(PROCESSED_DATA_DIR, 'ytrain.npy')
Xtest_path = os.path.join(PROCESSED_DATA_DIR, 'Xtest.npy')
ytest_path = os.path.join(PROCESSED_DATA_DIR, 'ytest.npy')
np.save('./processed_data/Xtrain.npy',X_train)
np.save('./processed_data/ytrain.npy', y_train)
np.save('./processed_data/Xtest.npy',  X_test)
np.save('./processed_data/ytest.npy',  y_test)
