# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:01:00 2021
PTM: wrist+thigh sensor
stack both wrist and thigh as input

train a model for ankle+thigh, use trial1+trial2 data
acts: 0-'Lying', 1-'Sitting', 2-'Standing', 3-'TUG', 4-'Walking'
@author: hao_y
"""

from keras.layers import Input, Conv2D,  Dense, Flatten,MaxPooling2D, concatenate,Permute,Reshape,LSTM
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import Adam

# =============================================================================
# build model and oompile
# =============================================================================
#number of activities
NUM_CLASSES = 4

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 60

# Batch Size
BATCH_SIZE = 100

# Number filters convolutional layers
NUM_FILTERS = 64

# Size filters convolutional layers
FILTER_SIZE = 5

# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128

NUM_CHANNEL = 6

def create_base_network():
    multi_input = Input(shape=(1, SLIDING_WINDOW_LENGTH, NUM_CHANNEL), name='multi_input')
    print(multi_input.shape)  # (?, 1, 24, 113)
    
    y = Conv2D(64, (5, 1), activation='relu', data_format='channels_first')(multi_input)
    print(y.shape)  # (?, 64, 20, 113)
    
    y = Conv2D(64, (5, 1), activation='relu', data_format='channels_first')(y)
    print(y.shape)
    y = Conv2D(64, (5, 1), activation='relu', data_format='channels_first')(y)
    print(y.shape)
    y = Conv2D(64, (5, 1), activation='relu', data_format='channels_first')(y)
    print(y.shape)
    
    y = Permute((2, 1, 3))(y)
    print(y.shape)  # (?, 20, 64, 113)
    
    # This line is what you missed
    # ==================================================================
    y = Reshape((int(y.shape[1]), int(y.shape[2]) * int(y.shape[3])))(y)

    # ==================================================================
    print(y.shape)  # (?, 20, 7232)
    
    y = LSTM(128,dropout=0.25,return_sequences=True)(y)
    y = LSTM(128)(y)
      # (?, 128)
    y = Dense(NUM_CLASSES,activation = 'softmax')(y)
    print(y.shape)
    return Model(inputs=multi_input, outputs=y)  

model = create_base_network()
adam_optim = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
model.compile(loss='binary_crossentropy', optimizer=adam_optim,metrics=['accuracy'])

# =============================================================================
# load data and train
# in the order of ['thigh','wrist','ankle']
# =============================================================================
import numpy as np
import glob
from tools import convert_list, convert_y
from sklearn import preprocessing
# to load the windowed data
def load_tensor(path):
    # Read the array from disk
    #new_data = np.loadtxt('./ActiveData/Sub'+str(name)+'_data.txt')
    new_data = np.loadtxt(path)
    
    # Note that this returned a 2D array!
    print (new_data.shape)
    
    # However, going back to 3D is easy if we know the 
    # load wrist+thigh
    new_data = new_data[:,:6].reshape((-1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    return new_data

def load_by_trial(trial_id):
    path = '../data/trial'+str(trial_id)+'/*_train_data.txt'
    x_list = glob.glob(path)
    path = '../data/trial'+str(trial_id)+'/*_train_label.txt'
    y_list = glob.glob(path)
    X,Y = [],[]
    for i in range(len(x_list)):
        x = load_tensor(x_list[i])
        y = np.loadtxt(y_list[i])
        X.append(x)
        Y.append(y)
    x_ = convert_list(X)
    y_ = convert_y(Y)
    idx = np.where(y_==4)
    y_[idx] = 3 
    return x_, y_


x1,y1 = load_by_trial(1)
x2,y2 = load_by_trial(2)

X = np.vstack((x1,x2))
X = X.reshape(-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL)
y = np.concatenate((y1,y2))

labels = np.array([0,1,2,3])
lb = preprocessing.LabelBinarizer()
lb.fit(y=labels)

y_onehot = lb.transform(y)

#start model training
import tensorflow as tf
import matplotlib.pyplot as plt

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
#for i in range(5):
    #with tf.device('/gpu:0'): 
history = model.fit(X, y_onehot,
      batch_size=BATCH_SIZE,
      epochs=100,
      validation_split=0.2,
      callbacks=[callback],
      shuffle=True)

# =============================================================================
# save training history plot and trained model
# =============================================================================
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('wrist_thigh_model.png')

# serialize model to JSON
model_json = model.to_json()
with open("wrist_thigh.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("wrist_thigh.h5")
print("Saved model to disk")