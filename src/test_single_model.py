# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 17:08:56 2021
test single models

'Lying', 'Sitting', 'Standing', 'TUG', 'Walking'
@author: hao_y
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sn
from keras.models import model_from_json

sensor = 'wrist'

#evaluate and print predict result
def eval_perf(ground_truth, predicted_event):
    print('Accuracy score is: ')
    acc = accuracy_score(ground_truth, predicted_event)
    print(acc)
    print('Confusion Matrix is:')
    my_matrix = confusion_matrix(ground_truth, predicted_event)
    my_matrix_n = normalize(my_matrix, axis=1,norm = 'l1')
    print(pd.DataFrame(my_matrix_n).applymap(lambda x: '{:.2%}'.format(x)).values)

    #target_names = ['Walking','Jogging','Stairs','Sitting','Standing']
    target_names = ['Lying', 'Sitting', 'Standing', 'Walking']
    df_cm = pd.DataFrame(my_matrix_n, index = [i for i in target_names],
                  columns = [i for i in target_names])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    print(classification_report(ground_truth, predicted_event, target_names=target_names))  
    return acc

def load_model(sensor):
    json_file = open('../trained_model/'+sensor+'/4act_'+sensor+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('../trained_model/'+sensor+'/4act_'+sensor+'.h5')
    print("Loaded model from disk")
    print(model.summary())
    return model

model = load_model(sensor)

# =============================================================================
# load data and train
# =============================================================================
import numpy as np
import glob
from tools import convert_list, convert_y
from sklearn import preprocessing

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 60

NUM_CHANNEL = 3

# to load the windowed data
def load_tensor(path):
    # Read the array from disk
    #new_data = np.loadtxt('./ActiveData/Sub'+str(name)+'_data.txt')
    new_data = np.loadtxt(path)
    
    # Note that this returned a 2D array!
    print (new_data.shape)
    
    # However, going back to 3D is easy if we know the 
    # original shape of the array
    new_data = new_data.reshape((-1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    return new_data

def load_by_trial(trial_id,sensor):
    path = '../segmented/trial'+str(trial_id)+'/*_'+sensor+'_test_data.txt'
    x_list = glob.glob(path)
    path = '../segmented/trial'+str(trial_id)+'/*_'+sensor+'_test_label.txt'
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


x1,y1 = load_by_trial(1, sensor)
x2,y2 = load_by_trial(2, sensor)

X = np.vstack((x1,x2))
X = X.reshape(-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL)
y = np.concatenate((y1,y2))

labels = np.array([0,1,2,3])
lb = preprocessing.LabelBinarizer()
lb.fit(y=labels)

# =============================================================================
# test and save
# =============================================================================
predy = model.predict(X)
y_hat = lb.inverse_transform(predy)
acc = eval_perf(y_hat,y)