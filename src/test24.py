# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:47:46 2021
test on whole 24 hour trial
combination1: wrist + thigh
combination2: wrist + ankle
    
plot pie chart for each type of activity
count steps for each trial
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

timetable = pd.read_excel('../latest version/timetable.xlsx')
sid = 26
placement = timetable['24Hours_ldevice_location'][sid]
id_ = timetable['ID'][sid]
id_ = id_[-2:]

if placement == 1: 
    sensor1 = 'wrist'
    sensor2 = 'thigh'
else:
    sensor1 = 'ankle'
    sensor2 = 'wrist'


def load_model(sensor1,sensor2):
    json_file = open('../trained_model/'+sensor1+'_'+sensor2+'/'+sensor1+'_'+sensor2+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('../trained_model/'+sensor1+'_'+sensor2+'/'+sensor1+'_'+sensor2+'.h5')
    print(sensor1+'_'+sensor2+" model Loaded model from disk")
    print(model.summary())
    return model

# def load_model(sensor1,sensor2):
#     json_file = open(sensor1+'_'+sensor2+'.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     model = model_from_json(loaded_model_json)
#     # load weights into new model
#     model.load_weights(sensor1+'_'+sensor2+'.h5')
#     print("Loaded model from disk")
#     print(model.summary())
#     return model

model = load_model(sensor1,sensor2)

# =============================================================================
# load data and test
# divide data into non-overlapping 2s windows
# problem: has to check per second to align 2 sensor's data, super slow
# =============================================================================
import numpy as np
import glob
from tools import convert_list, convert_y, make_context_window,plot_raw,normalize_data,remove_noise
from sklearn import preprocessing
import collections

def load_by_subject(id_,placement,start,end):
    path = '../latest version/RawTimeStamps and 1SEC/'+id_+'_1SEC/'
    path1 = glob.glob(path+'*Wrist_LowFreqRAW.csv')
    if placement == 1:
       path2 = glob.glob(path+'*Thigh_LowFreqRAW.csv')
    else:
       path2 = glob.glob(path+'*Ankle_LowFreqRAW.csv') 
    data1 = pd.read_csv(path1[0],skiprows=10)
    data2 = pd.read_csv(path2[0],skiprows=10)
    #extract by time from data1 and data2
    t1 = data1.iloc[:,0]
    #print(t1.str[:19])
    data1_ = data1[(t1.str[:19]>=start) & (t1.str[:19]<=end)]
    t1 = data2.iloc[:,0]
    data2_ = data2[(t1.str[:19]>=start) & (t1.str[:19]<=end)]
    
    if data1_.shape[0] == data2_.shape[0]:
        #return only acc readings as numpy array
        print(data1_.shape,data2_.shape)
        data1 = data1_.iloc[:,1:].values
        data2 = data2_.iloc[:,1:].values
        data1 = remove_noise(data1,4,30)
        data2 = remove_noise(data2,4,30)
        data1 = normalize_data(data1)
        data2 = normalize_data(data2)
        plot_raw(data1)
        plot_raw(data2)
        return np.hstack((data2, data1))
    else:
        print('data1 and data2 shape not equal!')
        return 0,0


#extract 24 hour data trial by given start-end time
#date1 = timetable['Date_1'][sid].strftime('%Y-%m-%d')
date1 = timetable['Date_1'][sid].strftime('%Y-%m-%d')
date2 = timetable['Date_1'][sid].strftime('%Y-%m-%d')
start = timetable['Time_Started'][sid].strftime('%H:%M:%S')
end = timetable['Time_Ended'][sid].strftime('%H:%M:%S')
start = '18:00:00'
end = '18:30:00'
start = date1 +' '+ start
end = date2+' '+ end

data = load_by_subject(id_,placement,start,end)

# =============================================================================
# load a labeled data trial, try to plot the pie chart
# =============================================================================
# trial_id = 2
# sid =1
# def load_by_sensor(trial_id,sid,sensor):
#     path = '../processed/trial'+str(trial_id)+'/sub'+str(sid)+'_'+sensor+'.txt'
#     return np.loadtxt(path)

# sensors = ['thigh','wrist','ankle']
# data1 = load_by_sensor(trial_id, sid, sensors[0])
# data2 = load_by_sensor(trial_id, sid, sensors[1])

# data1 = normalize_data(data1[:,:3])
# data2 = normalize_data(data2[:,:3])

#divide into non-overlapping windows
SLIDING_WINDOW_LENGTH = 60
NUM_CHANNEL = 6
def get_y(data1,model1):
    X = make_context_window(data1,SLIDING_WINDOW_LENGTH,0)
    #X = make_context_window(x,SLIDING_WINDOW_LENGTH,0)
    print(X.shape)
    X = X.reshape(-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL)
    predy = model1.predict(X)
    return predy
labels = np.array([0,1,2,3])
lb = preprocessing.LabelBinarizer()
lb.fit(y=labels)

def plot_pie(y_hat):
    dic = collections.Counter(y_hat)
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    total = y_hat.shape[0]
    a1 = dic[0]
    a2 = dic[1]
    a3 = dic[2]
    a4 = dic[3]
    
    labels =  ['Lying', 'Sitting', 'Standing', 'Walking']
    sizes = [a1/total, a2/total, a3/total, a4/total]
    explode = (0, 0, 0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
    
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    total = round(total/1800,2)
    ax1.set_title(f"Posture for JHCC02_{id_} in {total} hours")
    plt.show()
    
predy = get_y(data, model)
y_hat = lb.inverse_transform(predy)
plot_pie(y_hat)

#replace y_hat with second highest predy
def find_replace(predy,i):
    k = predy[i]
    ind = np.where(k==np.unique(k)[-2])
    return ind[0]

#fine invalid transitions, replace by the second highest probability
invalid = []
for i in range(1,y_hat.shape[0]):
    if y_hat[i] == 2 and y_hat[i-1] ==0:
        y_hat[i] = find_replace(predy,i)
        invalid.append(i)
    elif y_hat[i] == 3 and y_hat[i-1]==0:
        y_hat[i] = find_replace(predy,i)
        invalid.append(i)
    elif y_hat[i] == 0 and y_hat[i-1]==3:
        y_hat[i] = find_replace(predy,i)
        invalid.append(i)
    elif y_hat[i] == 3 and y_hat[i-1]==1:
        y_hat[i] = find_replace(predy,i)
        invalid.append(i)
    elif y_hat[i] == 0 and y_hat[i-1] ==2:
        y_hat[i] = find_replace(predy,i)
        invalid.append(i)

plot_pie(y_hat)