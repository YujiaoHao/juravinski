# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:53:29 2021
test 24 with a single sensor
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
plt.rcParams.update({'font.size': 17})
timetable = pd.read_excel('../latest version/timetable.xlsx')
sid = 28
placement = timetable['24Hours_ldevice_location'][sid]
id_ = timetable['ID'][sid]
id_ = id_[-2:]

if placement == 1: 
    sensor1 = 'wrist'
    sensor2 = 'thigh'
else:
    sensor1 = 'ankle'
    sensor2 = 'wrist'


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


model1 = load_model(sensor1)
model2 = load_model(sensor2)
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

def load_by_subject(id_,sensor,start,end):
    path = '../latest version/RawTimeStamps and 1SEC/'+id_+'_1SEC/'
    path1 = glob.glob(path+'*'+sensor+'_LowFreqRAW.csv')
    data1 = pd.read_csv(path1[0],skiprows=10)
    #extract by time from data1 and data2
    t1 = data1.iloc[:,0]
    #print(t1.str[:19])
    data1_ = data1[(t1.str[:19]>=start) & (t1.str[:19]<=end)]
    data1 = data1_.iloc[:,1:].values
       
    data1 = remove_noise(data1,5,30)
    data1 = normalize_data(data1[10:-10])
       
    plot_raw(data1)
    return data1
  


#extract 24 hour data trial by given start-end time
date1 = timetable['Date_2'][sid].strftime('%Y-%m-%d')
#date1 = timetable['Date_2'][sid].strftime('%Y-%m-%d')
date2 = timetable['Date_2'][sid].strftime('%Y-%m-%d')
start = timetable['Time_Started'][sid].strftime('%H:%M:%S')
end = timetable['Time_Ended'][sid].strftime('%H:%M:%S')
start = '02:00:00'
end = '02:30:00'
start = date1 +' '+ start
end = date2+' '+ end

data1 = load_by_subject(id_,sensor1,start,end)
data2 = load_by_subject(id_,sensor2,start,end)

# =============================================================================
# load a labeled data trial, try to plot the pie chart
# =============================================================================
#divide into non-overlapping windows
SLIDING_WINDOW_LENGTH = 60
NUM_CHANNEL = 3
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

def fixOverLappingText(text):

    # if undetected overlaps reduce sigFigures to 1
    sigFigures = 2
    positions = [(round(item.get_position()[1],sigFigures), item) for item in text]

    overLapping = collections.Counter((item[0] for item in positions))
    overLapping = [key for key, value in overLapping.items() if value >= 2]

    for key in overLapping:
        textObjects = [text for position, text in positions if position == key]

        if textObjects:

            # If bigger font size scale will need increasing
            scale = 0.05

            spacings = np.linspace(0,scale*len(textObjects),len(textObjects))

            for shift, textObject in zip(spacings,textObjects):
                textObject.set_y(key + shift)
                
def plot_pie(y_hat):
    dic = collections.Counter(y_hat)
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    total = y_hat.shape[0]
    labels =  ['Lying', 'Sitting', 'Standing', 'Walking']
    lb,sz = [],[]
    for i in range(4):
        if dic[i] == 0:
            continue
        else:
            lb.append(labels[i])
            sz.append(dic[i]/total)
   
    #explode = (0,0, 0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
    
    fig1, ax1 = plt.subplots()
    # text = ax1.pie(sz,  labels=lb, autopct='%1.1f%%',
    #         shadow=True, startangle=90)
    text = ax1.pie(sz,autopct='%1.1f%%',
            shadow=True, startangle=90)
    fixOverLappingText(text[1])
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    total = round(total/1800,2)   #predict per second
    ax1.set_title(f"Posture for JHCC02_{id_} in {total} hours",y=1.08)
    plt.legend(text[0], lb, loc="lower right")
    fig1.tight_layout()
    plt.show()
# =============================================================================
# make test
# generate pie chart
# count steps from walking activity where aid=3
# ========================================================================
predy = get_y(data1, model1)
y_hat = lb.inverse_transform(predy)
plot_pie(y_hat)
predy = get_y(data2, model2)
y_hat = lb.inverse_transform(predy)
plot_pie(y_hat)

# ts = np.arange(0,y_hat.shape[0],step=1)
# plt.plot(ts,y_hat)
# yint = np.arange(min(y_hat), np.ceil(max(y_hat))+1)
# plt.yticks(yint)
# plt.xlabel('unit (2s)')
# plt.ylabel('Pose')
# plt.show()

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

#plot_pie(y_hat)
            
    