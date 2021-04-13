# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:31:20 2021
manually select and label data without the given timetable
for normal cases, select by ankle trials; for abnormal cases, select by thigh trials.
doesn't work, too nasty
@author: hao_y
"""

import pandas as pd
import numpy as np
import glob
import tools
import datetime
from matplotlib.widgets import SpanSelector
import matplotlib.pyplot as plt

fs = 30
def onselect(xmin, xmax):
    return xmin,xmax

sensors = ['thigh','wrist','ankle']
def load_data(sid,sensor):
    path = '../latest version/RawTimeStamps and 1SEC/*_1SEC/'
    sublist = glob.glob(path)
    sub_path = sublist[sid]
    
    if sensor == sensors[0]:
        thigh_data_path = sub_path + '*Thigh_LowFreqRAW.csv'
        thigh_path = glob.glob(thigh_data_path)
        print(thigh_path)
        thigh_data = pd.read_csv(thigh_path[0],skiprows=10)
        print(thigh_data.shape)
        return thigh_data
    elif sensor == sensors[1]:
        wrist_data_path = sub_path + '*Wrist_LowFreqRAW.csv'
        wrist_path = glob.glob(wrist_data_path)
        print(wrist_path)
        wrist_data = pd.read_csv(wrist_path[0],skiprows=10)
        print(wrist_data.shape)
        return wrist_data
    elif sensor == sensors[2]:
        ankle_data_path_1 = sub_path + '*test1_Ankle_LowFreqRAW.csv'
        ankle_data_path_24 = sub_path + '*test24_Ankle_LowFreqRAW.csv'
        ankle_path1 = glob.glob(ankle_data_path_1)
        ankle_path24 = glob.glob(ankle_data_path_24)
        ankle_data1 = pd.read_csv(ankle_path1[0],skiprows=10)
        ankle_data24 = pd.read_csv(ankle_path24[0],skiprows=10)
        print(ankle_data1.shape)
        print(ankle_data24.shape)
        return ankle_data1,ankle_data24

def load_abnormal_data(sid,sensor):
    path = '../latest version/RawTimeStamps and 1SEC/*_1SEC/'
    sublist = glob.glob(path)
    sub_path = sublist[sid]
    
    if sensor == sensors[0]:
        ankle_data_path_1 = sub_path + '*test1_Thigh_LowFreqRAW.csv'
        ankle_data_path_24 = sub_path + '*test24_Thigh_LowFreqRAW.csv'
        ankle_path1 = glob.glob(ankle_data_path_1)
        ankle_path24 = glob.glob(ankle_data_path_24)
        ankle_data1 = pd.read_csv(ankle_path1[0],skiprows=10)
        ankle_data24 = pd.read_csv(ankle_path24[0],skiprows=10)
        print(ankle_data1.shape)
        print(ankle_data24.shape)
        return ankle_data1,ankle_data24
    elif sensor == sensors[1]:
        wrist_data_path = sub_path + '*Wrist_LowFreqRAW.csv'
        wrist_path = glob.glob(wrist_data_path)
        print(wrist_path)
        wrist_data = pd.read_csv(wrist_path[0],skiprows=10)
        print(wrist_data.shape)
        return wrist_data
    elif sensor == sensors[2]:
        thigh_data_path = sub_path + '*Ankle_LowFreqRAW.csv'
        thigh_path = glob.glob(thigh_data_path)
        print(thigh_path)
        thigh_data = pd.read_csv(thigh_path[0],skiprows=10)
        print(thigh_data.shape)
        return thigh_data




# 2 label the data
def label_data(data, aid):
    m = data.shape[0]
    ones = np.ones((m, 1))
    return np.hstack((data, aid * ones))


def extract_trial(trial_id,thigh_data,sid,plot=False):
    res = []
    timetable = pd.read_excel('../latest version/timetable.xlsx')
    actL = ['Lying', 'Sitting', 'Standing', 'TUG', 'Walking']
    tag1 = ['LDown_1_Initial','LDown_1_Final','Sitting_1_Initial','Sitting_1_Final',
        'Standing_1_Initial','Standing_1_Final','TUG_1_Initial','TUG_1_END','30MWT_1_Started','30MWT_1_Completed']
    tag2 = ['LDown_2_Initial','LDown_2_Final','Sitting_2_Initial','Sitting_2_Final',
        'Standing_2_Initial','Standing_2_Final','TUG_2_Initial','TUG_2_END','30MWT_2_Started','30MWT_2_Completed']
    
            
    for aid in range(len(actL)):
        date1 = timetable['Date_1'][sid]
        date2 = timetable['Date_2'][sid]
       
        if trial_id == 1:
           date = date1
           tag = tag1
        else:
           date = date2
           tag = tag2
           
        if aid ==3:
            idx1 = aid*2 #TUG, start and end in the same min
            idx2 = idx1
        else:
            idx1 = aid*2  #start time index in tag1 list
            idx2 = aid*2 + 1 #end time index in tag1 list
        
        #subid = timetable.iloc[sid, 0]
        lstr = timetable[tag[idx1]][sid].strftime('%H:%M:%S')
        if aid ==3:
            lend = timetable[tag[idx2]][sid]
            lend = datetime.datetime.combine(date,lend)+ datetime.timedelta(seconds=90)
            lend = lend.strftime('%H:%M:%S')
        elif aid==4:
            lend = timetable[tag[idx2]][sid]
            lend = datetime.datetime.combine(date,lend)+ datetime.timedelta(seconds=180)
            lend = lend.strftime('%H:%M:%S')
           
        else:    
            lend = timetable[tag[idx2]][sid].strftime('%H:%M:%S')
        t1 = thigh_data.iloc[:,0]
        date = date.strftime('%Y-%m-%d')
        lstr = date +' '+ lstr
        lend = date +' '+ lend
        ldown = thigh_data[(t1.str[:19]>=lstr) & (t1.str[:19]<=lend)]
        act_data = ldown.iloc[:,1:].values
        if plot:
            tools.plot_acti(act_data, actL[aid])
        act_data = label_data(act_data, aid)
        res.append(act_data)
    return res

trial_id = 1 #there are 2 trials before and after the 24 hour-collection
abnormal_sid = [2,4,6,10,12,14,19,22,24,26,27,29] #those subid have 2 separate thigh files, single ankle file
        
sid = 2
sensor = 'thigh'
if sid in abnormal_sid:
    #for sensor in sensors:
        if sensor == 'thigh':
            data1,data24 = load_abnormal_data(sid, sensor)
            data = data1.iloc[:,1:].values
            fig, ax = plt.subplots()
            ax.plot(np.arange(0,data.shape[0],step=1),data[:,0])
             # Let user to select the target signal (synchronization action data) to calculate synchronization time offset
            span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='red'))
            plt.show()
                
                




