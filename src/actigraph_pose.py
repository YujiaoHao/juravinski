# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:20:03 2021
select a time-span, plot the activity pie chart estimated by actigraph
@author: hao_y
"""

import pandas as pd
import numpy as np
import glob
import tools
import datetime

fs = 30

sensors = ['thigh','wrist','ankle']
def load_data(sid,sensor):
    path = '../latest version/RawTimeStamps and 1SEC/*_1SEC/'
    sublist = glob.glob(path)
    sub_path = sublist[sid]
    
    if sensor == sensors[0]:
        thigh_data_path = sub_path + '*Thigh_LowFreq1sec.csv'
        thigh_path = glob.glob(thigh_data_path)
        print(thigh_path)
        thigh_data = pd.read_csv(thigh_path[0],skiprows=10)
        print(thigh_data.shape)
        return thigh_data
    elif sensor == sensors[1]:
        wrist_data_path = sub_path + '*Wrist_LowFreq1sec.csv'
        wrist_path = glob.glob(wrist_data_path)
        print(wrist_path)
        wrist_data = pd.read_csv(wrist_path[0],skiprows=10)
        print(wrist_data.shape)
        return wrist_data
    elif sensor == sensors[2]:
        ankle_data_path_1 = sub_path + '*test1_Ankle_LowFreq1sec.csv'
        ankle_data_path_24 = sub_path + '*test24_Ankle_LowFreq1sec.csv'
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
        ankle_data_path_1 = sub_path + '*test1_Thigh_LowFreq1sec.csv'
        ankle_data_path_24 = sub_path + '*test24_Thigh_LowFreq1sec.csv'
        ankle_path1 = glob.glob(ankle_data_path_1)
        ankle_path24 = glob.glob(ankle_data_path_24)
        ankle_data1 = pd.read_csv(ankle_path1[0],skiprows=10)
        ankle_data24 = pd.read_csv(ankle_path24[0],skiprows=10)
        print(ankle_data1.shape)
        print(ankle_data24.shape)
        return ankle_data1,ankle_data24
    elif sensor == sensors[1]:
        wrist_data_path = sub_path + '*Wrist_LowFreq1sec.csv'
        wrist_path = glob.glob(wrist_data_path)
        print(wrist_path)
        wrist_data = pd.read_csv(wrist_path[0],skiprows=10)
        print(wrist_data.shape)
        return wrist_data
    elif sensor == sensors[2]:
        thigh_data_path = sub_path + '*Ankle_LowFreq1sec.csv'
        thigh_path = glob.glob(thigh_data_path)
        print(thigh_path)
        thigh_data = pd.read_csv(thigh_path[0],skiprows=10)
        print(thigh_data.shape)
        return thigh_data
    

    
# =============================================================================
# load date from timetable, set str&end time manually
# =============================================================================
def get_by_time(timetable,sid,start,end):
    placement = timetable['24Hours_ldevice_location'][sid]
    id_ = timetable['ID'][sid]
    id_ = id_[-2:]
    
    if placement == 1: 
        sensor1 = 'wrist'
        sensor2 = 'thigh'
    else:
        sensor1 = 'ankle'
        sensor2 = 'wrist'
    abnormal_sid = [2,4,6,10,12,14,19,22,24,26,27,29] #those subid have 2 separate thigh files, single ankle file
    if sid in abnormal_sid:  
        data1 = load_abnormal_data(sid,sensor1)
        data2 = load_abnormal_data(sid,sensor2)
    else:
        data2 = load_data(sid, sensor2)
        data1 = load_data(sid, sensor1)
        
    d = data2.iloc[:,0]
    t1 = data1.iloc[:,1]
    data1_ = data1[(t1>=start) & (t1<=end) & (d==date1)]
    t2 = data2.iloc[:,1]
    data2_ = data2[(t2>=start) & (t2<=end) & (d==date1)]
    return data1_,data2_

timetable = pd.read_excel('../latest version/timetable.xlsx')
sid = 28
id_ = timetable['ID'][sid]

#load the active time data
date1 = timetable['Date_1'][sid].strftime('%Y-%m-%d')
start = '17:00:00'
end = '17:30:00'
d1,d2 = get_by_time(timetable,sid,start,end)

#load the inactive time data
date1 = timetable['Date_2'][sid].strftime('%Y-%m-%d')
start = '02:00:00'
end = '02:30:00'
d3,d4 = get_by_time(timetable,sid,start,end)

# =============================================================================
# check predictions from ActiGraph, plot pie charts
# =============================================================================
import collections
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 17})
def get_pose(placement, d1,d2):
    thigh_label = ['Inclinometer Standing',	'Inclinometer Stepping','Inclinometer Sitting/Lying']
    wrist_label = ['Inclinometer Standing','Inclinometer Sitting','Inclinometer Lying']
    ankle_label = ['Inclinometer Standing','Inclinometer Sitting','Inclinometer Lying']
    if placement == 1:
        d1_1 = d1[wrist_label[0]].values
        d1_2 = d1[wrist_label[1]].values
        d1_3 = d1[wrist_label[2]].values
        d2_1 = d2[thigh_label[0]].values
        d2_2 = d2[thigh_label[1]].values
        d2_3 = d2[thigh_label[2]].values
    else:
        d1_1 = d1[ankle_label[0]].values
        d1_2 = d1[ankle_label[1]].values
        d1_3 = d1[ankle_label[2]].values
        d2_1 = d2[wrist_label[0]].values
        d2_2 = d2[wrist_label[1]].values
        d2_3 = d2[wrist_label[2]].values
    return d1_1*1+d1_2*2+d1_3*3, d2_1*1+d2_2*2+d2_3*3
    
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

def plot_pie_ankle(y_hat):
    dic = collections.Counter(y_hat)
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    total = y_hat.shape[0]
    labels =  ['Unknown','Standing','Sitting','Lying']
    lb,sz = [],[]
    for i in range(4):
        if dic[i] == 0:
            continue
        else:
            lb.append(labels[i])
            sz.append(dic[i]/total)
   
    #explode = (0,0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    
    fig1, ax1 = plt.subplots()
    # text = ax1.pie(sz,  labels=lb, autopct='%1.1f%%',
    #         shadow=True, startangle=90)
    text = ax1.pie(sz,autopct='%1.1f%%',
            shadow=True, startangle=90)
    fixOverLappingText(text[1])
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    total = round(total/3600,2)   #predict per second
    ax1.set_title(f"Posture for JHCC02_{id_} in {total} hours",y=1.08)
    plt.legend(text[0], lb, loc="best")
    fig1.tight_layout()
    plt.show()
    
def plot_pie_thigh(y_hat):
    dic = collections.Counter(y_hat)
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    total = y_hat.shape[0]
    labels =  ['Unknown','Standing', 'Walking','Lying/Sitting']
    lb,sz = [],[]
    for i in range(4):
        if dic[i] == 0:
            continue
        else:
            lb.append(labels[i])
            sz.append(dic[i]/total)
   
    #explode = (0,0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    
    fig1, ax1 = plt.subplots()
    # text = ax1.pie(sz,  labels=lb, autopct='%1.1f%%',
    #         shadow=True, startangle=90)
    text = ax1.pie(sz,autopct='%1.1f%%',
            shadow=True, startangle=90)
    fixOverLappingText(text[1])
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    total = round(total/3600,2)   #predict per second
    ax1.set_title(f"Posture for JHCC02_{id_} in {total} hours",y=1.08)
    plt.legend(text[0], lb, loc="upper right")
    fig1.tight_layout()
    plt.show()

placement = timetable['24Hours_ldevice_location'][sid]
d1_,d2_ = get_pose(placement, d1, d2)
d3_,d4_ = get_pose(placement, d3, d4)

def final_plot(d1_,d2_):
    if placement ==1:
        plot_pie_ankle(d1_)
        plot_pie_thigh(d2_)
    else:
        plot_pie_ankle(d1_)
        plot_pie_ankle(d2_)

final_plot(d1_, d2_) #plot for the afternoon
final_plot(d3_, d4_) #plot for the night