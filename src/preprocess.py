# -*- coding: utf-8 -*-
"""
# Created at 8:12 PM 3/14/2021 using PyCharm
trial sensor: all 3
24 hours sensor: wrist and thigh	1
wrist and ankle	2

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


def extract_trial(trial_id,thigh_data,sid,timetable,plot=False):
    res = []
    actL = ['Lying', 'Sitting', 'Standing', 'TUG', 'Walking']
    tag1 = ['LDown_1_Initial','LDown_1_Final','Sitting_1_Initial','Sitting_1_Final',
        'Standing_1_Initial','Standing_1_Final','TUG_1_Initial','TUG_1_END','30Predicted_Started_Time1','30Predicted_End_Time1']
    tag2 = ['LDown_2_Initial','LDown_2_Final','Sitting_2_Initial','Sitting_2_Final',
        'Standing_2_Initial','Standing_2_Final','TUG_2_Initial','TUG_2_END','30Predicted_Started_Time24','30Predicted_End_Time24']
    tug_tag1 = ['TUG_1_Tcom_MIN','TUG_1_Tcom_SEC','TUG_1_Tcom_MSEC']
    tug_tag2 = ['TUG_2_Tcom_MIN','TUG_2_Tcom_SEC','TUG_2_Tcom_MSEC']
            
    for aid in range(len(actL)):
        date1 = timetable['Date_1'][sid]
        date2 = timetable['Date_2'][sid]
        if aid==2:
            continue
        if trial_id == 1:
           date = date1
           tag = tag1
           tug_tag = tug_tag1
        else:
           date = date2
           tag = tag2
           tug_tag = tug_tag2
           
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
            min_ = timetable[tug_tag[0]][sid]
            #sec_ = timetable[tug_tag[1]][sid] + timetable[tug_tag[2]][sid]/1000
            sec_ = timetable[tug_tag[1]][sid] +1.0
            lend = datetime.datetime.combine(date,lend)+ datetime.timedelta(minutes=np.float(min_),seconds=sec_)
            lend = lend.strftime('%H:%M:%S')
            print(lstr,lend)
        else:    
            lend = timetable[tag[idx2]][sid].strftime('%H:%M:%S')
        t1 = thigh_data.iloc[:,0]
        date = date.strftime('%Y-%m-%d')
        lstr = date +' '+ lstr
        lend = date +' '+ lend
        ldown = thigh_data[(t1.str[:19]>=lstr) & (t1.str[:19]<=lend)]
        act_data = ldown.iloc[:,1:].values
        #if plot and aid==3: #only plot tug
        tools.plot_acti(act_data, actL[aid])
        act_data = label_data(act_data, aid)
        res.append(act_data)
    return res


# =============================================================================
# process with 2 trials
# =============================================================================
trial_id = 2 #there are 2 trials before and after the 24 hour-collection
abnormal_sid = [2,4,6,10,12,14,19,22,24,26,27,29] #those subid have 2 separate thigh files, single ankle file
timetable = pd.read_excel('../latest version/timetable.xlsx')
# sid = 8
# sensor='ankle'
# if sensor == 'ankle':
#     data1,data24 = load_data(sid, sensor)
#     if trial_id == 1:
#         s1 = extract_trial(trial_id, data1, sid)
#         data = tools.convert_list(s1)
#     else:
#         s1 = extract_trial(trial_id, data24, sid)
#         data = tools.convert_list(s1)
# else:
#     data = load_data(sid, sensor)
#     s1 = extract_trial(trial_id, data, sid)
#     data = tools.convert_list(s1)
# save_path = '../processed/trial'+str(trial_id)+'/sub'+str(sid)+'_'+sensor+'.txt'
# np.savetxt(save_path,data)
        
#only plot for the abnormal sensor
for sid in range(18,19):
    if sid in abnormal_sid:
        for sensor in sensors:
            if sensor == 'thigh':
                data1,data24 = load_abnormal_data(sid, sensor)
                if trial_id == 1:
                    s1 = extract_trial(trial_id, data1, sid, timetable, True)
                    data = tools.convert_list(s1)
                else:
                    s1 = extract_trial(trial_id, data24, sid,timetable,True)
                    data = tools.convert_list(s1)
            else:
                data = load_abnormal_data(sid, sensor)
                s1 = extract_trial(trial_id, data, sid,timetable)
                data = tools.convert_list(s1)
            save_path = '../processed/trial'+str(trial_id)+'/sub'+str(sid)+'_'+sensor+'.txt'
            #np.savetxt(save_path,data)
    else:
        for sensor in sensors:
            if sensor == 'ankle':
                data1,data24 = load_data(sid, sensor)
                if trial_id == 1:
                    s1 = extract_trial(trial_id, data1, sid,timetable,True)
                    data = tools.convert_list(s1)
                else:
                    s1 = extract_trial(trial_id, data24, sid,timetable,True)
                    data = tools.convert_list(s1)
            else:
                data = load_data(sid, sensor)
                s1 = extract_trial(trial_id, data, sid, timetable,True)
                data = tools.convert_list(s1)
            save_path = '../processed/trial'+str(trial_id)+'/sub'+str(sid)+'_'+sensor+'.txt'
            #np.savetxt(save_path,data)

# =============================================================================
# process with 24 hour chunk
# =============================================================================
sid = 0
combine_dev = timetable['24Hours_ldevice_location'][sid]
