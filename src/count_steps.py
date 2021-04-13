# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 16:07:23 2021
step counting
@author: hao_y
"""

#count steps throughout the collected data
import peakutils
import collections
from tools import select_by_id,cal_acc,remove_noise,normalize_data
import numpy as np
import pandas as pd

def load_by_sensor(trial_id,sid,sensor):
    path = '../processed/trial'+str(trial_id)+'/sub'+str(sid)+'_'+sensor+'.txt'
    return np.loadtxt(path)

timetable = pd.read_excel('../latest version/timetable.xlsx')
steps1 = timetable['StepCount_1'].values
steps2 = timetable['StepCount_2'].values
#steps2 = np.delete(steps2,23) #sub23 lost
gt_step = np.concatenate((steps1,steps2))
fs = 30

def err(steps,steps1):
    return np.abs(steps - steps1.reshape(-1,1))

sensors = ['thigh','wrist','ankle']
def steps_for_sub(trial_id,sid,sensor):
    steps = []
    data = load_by_sensor(trial_id, sid, sensor)
    #select only walking data
    walk,_ = select_by_id(data, 4)
    walk = remove_noise(walk, 4, fs)
    mag1 = cal_acc(walk)
    
    for i in range(10):
        step_index = peakutils.peak.indexes(mag1, 0.4+0.01*i, 10)
        step_ts = walk[step_index]
        num = len(step_ts)
    #print(step_ts)
    #print('Total step: %d' %len(step_ts))
        steps.append(num)
    return np.array(steps)


def fine_tune(trial_id,sid,sensor):
    steps = steps_for_sub(trial_id, sid,sensor)
    steps = steps.reshape(1,10).T
    if trial_id==1:
        error = err(steps,steps1[sid])
    if trial_id==2:
        error = err(steps,steps2[sid])
    mean_ = np.mean(error,axis=1)
    idx = np.argmin(np.abs(mean_))
    print(idx)
    res = steps[idx]
    return res

#trial_id = 2
trial1_steps = []
sensor = 'ankle'
for trial_id in range(1,3):
    # if trial_id==2:
    #     times = 29
    # else:
    #     times = 30
    for sid in range(30):
        # if trial_id==2 and sid==23:
        #         continue
        trial1_steps.append(fine_tune(trial_id,sid,sensor))   
my_step = np.array(trial1_steps)

# if trial_id==2:
#     steps1 = steps2
error = err(my_step,gt_step)
error_per = error/gt_step.reshape(-1,1)
idx = np.where(error_per>1)
error = np.delete(error,idx[0],axis=0)
error_per = np.delete(error_per,idx[0],axis=0)

def result_(error):
    mean_ = np.mean(error,axis=0)
    std_ = np.std(error,axis=0)
    return mean_,std_

m1,s1 = result_(error)
m1_,s1_= result_(error_per)
