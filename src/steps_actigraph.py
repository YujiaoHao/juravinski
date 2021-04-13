# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 10:47:35 2021
compare the steps counted by physio and actigraph
@author: hao_y
"""

import pandas as pd
import numpy as np

titles = ['Physio_StepCount_1','Wrist_StepCount_Test1','Thigh_StepCount_Test1',
          'Ankle_StepCount_Test1','Phyiso_StepCount_24','Wrist_StepCount_Test24',
          'Thigh_StepCount_Test24','Ankle_StepCount_Test24']

df = pd.read_excel('../latest version/steps.xlsx')
gt1 = df[titles[0]].values
w1 = df[titles[1]].values
t1 = df[titles[2]].values
a1 = df[titles[3]].values

gt2 = df[titles[4]].values
gt2 = np.delete(gt2,23,axis=0)
w2 = df[titles[5]].values
w2 = np.delete(w2,23,axis=0)
t2 = df[titles[6]].values
t2 = np.delete(t2,23,axis=0)
a2 = df[titles[7]].values
a2 = np.delete(a2,23,axis=0)

#sub23 has no wrist, thigh step count in trial2
gt = np.concatenate((gt1,gt2))
w = np.concatenate((w1,w2))
t = np.concatenate((t1,t2))
a = np.concatenate((a1,a2))

def err(steps,steps1):
    return np.abs(steps.reshape(-1,1) - steps1.reshape(-1,1))

def result_(error):
    mean_ = np.mean(error,axis=0)
    std_ = np.std(error,axis=0)
    return mean_,std_

def evaluate(w,gt):
    err_w = err(w,gt)
    errw_per = err_w/gt.reshape(-1,1)
    mw,sw = result_(err_w)
    mwp,swp = result_(errw_per)
    print(mw,sw)
    print(mwp,swp)
    
evaluate(t,gt)
evaluate(a, gt)