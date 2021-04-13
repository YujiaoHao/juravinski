# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 13:55:38 2021
refine by manually select
@author: hao_y
"""


import numpy as np
from tools import select_by_id, plot_acti,crop_out_data,magnitude,plot_raw
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

trial_id =2
sid=29

def load_by_sensor(trial_id,sid,sensor):
    path = '../processed/trial'+str(trial_id)+'/sub'+str(sid)+'_'+sensor+'.txt'
    return np.loadtxt(path)

data1 = load_by_sensor(trial_id,sid,'ankle')
data2 = load_by_sensor(trial_id,sid,'thigh')
data3 = load_by_sensor(trial_id,sid,'wrist')

tug1,idx_tug1 = select_by_id(data1, 3) #the index of TUG are the same for 3 sensors, take 1 is enough
tug2,_ = select_by_id(data2, 3)
tug3,_ = select_by_id(data3, 3)

    
# =============================================================================
# Plot with magnitude and allow select with mouse 
# =============================================================================
def cal_acc(acc):
    m = acc.shape[0]
    res = np.zeros(m)
    for i in range(m):
        res[i] = np.sqrt(acc[i,0]**2+acc[i,1]**2+acc[i,2]**2)
    return res


plot_raw(tug2)
mag1 = cal_acc(tug2)

###################plot a selectable fig##################################
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(211)

x = np.arange(0,mag1.shape[0],step=1)
y = tug2[:,1]

ax.plot(x, y, '-')
ax.set_title('Press left mouse button and drag to test')

ax2 = fig.add_subplot(212)
line2, = ax2.plot(x, y, '-')
    
def onselect(xmin, xmax):
    indmin, indmax = np.searchsorted(x, (xmin, xmax))
    indmax = min(len(x) - 1, indmax)

    thisx = x[indmin:indmax]
    thisy = y[indmin:indmax]
    line2.set_data(thisx, thisy)
    ax2.set_xlim(thisx[0], thisx[-1])
    ax2.set_ylim(thisy.min(), thisy.max())
    fig.canvas.draw_idle()

    # save
    np.savetxt("text.txt", np.c_[thisx, thisy])

# set useblit True on gtkagg for enhanced performance
span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='red'))

plt.show()   

idx_ = np.loadtxt("text.txt")
idx = np.floor(idx_[:,0])
print(idx.shape)

def replace_save(tug1,idx,data1,idx_tug1,trial_id=trial_id,sid=sid,sensor='ankle'):
    idx1 = int(idx[0])
    idx2 = int(idx[-1])
    #replace and save back
    tug1 = tug1[idx1:idx2]
    plot_acti(tug1, 'TUG')
    data1 = np.delete(data1,idx_tug1,axis=0)
    data1 = np.vstack((data1,tug1))
    print(data1.shape)
    # save_path = '../processed/trial'+str(trial_id)+'/sub'+str(sid)+'_'+sensor+'.txt'
    # np.savetxt(save_path,data1)
    # print('data saved!')
replace_save(tug1, idx, data1, idx_tug1,sensor='ankle')
replace_save(tug2, idx, data2, idx_tug1,sensor='thigh')
replace_save(tug3, idx, data3, idx_tug1,sensor='wrist')

