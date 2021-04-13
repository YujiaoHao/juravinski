# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:10:09 2021
refine the extracted TUG data
@author: hao_y
"""

import numpy as np
from tools import select_by_id, plot_acti,crop_out_data,magnitude,plot_raw
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

trial_id =1
sid=21

def load_by_sensor(trial_id,sid,sensor):
    path = '../processed/trial'+str(trial_id)+'/sub'+str(sid)+'_'+sensor+'.txt'
    return np.loadtxt(path)

data1 = load_by_sensor(trial_id,sid,'ankle')
data2 = load_by_sensor(trial_id,sid,'thigh')
data3 = load_by_sensor(trial_id,sid,'wrist')


def replace_save(tug1,idx,data1,idx_tug1,trial_id=1,sid=sid,sensor='ankle'):
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

def refine(data):
    idx = []
    for i in range(1,data.shape[0]-1):
        if data[i,1]-data[i-1,1]>0 and data[i+1,1]-data[i,1]>0:
            idx.append(i-1)
    return idx

tug1,idx_tug1 = select_by_id(data1, 3) #the index of TUG are the same for 3 sensors, take 1 is enough
tug2,_ = select_by_id(data2, 3)
tug3,_ = select_by_id(data3, 3)
# idx = crop_out_data(tug1,tug2,tug3,0.015)
# idx2 = refine(tug1)
# #idx = [idx[0],idx[-1]-100]
# idx = [82,idx[-1]]
# replace_save(tug1, idx, data1, idx_tug1,sensor='ankle')
# replace_save(tug2, idx, data2, idx_tug1,sensor='thigh')
# replace_save(tug3, idx, data3, idx_tug1,sensor='wrist')

# idx = np.array(idx)
# idx_diff = idx[1:]-idx[:-1]
# t = np.where(idx_diff>100)[0]
# if not t.size > 0:
#   print("List is empty")
# else:
#     tug1 = tug1[:idx[t[-1]]]
#     tug2 = tug2[:idx[t[-1]]]
#     tug3 = tug3[:idx[t[-1]]]
#     idx = crop_out_data(tug1,tug2,tug3,0.1)
#     #idx = [idx[0]+400,idx[-1]]    
#     replace_save(tug1, idx, data1, idx_tug1,sensor='ankle')
#     replace_save(tug2, idx, data2, idx_tug1,sensor='thigh')
#     replace_save(tug3, idx, data3, idx_tug1,sensor='wrist')   
    
# =============================================================================
# Plot with magnitude and allow select with mouse 
# =============================================================================
def cal_acc(acc):
    m = acc.shape[0]
    res = np.zeros(m)
    for i in range(m):
        res[i] = np.sqrt(acc[i,0]**2+acc[i,1]**2+acc[i,2]**2)
    return res


plot_raw(tug1)
#plot_raw(data1)
mag1 = cal_acc(tug1)

###################plot a selectable fig##################################
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(211)

x = np.arange(0,mag1.shape[0],step=1)
y = mag1

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
replace_save(tug1, idx, data1, idx_tug1,sensor='ankle')
replace_save(tug2, idx, data2, idx_tug1,sensor='thigh')
replace_save(tug3, idx, data3, idx_tug1,sensor='wrist')
# nmdata1 = normalize_data(data1)
# plot_raw(nmdata1)
