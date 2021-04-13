# -*- coding: utf-8 -*-
"""
# Created at 8:13 PM 3/14/2021 using PyCharm

@author: hao_y
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

def plot_acti(rawdata,name):
    fig, ax = plt.subplots(3, 1)
    plt.title(name)
    ts = np.arange(0,rawdata.shape[0],step=1)
    ax[0].plot(ts, rawdata[:, 0], 'k-')
    ax[1].plot(ts, rawdata[:, 1], 'r-')
    ax[2].plot(ts, rawdata[:, 2], 'g-')
    plt.show()
    
def convert_list(my_list):
    length = len(my_list)
    if length == 0:
        return
    res = my_list[0]
    for i in range(1, length):
        res = np.vstack((res, my_list[i]))
    return res

def convert_y(my_list):
    length = len(my_list)
    if length==0:
        return
    res = my_list[0]
    for i in range(1,length):
        res = np.concatenate((res,my_list[i]))
    return res

from scipy import signal
def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, np.ravel(data))
    return y

def remove_noise(data, fc_lpf,fs):
    """Remove noise from accelerometer data via low pass filter
    INPUT: 
        data -- input accelerometer data Nx3 (x, y, z axis)
        fc_lpf -- low pass filter cutoff frequency
    OUTPUT: 
        lpf_output -- filtered accelerometer data Nx3 (x, y, z axis)
    """

    # the number of accelerometer readings
    num_data = data[:,0].shape[0]
    # lpf_output is used to store the filtered accelerometer data by low pass filter
    lpf_output = np.zeros((num_data, 3))

    # compute linear acceleration for x axis
    acc_X = data[:,0]
    butterfilter_output= butter_lowpass_filter(acc_X, fc_lpf, fs/2)
    lpf_output[:,0] = butterfilter_output.reshape(1, num_data)

    # compute linear acceleration for y axis
    acc_Y = data[:,1]
    butterfilter_output= butter_lowpass_filter(acc_Y, fc_lpf, fs/2)
    lpf_output[:,1] = butterfilter_output.reshape(1, num_data)

    # compute linear acceleration for z axis
    acc_Z = data[:,2]
    butterfilter_output= butter_lowpass_filter(acc_Z, fc_lpf, fs/2)
    lpf_output[:,2] = butterfilter_output.reshape(1, num_data)

    return lpf_output
def magnitude(A,B,C):
    return np.sqrt(A**2 + B**2 + C**2)

def crop_out_data(data_input1,
                  data_input2,
                  data_input3,thres=0.01):
    idx = []
    for i in range(data_input1.shape[0]-1):
        mag1 = magnitude(data_input1[i,0], data_input1[i,1], data_input1[i,2])
        mag2 = magnitude(data_input2[i,0], data_input2[i,1], data_input2[i,2])
        mag3 = magnitude(data_input3[i,0], data_input3[i,1], data_input3[i,2])
        
        mag11 = magnitude(data_input1[i+1,0], data_input1[i+1,1], data_input1[i+1,2])
        mag21 = magnitude(data_input2[i+1,0], data_input2[i+1,1], data_input2[i+1,2])
        mag31 = magnitude(data_input3[i+1,0], data_input3[i+1,1], data_input3[i+1,2])
        # if mag1>thres and mag2>thres and mag3>thres:
        #     idx.append(i)
        # if np.abs(mag1-mag11)>thres and np.abs(mag2-mag21)>thres and np.abs(mag3-mag31)>thres:
        #     idx.append(i)
        # if np.abs(mag1-mag11)>thres and np.abs(mag2-mag21)>thres: #check thigh and ankle sensor only, wrist is irrelavant
        #     idx.append(i)
        if np.abs(mag2-mag21)>thres: #check thigh only
            idx.append(i)
    return idx

def select_by_id(data,aid):
    idx1 = np.where(data[:,3]==aid)
    return data[idx1],idx1

def cal_acc(acc):
    m = acc.shape[0]
    res = np.zeros(m)
    for i in range(m):
        res[i] = np.sqrt(acc[i,0]**2+acc[i,1]**2+acc[i,2]**2)
    return res

def normalize_data(data):
    res = normalize(data, axis=0, norm='max')
    return res

# def normalize_data(data):
#     mag1 = cal_acc(data)
#     #normalize by divide the max of mag
#     obx = np.zeros((data.shape[0],data.shape[1]))
#     obx = data/mag1.max()
#     return obx

def make_context_window(X_raw,L,s):
    m,n = X_raw.shape
    res = []
    i = 0
    while i<=m:
        ind1 = i*(L-s)
        ind2 = ind1+L
        if ind1>=m:
            break
        if ind2>m and ind1<m:
            mzeros = np.zeros((ind2-m,n))
            temp = np.vstack((X_raw[ind1:m,:],mzeros))
            res.append(temp)
            break
        res.append(X_raw[ind1:ind2, :])
        i = i+1
    res = np.stack(res, axis=0) 
    return res

def plot_raw(rawdata):
    ts = np.arange(0,rawdata.shape[0],step=1)
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(ts, rawdata[:, 0], 'k-')
    ax[1].plot(ts, rawdata[:, 1], 'r-')
    ax[2].plot(ts, rawdata[:, 2], 'g-')
    plt.show()