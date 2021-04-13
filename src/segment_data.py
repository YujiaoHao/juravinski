# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:47:15 2021
L: 2s with 1.6s overlapping
TUG=3 and walking=4 are inseparable
80% train, 20% test

normalize data by max mag, select act, divide into windows, split 
@author: hao_y
"""
import numpy as np
from tools import normalize_data,plot_acti,make_context_window,select_by_id, convert_list, convert_y
from sklearn.model_selection import train_test_split

L = 60
s = 48

def label_per_window(X):
    m,n,o = X.shape
    mones = np.ones((n,1))
    for i in range(m):
        temp = X[i,:,12:13]
        unique, counts = np.unique(temp, return_counts=True)
        label = unique[counts.argmax(axis=0)]*mones
        X[i,:,:] = np.hstack((X[i,:,:12],label))
    return X

def load_by_sensor(trial_id,sid,sensor):
    path = '../processed/trial'+str(trial_id)+'/sub'+str(sid)+'_'+sensor+'.txt'
    return np.loadtxt(path)

# Save processed windowed data
def save_tensor(data,sid,sensor,name):
    # Write the array to disk
    with open("../segmented/trial1/Sub"+str(sid)+"_"+sensor+"_"+name+"_data.txt", 'w') as outfile:
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(data.shape))
    
        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in data:
    
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.  
            np.savetxt(outfile, data_slice, fmt='%-2.7f')
    
            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')

trial_id = 1
# sid = 0
# sensor = 'ankle'  
          
def process_data(sid,sensor):
    data = load_by_sensor(trial_id, sid, sensor)
    dt = normalize_data(data[:,:3])
    
    X,y = [],[]
    for aid in range(5):
        _,idx = select_by_id(data, aid)
        if not idx[0].size > 0:
            continue
        d_piece = make_context_window(dt[idx], L, s)
        #discard last piece
        d_piece = d_piece[:-1]
        X.append(d_piece)
        label = np.ones((d_piece.shape[0],)) * aid
        y.append(label)
    
    x_ = convert_list(X)
    y_ = convert_y(y)
    X_train,X_test,y_train,y_test = train_test_split(x_,y_,test_size=0.2,stratify=y_,
                                                           random_state=42)
    
    path = "../segmented/trial1/Sub"+str(sid)+"_"+sensor+"_"
    save_tensor(X_train, sid, sensor, 'train')
    np.savetxt(path+"train_label.txt",y_train)
    save_tensor(X_test, sid, sensor, 'test')
    np.savetxt(path+"test_label.txt",y_test)

#process_data(23, 'ankle')    
sensors = ['thigh','wrist','ankle']
for sid in range(23,24):
    for sensor in sensors:
        process_data(sid, sensor)