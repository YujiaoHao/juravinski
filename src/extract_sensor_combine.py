# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:57:31 2021
extract and save train_test data as combination of sensors
in the order of ['thigh','wrist','ankle']
sub23: not all 3 sensor are available, discard
@author: hao_y
"""

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
from tools import normalize_data,plot_acti,make_context_window,select_by_id, convert_list, convert_y,plot_raw
from sklearn.model_selection import train_test_split

L = 60
s = 48


def load_by_sensor(trial_id,sid,sensor):
    path = '../processed/trial'+str(trial_id)+'/sub'+str(sid)+'_'+sensor+'.txt'
    return np.loadtxt(path)

# Save processed windowed data
def save_tensor(data,sid,name):
    # Write the array to disk
    with open("../data/trial2/Sub"+str(sid)+"_"+name+"_data.txt", 'w') as outfile:
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

trial_id = 2
# sid = 0
# sensor = 'ankle'  
          
def process_data(sid):
    sensors = ['thigh','wrist','ankle']
    data1 = load_by_sensor(trial_id, sid, sensors[0])
    data2 = load_by_sensor(trial_id, sid, sensors[1])
    data3 = load_by_sensor(trial_id, sid, sensors[2])
    dt1 = normalize_data(data1[:,:3])
   
    dt2 = normalize_data(data2[:,:3])
    
    dt3 = normalize_data(data3[:,:3])
    plot_raw(dt3)
    
    X,y = [],[]
    for aid in range(5):
        # if aid==2 or aid==3:
        #     continue
        _,idx1 = select_by_id(data1, aid)
        # _,idx2 = select_by_id(data2, aid)
        # _,idx3 = select_by_id(data3, aid)
        if not idx1[0].size > 0:
            continue
        #print(aid)
        dt = np.hstack((dt1[idx1],dt2[idx1],dt3[idx1]))
        d_piece = make_context_window(dt, L, s)
       
        #discard last piece
        d_piece = d_piece[:-1]
        X.append(d_piece)
        label = np.ones((d_piece.shape[0],)) * aid
        y.append(label)
        #print(aid)
    
    x_ = convert_list(X)
    y_ = convert_y(y)
    X_train,X_test,y_train,y_test = train_test_split(x_,y_,test_size=0.2,stratify=y_,
                                                           random_state=42)
    
    # path = "../data/trial2/Sub"+str(sid)+"_"
    # save_tensor(X_train, sid, 'train')
    # np.savetxt(path+"train_label.txt",y_train)
    # save_tensor(X_test, sid, 'test')
    # np.savetxt(path+"test_label.txt",y_test)
    print(X_train.shape, y_train.shape)

#process_data(23, 'ankle')    

for sid in range(2,3):
    process_data(sid)