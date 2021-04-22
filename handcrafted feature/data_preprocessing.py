# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 18:01:40 2019

@author: hao_y
"""
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import normalize


def select_col(X):
    t1 = X[82:88]
    t2 = X[98:104]
    t3 = X[114]
    return np.hstack((t1,t2,t3))

def preprocess_data(sub_id,mo_id):
    for i in range(1,6):
        path = '../Opportunity/S'+str(sub_id)+'-ADL'+str(i)+'.dat'
        data = np.loadtxt(path)
        res = []
        n = data.shape[0]
        for j in range(n):
            #if data[j][mo_id]!=0:
                t = select_col(data[j,:])
                res.append(t)
        path1 = '../Opportunity/S'+str(sub_id)+'-ADL'+str(i)+'_'+str(mo_id)+'.dat'
        res = np.asarray(res)
        np.savetxt(path1,res)
        
def load_sub_data(sub_id):
    res = []
    for i in range(1,6):
        path = '../Opportunity/S'+str(sub_id)+'-ADL'+str(i)+'_'+str(114)+'.dat'
        data = np.loadtxt(path)
        res.append(data)
    return res

data = load_sub_data(1)

def convert_list(my_list):
    length = len(my_list)
    if length==0:
        return
    res = my_list[0]
    for i in range(1,length):
        res = np.vstack((res,my_list[i]))
    return res


def normalize_data(sub_id):
    res = load_sub_data(sub_id)
    res = convert_list(res)
    #check if there are nans
    check_nan = np.argwhere(np.isnan(res))
    if check_nan.size!=0:
        n = check_nan.shape[0]
        for i in range(n):
            ind1 = check_nan[i][0]
            ind2 = check_nan[i][1]
            res[ind1,ind2] = 0
    temp = normalize(res[:,:12], axis=0, norm='max')
    m = temp.shape[0]
    subid = np.ones((m,1))*sub_id
    #res = np.hstack((temp,res[:,12:13],subid))
    res = np.hstack((temp,res[:,12:13]))
    return res

def make_context_window(X_raw,L,s):
    m,n = X_raw.shape
    res = []
    i = 0
    while i<=m:
        ind1 = i*(L-s)
        ind2 = ind1+L
        if ind2>m and ind1<m:
            mzeros = np.zeros((ind2-m,n))
            temp = np.vstack((X_raw[ind1:m,:],mzeros))
            res.append(temp)
            break
        res.append(X_raw[ind1:ind2, :])
        i = i+1
    res = np.stack(res, axis=0) 
    return res

def label_per_window(X):
    m,n,o = X.shape
    mones = np.ones((n,1))
    for i in range(m):
        temp = X[i,:,12:13]
        unique, counts = np.unique(temp, return_counts=True)
        label = unique[counts.argmax(axis=0)]*mones
        X[i,:,:] = np.hstack((X[i,:,:12],label))
    return X

SLIDING_WINDOW_LENGTH = 60
def load_tensor(name):
    # Read the array from disk
    new_data = np.loadtxt('./SplittedData/bysubject/'+name+'.txt')
    
    # Note that this returned a 2D array!
    print (new_data.shape)
    
    # However, going back to 3D is easy if we know the 
    # original shape of the array
    new_data = new_data.reshape((-1,SLIDING_WINDOW_LENGTH,12))
    return new_data



def eval_perf(ground_truth, predicted_event):
    print('Accuracy score is: ')
    acc = accuracy_score(ground_truth, predicted_event)
    print(acc)
    print('Confusion Matrix is:')
    my_matrix = confusion_matrix(ground_truth, predicted_event)
    my_matrix_n = normalize(my_matrix, axis=1,norm = 'l1')
    print(pd.DataFrame(my_matrix_n).applymap(lambda x: '{:.2%}'.format(x)).values)

    target_names = ['null','Stand', 'Walk', 'Sit','Lie']
    print(classification_report(ground_truth, predicted_event, target_names=target_names))  
    return acc
    


    