# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 21:07:43 2020
test ECDF on opportunity
@author: hao_y
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVC
from sklearn import tree
from sklearn.metrics import confusion_matrix
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.interpolate import interp1d 
import data_preprocessing

# =============================================================================
# filter selected type of motions from the dataset
    #col 115:locomotion
    #col 116:gesture
# =============================================================================
def select_col(X):
    t1 = X[82:88]
    t2 = X[98:104]
    t3 = X[114]
    return np.hstack((t1,t2,t3))

def preprocess_data(sub_id,mo_id):
    for i in range(1,6):
        path = './Opportunity/S'+str(sub_id)+'-ADL'+str(i)+'.dat'
        data = np.loadtxt(path)
        res = []
        n = data.shape[0]
        for j in range(n):
            #if data[j][mo_id]!=0:
                t = select_col(data[j,:])
                res.append(t)
        path1 = './Opportunity/S'+str(sub_id)+'-ADL'+str(i)+'_'+str(mo_id)+'.dat'
        res = np.asarray(res)
        np.savetxt(path1,res)
                
        
#preprocess_data(4,114)

def load_sub_data(sub_id):
    res = []
    for i in range(1,6):
        path = './Opportunity/S'+str(sub_id)+'-ADL'+str(i)+'_'+str(114)+'.dat'
        data = np.loadtxt(path)
        res.append(data)
    return res

def convert_list(my_list):
    length = len(my_list)
    if length==0:
        return
    res = my_list[0]
    for i in range(1,length):
        res = np.vstack((res,my_list[i]))
    return res

from sklearn.preprocessing import normalize
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
    res = np.hstack((temp,res[:,12:13]))
    return res

sub1_data = data_preprocessing.normalize_data(1)
sub2_data = data_preprocessing.normalize_data(2)
sub3_data = data_preprocessing.normalize_data(3)
sub4_data = data_preprocessing.normalize_data(4)



# =============================================================================
# make window: size 120, overlap 96
# =============================================================================
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

window1 = make_context_window(sub1_data,60,48)
window2 = make_context_window(sub2_data,60,48)
window3 = make_context_window(sub3_data,60,48)
window4 = make_context_window(sub4_data,60,48)

# =============================================================================
# Motion label
#101   -   Locomotion   -   Stand
#102   -   Locomotion   -   Walk
#104   -   Locomotion   -   Sit
#105   -   Locomotion   -   Lie
# =============================================================================
def label_per_window(X):
    m,n,o = X.shape
    mones = np.ones((n,1))
    for i in range(m):
        temp = X[i,:,12:13]
        unique, counts = np.unique(temp, return_counts=True)
        label = unique[counts.argmax(axis=0)]*mones
        X[i,:,:] = np.hstack((X[i,:,:12],label))
    return X

label_window1 = label_per_window(window1)
label_window2 = label_per_window(window2)
label_window3 = label_per_window(window3)
label_window4 = label_per_window(window4)

# =============================================================================
# Feature extraction
# calculate ECDF feature, use d=30
# =============================================================================
# 1.calculate ecdf
def ecdf(sample):
    '''
    quantiles : x in the paper
    cumprob : f in the paper

    '''

    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

def get_ecdf_feature(X_raw,num = 30):
    m,n,k = X_raw.shape
    final = []
    for i in range(m):
        #mean feature per channel
        temp1 = X_raw[i,:,:].mean(axis=0)
        X = []
        for j in range(k):
            d = X_raw[i,:,j]
            y = d + np.random.rand(d.shape[0]) * 0.01
            x,f = ecdf(y)
            
            #interpolate
            interp =interp1d(f,x,kind='cubic',fill_value="extrapolate")
            ll = interp(np.linspace(0,1,num))
            X.append(ll)
        #ecdf feature per channel    
        temp2 = np.array(X).reshape(-1,num)
        #put together feature per channel
        temp3 = np.vstack((temp1,temp2.T))
        final.append(temp3)
    
    return final

# =============================================================================
# Machine learning part
# =============================================================================

#1. To help you evaluate, we provide the following function to evaluate the performance
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd

def eval_perf(ground_truth, predicted_event):
    print('Accuracy score is: ')
    print(accuracy_score(ground_truth, predicted_event))
    print('Confusion Matrix is:')
    my_matrix = confusion_matrix(ground_truth, predicted_event)
    my_matrix_n = normalize(my_matrix, axis=1,norm = 'l1')
    print(pd.DataFrame(my_matrix_n).applymap(lambda x: '{:.2%}'.format(x)).values)

    target_names = ['null','Stand', 'Walk', 'Sit','Lie']
    print(classification_report(ground_truth, predicted_event, target_names=target_names))      


#3.get train set and test set
def prepare_data(X,y, num):
    feature_ = get_ecdf_feature(X,num)
    feature1 = np.array(feature_).reshape(-1,(num+1)*12)
    data1 = np.hstack((feature1,y.reshape(-1,1)))
    return data1

data1 = prepare_data(label_window1[:,:,:12],label_window1[:,0,-1],num=30)
data2 = prepare_data(label_window2[:,:,:12],label_window2[:,0,-1],num=30)
data3 = prepare_data(label_window3[:,:,:12],label_window3[:,0,-1],num=30)
data4 = prepare_data(label_window4[:,:,:12],label_window4[:,0,-1],num=30)

dataset = np.vstack((data1,data2,data3,data4))

X = dataset[:,:-1]
y = dataset[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42)

# =============================================================================
# 4. Try to use decision tree to solve this problem
# 4-1. Train and test with divided dataset
# 4-2. k-fold cross validation
# =============================================================================
#4-1.train and test DT with train_X,test_X
clf_DT = tree.DecisionTreeClassifier()
clf_DT.fit(X_train, y_train)

#get inference time
start_time = time.time()
predicted_Y = clf_DT.predict(X_test)
p_y = predicted_Y.reshape(-1,1)
eval_perf(y_test,p_y)
print("Run in --- %s seconds ---" % (time.time() - start_time))


#4-2.train and test SVM with train_X,test_X
clf_SVM = SVC(C=1000,gamma=0.01,kernel='rbf')
clf_SVM.fit(X_train, y_train.ravel())

start_time = time.time()
predicted_Y = clf_SVM.predict(X_test)
eval_perf(y_test,predicted_Y.reshape(-1,1))
print("Run in --- %s seconds ---" % (time.time() - start_time))