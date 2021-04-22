#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 21:48:02 2019

@author: yujiaohao
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVC
from sklearn import tree
from sklearn.metrics import confusion_matrix
from scipy import stats
from statsmodels import robust
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
# I. time-domain features
# 1.mean
# 2.standard deviation
# 3.relative standard deviation
# 4.mean absolute deviation
# 5.max
# 6.min
# 7.interquartile range
# 8.variation
#9.Zero crossing rate (ZC)
# =============================================================================
# 1.mean
def get_mean_feature(X_raw):
    m,n,k = X_raw.shape
    res = np.ones((m,k))
    for i in range(m):
        res[i,:] = X_raw[i,:,:].mean(axis=0)
    return res

# 2.standard deviation
def get_sd_feature(X_raw):
    m,n,k = X_raw.shape
    res = np.ones((m,k))
    for i in range(m):
        res[i,:] = X_raw[i,:,:].std(axis=0)
    return res

# 3.relative standard deviation
def get_cv_feature(X_raw):
    m,n,k = X_raw.shape
    res = np.ones((m,k))
    for i in range(m):
        res[i,:] = stats.variation(X_raw[i,:,:])
    return res    

# 4.mean absolute deviation
def get_mad_feature(X_raw):
    m,n,k = X_raw.shape
    res = np.ones((m,k))
    for i in range(m):
        res[i,:] = robust.mad(X_raw[i,:,:])
    return res    

# 5.max
def get_max_feature(X_raw):
    m,n,k = X_raw.shape
    res = np.ones((m,k))
    for i in range(m):
        res[i,:] = np.max(X_raw[i,:,:])
    return res 
   
# 6.min
def get_min_feature(X_raw):
    m,n,k = X_raw.shape
    res = np.ones((m,k))
    for i in range(m):
        res[i,:] = np.min(X_raw[i,:,:])
    return res 

# 7.interquartile range
def get_iqr_feature(X_raw):
    m,n,k = X_raw.shape
    res = np.ones((m,k))
    for i in range(m):
        res[i,:] = stats.iqr(X_raw[i,:,:])
    return res 

# 8.variation
def get_variation_feature(X_raw):
    m,n,k = X_raw.shape
    res = np.ones((m,k))
    for i in range(m):
        res[i,:] = X_raw[i,:,:].var(axis=0)
    return res

#  9.Zero crossing rate (ZC)
def feature_ZC(X,T):
    m,n,k = X.shape
    res = np.ones((m,k))
    for i in range(m):
        temp = 0
        for t in range(k):
            for j in range(n-1):   
                if abs(X[i,j+1,t]-X[i,j,t])>max(abs(X[i,j+1,t]+X[i,j,t]),T):
                    temp+=1
            res[i,t] = temp    
    return res

import scipy as sp
def get_energy(X):
    # Create input of real sine wave
    fs = 1.0
    fc = 0.25
    n = sp.arange(0, 300)
    x = sp.cos(2*sp.pi*n*fc/fs)
    
    # Rearrange x into 10 30 second windows
    x = sp.reshape(x, (-1, 30))
    
    # Calculate power over each window [J/s]
    p = sp.sum(x*x, 1)/x.size
    
    # Calculate energy [J = J/s * 30 second]
    e = p*x.size
    return e

#def get_sma(X):
#    m,n,o = X.shape
#    sum = 0
#    for i in range(m):
#        
#    public double sma(double[] x, double[] y, double[] z){
#        double sum = 0;
#        for(int i=0; i<x.length; i++)
#            sum += (Math.abs(x[i]) + Math.abs(y[i]) + Math.abs(z[i]));
#        return sum /x.length;
#    }

def get_meanfreq_feature(X):
    m,n,o = X.shape
    res = np.ones((m,o))
    for i in range(m):
        temp = X[i,:,:]
        frq = np.fft.fft2(temp)
        #use numpy.mean will lose the imaginary part
        #res[i,:] = frq.mean(axis=0)
        for k in range(o):
            avg = (np.real(frq[:,k])).sum() / n + 1j * (np.imag(frq[:,k])).sum() / n
            res[i,k] = avg
    return res

res = get_meanfreq_feature(label_window1[:,:,:12])
print(res)
# =============================================================================
# Calculate Pearson correlation coefficient as feature, replace nan with 0(no relationship )
# =============================================================================
from scipy.stats import pearsonr
def get_correlation_coefficient(X):
    m,n,o = X.shape
    res = np.ones((m,o))
    for i in range(m):
        temp = X[i,:,:]
        for j in range(o):
            ind1 = j
            ind2 = j+1
            if (ind1+1)%3==0:
                ind2 = j-2
            pe,p = pearsonr(temp[:,ind1],temp[:,ind2])
            if np.isnan(pe):
                pe = 0
            res[i,j] = pe
    return res

from scipy.special import entr
from scipy.stats import entropy
def get_entropy(X):
    m,n,o = X.shape
    res = np.ones((m,o))
    res2 = np.ones((m,o))
    for i in range(m):
        p = X[i,:,:]
        res[i,:] = (entr(p).sum(axis=0))/np.log(2)
        for j in range(o):
            res2[i,j] = entropy(p[:,j],)
    return res,res2
res,res2 = get_entropy(label_window1[:,:,:12])
print(res)
print(res2[0])


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

def feature_matrix(data):
    X1 = get_mean_feature(data)
    X2 = get_max_feature(data)
    X3 = get_min_feature(data)
    X4 = get_sd_feature(data)
    #X5 = get_mad_feature(data)
    X5 = get_meanfreq_feature(data)
    X6 = get_iqr_feature(data)
    #X7 = get_variation_feature(data)
    #X7 = get_entropy(data)
    X7 = get_correlation_coefficient(data)
    return np.hstack((X1,X2,X3,X4,X5,X6,X7))

feature1 = feature_matrix(label_window1[:,:,:12])
data1 = np.hstack((feature1,label_window1[:,0,12:13]))
feature2 = feature_matrix(label_window2[:,:,:12])
data2 = np.hstack((feature2,label_window2[:,0,12:13]))
feature3 = feature_matrix(label_window3[:,:,:12])
data3 = np.hstack((feature3,label_window3[:,0,12:13]))
feature4 = feature_matrix(label_window4[:,:,:12])
data4 = np.hstack((feature4,label_window4[:,0,12:13]))

#data1[:,84] = 1
#data2[:,84] = 2
#data1 = data1[np.random.randint(data1.shape[0], size=2000), :]
#data2 = data2[np.random.randint(data2.shape[0], size=2000), :]
#mydata = np.vstack((data1[:,:84],data2[:,:84]))
#np.savetxt('sub12.txt',mydata)
#mylabel = np.vstack((data1[:,84].reshape(-1,1),data2[:,84].reshape(-1,1)))
#np.savetxt('label12.txt',mylabel)

#np.savetxt('data4.txt',data4[:,:84])
#np.savetxt('label4.txt',data4[:,84])


# =============================================================================
# #2. Use PCA to get the key column
# =============================================================================
#from sklearn.decomposition import PCA
#
#def PCA_analysis(X):
#    my_model = PCA(n_components=3)
#    my_model.fit_transform(X)
#    res = my_model.transform(X)
#    return res
#
#def mle_PCA(X):
#    pca = PCA(n_components='mle', svd_solver='full')
#    #pca = decomposition.PCA(n_components=28)
#    pca.fit(X)
#    X = pca.transform(X)
#    return X
#
#
#def select_activity(X,action_id):
#    m,n = X.shape
#    res = []
#    for i in range(m):
#        if X[i,84]==action_id:
#            res.append(X[i,:84])
#    return np.asarray(res)
#
#
#null_data = select_activity(data1,0)
#stand_data = select_activity(data1,101)
#walk_data = select_activity(data1,102)
#sit_data = select_activity(data1,104)
#lie_data = select_activity(data1,105)
#
#A1 = PCA_analysis(null_data)
#A2 = PCA_analysis(stand_data)
#A3 = PCA_analysis(walk_data)
#A4 = PCA_analysis(sit_data)
#A5 = PCA_analysis(lie_data)
#
#from mpl_toolkits.mplot3d import Axes3D
#L = 500
#fig = plt.figure()
#ax = Axes3D(fig)
##ax.scatter(A1[:,0],A1[:,1],A1[:,2],label='null')
#ax.scatter(A2[:L,0],A2[:L,1],A2[:L,2],label='Stand')
#ax.scatter(A3[:L,0],A3[:L,1],A3[:L,2],label='Walk')
#ax.scatter(A4[:L,0],A4[:L,1],A4[:L,2],label='Sit')
#ax.scatter(A5[:L,0],A5[:L,1],A5[:L,2],label='Lie')
#ax.legend()
#plt.show()
#
#d2 = PCA_analysis(data2[:,:84])
#d3 = PCA_analysis(data3[:,:84])
#fig = plt.figure()
#ax = Axes3D(fig)
##ax.scatter(X1[:,0],X1[:,1],X1[:,2],color='y')
#ax.scatter(d2[:2000,0],d2[:2000,1],d2[:2000,2],label='Sub2')
#ax.scatter(d3[:2000,0],d3[:2000,1],d3[:2000,2],label='Sub3')
##ax.scatter(X4[:,0],X4[:,1],X4[:,2],color='b')
#ax.legend()
#plt.show()

# =============================================================================
# end of PCA
# =============================================================================
dataset = np.vstack((data4,data1,data3,data2))
#dataset = np.vstack((data1[:,:84],data2[:,:84]))

X = dataset[:,:84]
y = dataset[:,84].reshape(-1,1)
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

#4-2. k-fold cross validation(k=10)
print('--------Start the K fold evaluation----')
k=10
kf = KFold(n_splits=k, random_state=True, shuffle=True)
scores=np.zeros(k)
i=0
for train_index, test_index in kf.split(X,y):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf_DT.fit(X_train, y_train)
    predicted_Y = clf_DT.predict(X_test)
    eval_perf(y_test,predicted_Y)
    scores[i] = clf_DT.score(X_test, y_test)
    i+=1
DT_avg_score = scores.mean()
print('Average score is: %6.8f\n' % (DT_avg_score))

#4-2.train and test SVM with train_X,test_X
clf_SVM = SVC(C=1000,gamma=0.01,kernel='rbf')
clf_SVM.fit(X_train, y_train.ravel())

start_time = time.time()
predicted_Y = clf_SVM.predict(X_test)
eval_perf(y_test,predicted_Y.reshape(-1,1))
print("Run in --- %s seconds ---" % (time.time() - start_time))

#from sklearn.model_selection import GridSearchCV
#
#def svc_param_selection(X, y, nfolds):
#    Cs = [0.001, 0.01, 0.1, 1, 10]
#    gammas = [0.001, 0.01, 0.1, 1]
#    param_grid = {'C': Cs, 'gamma' : gammas}
#    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
#    grid_search.fit(X, y)
#    grid_search.best_params_
#    return grid_search.best_params_
#
#para = svc_param_selection(X_train,y_train,10)

##3-3. hyper parameter tuning
#from sklearn.model_selection import GridSearchCV
## Set the parameters by cross-validation
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
#                   ]
#
#scores = ['precision', 'recall']
#
#for score in scores:
#    print("# Tuning hyper-parameters for %s" % score)
#    print()
#
#    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
#                       scoring='%s_macro' % score)
#    clf.fit(X_train, y_train)
#    
#    print("Best parameters set found on development set:")
#    print()
#    print(clf.best_params_)
#    print()
#    print("Grid scores on development set:")
#    print()
#    means = clf.cv_results_['mean_test_score']
#    stds = clf.cv_results_['std_test_score']
#    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean, std * 2, params))
#    print()

# =============================================================================
# Test with unseen data
    #Train with sub1+sub2
    #Test with sub3
# =============================================================================

X_train = dataset[:,:84]
y_train = dataset[:,84].reshape(-1,1)
X_test = data2[:,:84]
y_test = data2[:,84].reshape(-1,1)

#1.train and test SVM with train_X,test_X
clf_SVM = SVC(C=1000,gamma=0.01,kernel='rbf')
clf_SVM.fit(X_train, y_train)

start_time = time.time()
predicted_Y = clf_SVM.predict(X_test)
eval_perf(y_test,predicted_Y.reshape(-1,1))
print("Run in --- %s seconds ---" % (time.time() - start_time))

#2. train and test with decision tree
clf_DT = tree.DecisionTreeClassifier()
clf_DT.fit(X_train, y_train)

#get inference time
start_time = time.time()
predicted_Y = clf_DT.predict(X_test)
p_y = predicted_Y.reshape(-1,1)
eval_perf(y_test,p_y)
print("Run in --- %s seconds ---" % (time.time() - start_time))