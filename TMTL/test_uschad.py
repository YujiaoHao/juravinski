# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:47:07 2020
acc1: fast adaptation by bmtl
acc2: fast adaptation by tmtl
acc3: STL

@author: hao_y
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:54:51 2019
Fine=tune based fast adaptation
1.Use sub 106 data fine-tune each branch of leave-one subject out model
2.See the subject similarity with subject classifier
----the subject similarity has nothing to do with the fine-tune result
@author: yujiaohao
"""



import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import normalize
from keras.layers import Input, Conv2D,  Dense, Flatten,MaxPooling2D, concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from itertools import permutations
from keras.layers import LSTM, Permute,Reshape
import keras
import tensorflow as tf
from keras.models import model_from_json
from keras.constraints import max_norm
import collections
from sklearn.model_selection import train_test_split
import os
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import seaborn as sn
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE
import matplotlib.cm as cm

os.listdir('.')

NUM_CLASSES = 10

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 200

# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = 160

NUM_CHANNEL = 6

# =============================================================================
# Load and preprocess training and testing data
# =============================================================================

# to load the windowed data
def load_tensor(name):
    # Read the array from disk
    #new_data = np.loadtxt('./ActiveData/Sub'+str(name)+'_data.txt')
    new_data = np.loadtxt('./processed_uschad/Sub'+str(name)+'_data.txt')
    
    # Note that this returned a 2D array!
    print (new_data.shape)
    
    # However, going back to 3D is easy if we know the 
    # original shape of the array
    new_data = new_data.reshape((-1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    return new_data

def load_data_by_id(subid):
    X = load_tensor(subid)
    #y = np.loadtxt("./ActiveData/Sub"+str(subid)+"_label.txt")
    y = np.loadtxt("./processed_uschad/Sub"+str(subid)+"_label.txt")
    return X,y

labels = np.array([1,2,3,4,5,6,7,8,9,10])
lb = preprocessing.LabelBinarizer()
lb.fit(y=labels)

#evaluate and print predict result
def eval_perf(ground_truth, predicted_event):
    print('Accuracy score is: ')
    acc = accuracy_score(ground_truth, predicted_event)
    print(acc)
    print('Confusion Matrix is:')
    my_matrix = confusion_matrix(ground_truth, predicted_event)
    my_matrix_n = normalize(my_matrix, axis=1,norm = 'l1')
    print(pd.DataFrame(my_matrix_n).applymap(lambda x: '{:.2%}'.format(x)).values)

    #target_names = ['Null','WalkBK', 'Walk', 'Sit','Lie']
    target_names = ['walkF','walkL','walkR','upstairs','downstairs','run',
                    'jump','sit','stand','lying']
    df_cm = pd.DataFrame(my_matrix_n, index = [i for i in target_names],
                  columns = [i for i in target_names])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    print(classification_report(ground_truth, predicted_event, target_names=target_names))  
    return acc


# =============================================================================
# Test classification result 
# get intermediate output from L, and retrain S with 1% data from sub106
# =============================================================================
def convert_list(my_list):
    length = len(my_list)
    if length==0:
        return
    res = my_list[0]
    for i in range(1,length):
        res = np.vstack((res,my_list[i]))
    return res

def convert_y(my_list):
    length = len(my_list)
    if length==0:
        return
    res = my_list[0]
    for i in range(1,length):
        res = np.concatenate((res,my_list[i]))
    return res

def randomly_sample(X,y,num):
    resx, resy = [],[]
    for i in range(len(labels)):
        list0 = np.where(y==labels[i])
        ind0 = random.choice (list0)[:num]
        resy.append(y[ind0].reshape(-1,1))
        resx.append(X[ind0])
        X = np.delete(X,ind0,0)
        y = np.delete(y,ind0,0)
    X_train = convert_list(resx)
    y_train = convert_list(resy)
    return X_train,y_train,X,y

#selectively replace certain activity from training samples, eg: jump,run，lying=6,7,10
def reduce_data(x,y,aid):
    ind = np.where(y==aid)
    y = np.delete(y,ind)
    x = np.delete(x,ind,axis=0)
    return x,y

def select_act(x,y,aid):
    ind = np.where(y==aid)
    res = x[ind,:,:]
    return res.reshape(-1,200,6),y[ind]

#X2,y2 = load_data_by_id(2)
#d1,dy1 = select_act(X2,y2,6)
#d2,dy2 = select_act(X2,y2,7)
#X = np.vstack((d1,d2))
#y = np.concatenate((dy1,dy2))
#selectively remove certain activity from training samples, eg: jog,outbed,squat=5,6,7
def partial_reduce_data(x,y,aid,num):
    ind = np.where(y==aid)
    #t = ind.shape[0]
    y = np.delete(y,ind[0][:int(10-num)])
    x = np.delete(x,ind[0][:int(10-num)],axis=0)
    return x,y

def load_data_num(n):
    #1649 is good
    X,y = load_data_by_id(1)
    #fix 100 per class as test
    X,Xtest,y,ytest = train_test_split(X,y,test_size=1100,stratify=y,
                                                       random_state=42)
    #remove jump and run from sub1
    X,y = partial_reduce_data(X,y,10,n)
    X,y = partial_reduce_data(X,y,6,n)
    #replace jump and run with sub2
    X2,y2 = load_data_by_id(5)
    d1,dy1 = select_act(X2,y2,10)
    d2,dy2 = select_act(X2,y2,6)
    X = np.vstack((X,d1[:n,:,:]))
    X = np.vstack((X,d2[:n,:,:]))
    y = np.concatenate((y,dy1[:n]))
    y = np.concatenate((y,dy2[:n]))
    
    X_train,y_train,Xval,yval = randomly_sample(X,y,10)
    ytrain = y_train.reshape(-1,)
    print(collections.Counter(ytrain))
    return X_train,y_train,Xval,Xtest,yval,ytest


def load_tmtl():
    json_file = open('./models/uschad/TMTL/leave sub1/trimtl_model_us.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('./models/uschad/TMTL/leave sub1/trimtl_weights_us.h5')
    print("Loaded model from disk")
    print(model.summary())
    return model

def load_bmtl():
    json_file = open('./models/uschad/BMTL/leave sub1/bmtl_model_us1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('./models/uschad/BMTL/leave sub1/bmtl_weights_us1.h5')
    print("Loaded model from disk")
    print(model.summary())
    return model

# =============================================================================
# direct predict and check，
# then fast adaptation with bmtl
# =============================================================================
#direct predict with tmtl model
#model = load_tmtl()
def duplicate(data):
    res=[]
    res.append(data.reshape(-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    res.append(data.reshape(-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    res.append(data.reshape(-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    return res
    
#yp = model.predict(duplicate(Xtest))
#ypp = lb.inverse_transform(yp[2])
#acc1 = eval_perf(ytest,ypp)
def evaluate_bmtl(X_train,y_train,Xval,Xtest,yval,ytest):
    model = load_bmtl()
    #yp = model.predict(Xtest.reshape(-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    #ypp = lb.inverse_transform(yp[0])
    #acc1 = eval_perf(ytest,ypp)
    ytrain = y_train.reshape(-1,)
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('lstm_2').output)
    
    xtest_ = Xtest.reshape(-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL)
    test = intermediate_layer_model.predict(xtest_)
    
    
    intermediate_output_ = intermediate_layer_model.predict(X_train.reshape(-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    xval_ = intermediate_layer_model.predict(Xval.reshape(-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    y_ = lb.transform(ytrain)
    yval_ = lb.transform(yval)
    
    es = EarlyStopping(monitor='val_acc', mode='auto',patience=200,verbose=2)
    mc = ModelCheckpoint('best_classifier.h5',monitor='val_acc',mode='max',save_best_only=True)
    acc1 = []
    #7 seen subjects in bmtl(9,22)
    for i in range(9,11):
        W1 = model.layers[i].get_weights()
        #to define initial weights, have to pass a function
        def init_S_(shape, dtype=None):
            ker = np.zeros(shape, dtype=dtype)
            ker = W1[0]
            return ker
        
        Classifier_input = Input((128,))
        Classifier_output = Dense(NUM_CLASSES, 
                                  kernel_initializer=init_S_,
                                  activation='softmax')(Classifier_input)
        Classifier_model = Model(Classifier_input, Classifier_output)
        Classifier_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        #with tf.device('/gpu:0'): 
        Classifier_model.fit(intermediate_output_,y_,callbacks=[es,mc],verbose=0,
                                 validation_data=(xval_,yval_),shuffle=True,epochs=3000)
        
        
        myy = Classifier_model.predict(test)
        predict_class = lb.inverse_transform(myy)
        tmp = eval_perf(ytest,predict_class)
        acc1.append(tmp)
    return acc1,np.asarray(acc1).max()


# =============================================================================
# test tripletNN+GOMTL with given split
# =============================================================================
#load model
def evaluate_tmtl(X_train,y_train,Xval,Xtest,yval,ytest):
    model = load_tmtl()
    
    #filter out data that no need for fast adaptation
    #Xtrain,ytrain = reduce_data(Xtrain,ytrain,6)
    #Xtrain,ytrain = partial_reduce_data(Xtrain,ytrain,5, 0.8)
    #Xtrain,ytrain = reduce_data(Xtrain,ytrain,8)
    #y_ = lb.transform(ytrain)
    
    xtrain_ = duplicate(X_train)
    val_ = duplicate(Xval)
    ytrain = y_train.reshape(-1,)
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('merged_layer').output)
    intermediate_output_ = intermediate_layer_model.predict(xtrain_)
    xval_ = intermediate_layer_model.predict(val_)
    y_ = lb.transform(ytrain)
    yval_ = lb.transform(yval)
    xtest_ = duplicate(Xtest.reshape(-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    test = intermediate_layer_model.predict(xtest_)
    
    es = EarlyStopping(monitor='val_acc', mode='auto',patience=200,verbose=2)
    mc = ModelCheckpoint('best_classifier.h5',monitor='val_acc',mode='max',save_best_only=True)
    
    acc2 = []
    #7 seen subjects in tmtl(5,18)
    for i in range(5,7):
        W1 = model.layers[i].get_weights()
        #to define initial weights, have to pass a function
        def init_S_(shape, dtype=None):
            ker = np.zeros(shape, dtype=dtype)
            ker = W1[0]
            return ker
        
        Classifier_input = Input((384,))
        Classifier_output = Dense(NUM_CLASSES, 
                                  kernel_initializer=init_S_,
                                  activation='softmax')(Classifier_input)
        Classifier_model = Model(Classifier_input, Classifier_output)
        Classifier_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        #with tf.device('/gpu:0'): 
        Classifier_model.fit(intermediate_output_,y_,callbacks=[es,mc],verbose=0,
                                 validation_data=(xval_,yval_),shuffle=True,epochs=3000)
        
        
        myy = Classifier_model.predict(test)
        predict_class = lb.inverse_transform(myy)
        tmp = eval_perf(ytest,predict_class)
        acc2.append(tmp)
    return acc2,np.asarray(acc2).max()


    
# =============================================================================
# do t-sne of test data
# =============================================================================
def tsne(test,ytest):
    data1 = np.hstack((test,ytest.reshape(-1,1)))
    Y = TSNE(n_components=2).fit_transform(data1)
    
    fig, ax = plt.subplots()
    groups = pd.DataFrame(Y, columns=['x', 'y']).assign(category=ytest).groupby('category')
    listact = ['walkF','walkL','walkR','upstairs','downstairs','run',
               'jump','sit','stand','lying']
    #colors = cm.rainbow(np.linspace(0, 1, len(listact)))
    ind=0
    for name, points in groups:
        f = listact[int(name-1)]
        print(f)
        #ax.scatter(points.x, points.y, label=f, color=colors[ind])
        ax.scatter(points.x, points.y, label=f)
        ind+=1
    ax.legend()
# =============================================================================
# do fast adaptation
# ====================================================================

#
#Classifier_input = Input((384,))
#Classifier_output = Dense(NUM_CLASSES, activation='softmax')(Classifier_input)
#Classifier_model = Model(Classifier_input, Classifier_output)
#Classifier_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#with tf.device('/gpu:0'): 
#    Classifier_model.fit(intermediate_output_,y_, callbacks=[es,mc],verbose=0,
#                         validation_data=(xval_,yval_),epochs=3000)
#
#xtest_ = duplicate(Xtest.reshape(-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
#test = intermediate_layer_model.predict(xtest_)
#myy = Classifier_model.predict(test)
#predict_class = lb.inverse_transform(myy)
#acc1 = eval_perf(ytest,predict_class)
    
# =============================================================================
# train a baseline with same data
# =============================================================================
# Batch Size
BATCH_SIZE = 100

# Number filters convolutional layers
NUM_FILTERS = 64

# Size filters convolutional layers
FILTER_SIZE = 5

# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128

n_epoch = 100


def network_test():
    multi_input = Input(shape=(1, SLIDING_WINDOW_LENGTH, NUM_CHANNEL), name='multi_input')
    print(multi_input.shape)  # (?, 1, 24, 113)
    
    y = Conv2D(64, (5, 1), activation='relu', data_format='channels_first')(multi_input)
    print(y.shape)  # (?, 64, 20, 113)
    
    y = Conv2D(64, (5, 1), activation='relu', data_format='channels_first')(y)
    print(y.shape)
    y = Conv2D(64, (5, 1), activation='relu', data_format='channels_first')(y)
    print(y.shape)
    y = Conv2D(64, (5, 1), activation='relu', data_format='channels_first')(y)
    print(y.shape)
    
    y = Permute((2, 1, 3))(y)
    print(y.shape)  # (?, 20, 64, 113)
    
    # This line is what you missed
    # ==================================================================
    y = Reshape((int(y.shape[1]), int(y.shape[2]) * int(y.shape[3])))(y)

    # ==================================================================
    print(y.shape)  # (?, 20, 7232)
    
    y = LSTM(128,dropout=0.25,return_sequences=True)(y)
    y = LSTM(128)(y)
      # (?, 128)
    y = Dense(NUM_CLASSES,activation = 'softmax')(y)
    print(y.shape)
    return keras.Model(inputs=multi_input, outputs=y)  

def evaluate_stl(X_train,y_train,Xval,X_test,yval,y_test):
    X_train = X_train.reshape(-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL) 
    m = network_test()
    #m.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(m.summary())
    res = lb.transform(y_train)
    #train with gpu
    #with tf.device('/gpu:0'):
    train_history = m.fit(X_train,res,epochs=n_epoch,
                          batch_size=100,verbose=0,shuffle=True)
    
    import time
    #predict label
    X_test = X_test.reshape(-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL)
    #count the predict time
    start_time = time.time()
    predict_y = m.predict(X_test)
    print("Run in --- %s seconds ---" % (time.time() - start_time))
    predict_class = lb.inverse_transform(predict_y)
    
    #evaluate and print predict result
    acc3 = eval_perf(y_test,predict_class)
    return acc3

i = 5
acc1_,a1_,acc2_,a2_,acc3_=[],[],[],[],[]
while i<=10:
    X_train,y_train,Xval,Xtest,yval,ytest = load_data_num(i)
    acc1,a1 = evaluate_bmtl(X_train,y_train,Xval,Xtest,yval,ytest)
    acc2,a2 = evaluate_tmtl(X_train,y_train,Xval,Xtest,yval,ytest)
    acc3 = evaluate_stl(X_train,y_train,Xval,Xtest,yval,ytest)
    acc1_.append(acc1)
    a1_.append(a1)
    acc2_.append(acc2)
    a2_.append(a2)
    acc3_.append(acc3)
    i+=3
print(a1_)
print(a2_)
print(acc3_)