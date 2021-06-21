# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:53:46 2020
uschad TMTL for all subjects
@author: hao_y
"""

# -*- coding: utf-8 -*-

"""
Created on Sun Feb 16 21:09:50 2020
leave one out T-MTL on USCHAD dataset
@author: hao_y
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
from sklearn.model_selection import train_test_split
import collections
import random
import tensorflow as tf
import gc

ALPHA = 2
#number of activities
NUM_CLASSES = 10

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 200

# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = 160

# Batch Size
BATCH_SIZE = 100

# Number filters convolutional layers
NUM_FILTERS = 64

# Size filters convolutional layers
FILTER_SIZE = 5

# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128

NUM_CHANNEL = 6

#flatten data into matrix(28x28-->784),for sensor data no need to do this 
# =============================================================================
# Load and preprocess training and testing data
# =============================================================================

# to load the windowed data
def load_tensor(sid,name=''):
    # Read the array from disk
    #new_data = np.loadtxt('./ActiveData/Sub'+str(name)+'_data.txt')
    new_data = np.loadtxt('./split_uschad/Sub'+str(sid)+name+'_data.txt')
    
    # Note that this returned a 2D array!
    print (new_data.shape)
    
    # However, going back to 3D is easy if we know the 
    # original shape of the array
    new_data = new_data.reshape((-1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    return new_data

def load_train_data(sid):
    X_train = load_tensor(sid,'train')
    X_val = load_tensor(sid,'val')
    #y = np.loadtxt("./ActiveData/Sub"+str(subid)+"_label.txt")
    y_train = np.loadtxt("./split_uschad/Sub"+str(sid)+"train_label.txt")
    y_val = np.loadtxt("./split_uschad/Sub"+str(sid)+"val_label.txt")
    return X_train,X_val,y_train,y_val

#remove 11 and 12
#labels = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
labels = np.array([1,2,3,4,5,6,7,8,9,10])
lb = preprocessing.LabelBinarizer()
lb.fit(y=labels)

#1.load and divide data for each subject in the 13 subjects
X_train,X_val,y_train,y_val = [],[],[],[]
for i in range(1,15):
    xtrain,xval,ytrain,yval = load_train_data(i)
    X_train.append(xtrain)
    y_train.append(ytrain)
    X_val.append(xval)
    y_val.append(yval)


print(len(X_train))

#3. flatten the 60X12 part
x_train_flat = []
x_val_flat = []

for i in range(14):
    x_train_flat.append(X_train[i].reshape(-1,1200))
    x_val_flat.append(X_val[i].reshape(-1,1200))
    #the fewest label is 16 and 161 respectively, so pick m=16 n=160
    print(collections.Counter(y_train[i].ravel()))
    print(collections.Counter(y_val[i].ravel()))
    

# =============================================================================
# Replicate triplet nn method
# 1. preprocess data into triplets
#   This data generator has to be updated as the memory can't bear all these samples
# 2. construct triplet nn
# =============================================================================
def generate_triplet(x,y,ap_pairs,an_pairs):
    """
    To generate triplet from original dataset
    Arguments:
    ap_pairs -- how many random anchor-positive samples generate for a given class
    an_pairs -- how many random anchor-negative samples generate for a given class
                Does not required to be the same as ap_pairs, but for banlancing, should set the same?
    
    Returns:
    triplet_pairs -- 3d tensor; its shape[0]=NUM_CLASSES*ap_pairs*an_pairs
    """
    data_xy = tuple([x,y])

    triplet_pairs = []
    labels = []
   
    for data_class in sorted(set(data_xy[1])):
        same_class_idx = np.where((data_xy[1] == data_class))[0]
        diff_class_idx = np.where(data_xy[1] != data_class)[0]
        
        if same_class_idx.shape[0] == 1:
               same_ = [(same_class_idx[0],same_class_idx[0])]
        else:
            same_ = list(permutations(same_class_idx,2))
        diff_ = list(diff_class_idx)
        if len(same_) >= ap_pairs:
            A_P_pairs = random.sample(same_,k=ap_pairs) #Generating Anchor-Positive pairs
        else:
            A_P_pairs = same_
        if len(diff_) >= an_pairs:
            Neg_idx = random.sample(diff_,k=an_pairs)
        else:
            Neg_idx = diff_
        
        #Put data into triplets according to the indices in (A_P_pairs and Neg_idx)
        A_P_len = len(A_P_pairs)
        #print(A_P_len)
        for ap in A_P_pairs[:int(A_P_len)]:
            Anchor = data_xy[0][ap[0]]
            Positive = data_xy[0][ap[1]]
            for n in Neg_idx:
                Negative = data_xy[0][n]
                triplet_pairs.append([Anchor,Positive,Negative])
                labels.append(data_class)
                
    return np.array(triplet_pairs),np.array(labels)
# =============================================================================
# Loss function
# =============================================================================
def triplet_loss(model, alpha = 0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    myy_pred = model.get_layer("merged_layer").output
    #print('triplet NN.shape = ',myy_pred)
    
    total_lenght = myy_pred.shape.as_list()[-1]
#     print('total_lenght=',  total_lenght)
#     total_lenght =12
    
    anchor = myy_pred[:,0:int(total_lenght*1/3)]
    positive = myy_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
    negative = myy_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)

    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
 
    return loss


def multi_task_loss(model, alpha = 1):
    def _loss(y_true, y_pred):
        myloss = K.mean(K.binary_crossentropy(y_true, y_pred))
        miu = 0.8
        S0 = model.layers[5].get_weights()[0]
        S1 = model.layers[6].get_weights()[0]
        S2 = model.layers[7].get_weights()[0]
        S3 = model.layers[8].get_weights()[0]
        
        S4 = model.layers[9].get_weights()[0]
        S5 = model.layers[10].get_weights()[0]
        S6 = model.layers[11].get_weights()[0]
        S7 = model.layers[12].get_weights()[0]
        
        S8 = model.layers[13].get_weights()[0]
        S9 = model.layers[14].get_weights()[0]
        S10 = model.layers[15].get_weights()[0]
        S11 = model.layers[16].get_weights()[0]
        S12 = model.layers[17].get_weights()[0]
        S13 = model.layers[18].get_weights()[0]
        
        S = np.hstack((S0,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13))
#        myloss+= miu*np.linalg.norm(S[0].reshape(-1,5),ord=1)
        myloss+= miu*np.linalg.norm(S,ord=1)
       
        return myloss*alpha + triplet_loss(model)
    return _loss

# =============================================================================
# Construct the neural network
# =============================================================================
#The 'Net' part, will be replaced by 4CNN+2LSTM
def create_base_network(in_dims):
    """
    Base network to be shared.
    """
    multi_input = Input(shape=(1, SLIDING_WINDOW_LENGTH,NUM_CHANNEL), name='multi_input')
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
    
    return keras.Model(inputs=multi_input, outputs=y)  

adam_optim = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
#define 3 input .reshape(-1,1,SLIDING_WINDOW_LENGTH,12)
anchor_input = Input((1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL, ), name='anchor_input')
positive_input = Input((1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL, ), name='positive_input')
negative_input = Input((1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL, ), name='negative_input')

# Shared embedding layer for positive and negative items
Shared_DNN = create_base_network([1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL,])


encoded_anchor = Shared_DNN(anchor_input)
encoded_positive = Shared_DNN(positive_input)
encoded_negative = Shared_DNN(negative_input)


merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')

#Define the S layers after the triplet NN model
finalAct = 'softmax'
sub1 = Dense(output_dim=NUM_CLASSES,use_bias=True,activation=finalAct)(merged_vector)
sub2 = Dense(output_dim=NUM_CLASSES,use_bias=True,activation=finalAct)(merged_vector)
sub3 = Dense(output_dim=NUM_CLASSES,use_bias=True,activation=finalAct)(merged_vector)
sub4 = Dense(output_dim=NUM_CLASSES,use_bias=True,activation=finalAct)(merged_vector) 

sub5 = Dense(output_dim=NUM_CLASSES,use_bias=True,activation=finalAct)(merged_vector)
sub6 = Dense(output_dim=NUM_CLASSES,use_bias=True,activation=finalAct)(merged_vector)
sub7 = Dense(output_dim=NUM_CLASSES,use_bias=True,activation=finalAct)(merged_vector)
sub8 = Dense(output_dim=NUM_CLASSES,use_bias=True,activation=finalAct)(merged_vector) 

sub9 = Dense(output_dim=NUM_CLASSES,use_bias=True,activation=finalAct)(merged_vector)
sub10 = Dense(output_dim=NUM_CLASSES,use_bias=True,activation=finalAct)(merged_vector)
sub11 = Dense(output_dim=NUM_CLASSES,use_bias=True,activation=finalAct)(merged_vector)
sub12 = Dense(output_dim=NUM_CLASSES,use_bias=True,activation=finalAct)(merged_vector) 

sub13 = Dense(output_dim=NUM_CLASSES,use_bias=True,activation=finalAct)(merged_vector)  
sub14 = Dense(output_dim=NUM_CLASSES,use_bias=True,activation=finalAct)(merged_vector)  

model = Model(inputs=[anchor_input,positive_input, negative_input], 
              outputs=[sub1,sub2,sub3,sub4,sub5,sub6,sub7,sub8,sub9,
                                sub10,sub11,sub12,sub13,sub14])
model.compile(loss=multi_task_loss(model,ALPHA), optimizer=adam_optim)

model.summary()

# =============================================================================
# Construct the training process
# The Si layers are from 5 to 21
# =============================================================================
#from keras.models import model_from_json
#json_file = open('trimtl_model110.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#model = model_from_json(loaded_model_json)
## load weights into new model
#model.load_weights("trimtl_weights110.h5")
#print("Loaded model from disk")
#print(model.summary())

count = 0
epochs = 32


def make_ytrain(y):
    y_onehot = lb.transform(y)
    res = []
    for i in range(14):
        res.append(y_onehot)   
    return res

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
    
#get the triplets for training process
def get_triplets(xtrain):
    Anchor = xtrain[:,0,:].reshape(-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL)
    Positive = xtrain[:,1,:].reshape(-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL)
    Negative = xtrain[:,2,:].reshape(-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL)
    return Anchor,Positive,Negative

# generate triplets and store them in lists: X_train_,y_train_; X_val and y_val just from one subject, and they're arrays
# divide into anchor,positive,and negative as input
def get_train_val(x_train_flat,y_train,x_val_flat,y_val):
    print('start generating triplets')
    m = 280
    n = 120
    X_train_ = []
    y_train_ = []
    X_val_ = []
    y_val_ = []
    anchor_,positive_,negative_,anchor_test,positive_test,negative_test = [],[],[],[],[],[]
    for i in range(14):
        tempx,tempy = generate_triplet(x_train_flat[i],y_train[i].ravel(),ap_pairs=m,an_pairs=m)
        X_train_.append(tempx)
        y_train_.append(tempy)
        tempa,tempp,tempn = get_triplets(tempx)
        anchor_.append(tempa)
        positive_.append(tempp)
        negative_.append(tempn)
        
        tempx,tempy = generate_triplet(x_val_flat[i],y_val[i].ravel(),ap_pairs=n,an_pairs=n)
        X_val_.append(tempx)
        y_val_.append(tempy)
        tempa,tempp,tempn = get_triplets(tempx)
        anchor_test.append(tempa)
        positive_test.append(tempp)
        negative_test.append(tempn)
        
    return anchor_,positive_,negative_,anchor_test,positive_test,negative_test,y_train_,y_val_
      

#modify training process, randomly pick m triplets for training n for testing with each epoch
# to control the training process, need to call model.compile to make 'freeze' effective
while count<epochs:
     
    #Get the training and validation data for current epoch
    anchor_,positive_,negative_,anchor_test,positive_test,negative_test,y_train_,y_val_ = get_train_val(x_train_flat,
                                                                                                        y_train,
                                                                                                        x_val_flat,
                                                                                                        y_val)
    #freeze all Si layers(from 5 to 20)
    for i in range(5,19):
            model.layers[i].trainable = False
            
    #fix all Si and update L first
    #if count>0:
    trainset_anchor = convert_list(anchor_)
    trainset_Positive = convert_list(positive_)
    trainset_Negative = convert_list(negative_)
    validation_anchor = convert_list(anchor_test)
    validation_positive = convert_list(positive_test)
    validation_negative = convert_list(negative_test)
    traintarget = convert_y(y_train_)
    target_onehot = make_ytrain(traintarget)
    testtarget = convert_y(y_val_)
    test_onehot = make_ytrain(testtarget)
    print(trainset_anchor.shape,traintarget.shape)
    #make L layers trainable
    for i in range(9):
        Shared_DNN.layers[i].trainable = True
    #model.compile(loss=custom_loss(model),
    model.compile(loss=multi_task_loss(model,ALPHA),
      optimizer = adam_optim)
    with tf.device('/gpu:0'): 
        model.fit([trainset_anchor,trainset_Positive,trainset_Negative],
              y=target_onehot,validation_data=([validation_anchor,validation_positive,validation_negative],
                                     test_onehot), 
              batch_size=512, 
              epochs=1)
    
    #Before training Si, freeze L
    for i in range(9):
        Shared_DNN.layers[i].trainable = False
    
    #train Si(i=5 to 20) branch, freeze all the other S branches
    for j in range(5,19):
        #freeze all Si
        for i in range(5,19):
            model.layers[i].trainable = False
        #make current Si trainable
        model.layers[j].trainable = True
        trainset_anchor = anchor_[j-5]
        trainset_Positive = positive_[j-5]
        trainset_Negative = negative_[j-5]
        y=make_ytrain(y_train_[j-5])
        
        validation_anchor = anchor_test[j-5]
        validation_Positive = positive_test[j-5]
        validation_Negative = negative_test[j-5]
        test_ = make_ytrain(y_val_[j-5])
       
        model.compile(loss=multi_task_loss(model,ALPHA),
          metrics = ['accuracy'], optimizer=adam_optim)
        with tf.device('/gpu:0'): 

            # is that necessary to do cross validation for each branch?
            model.fit([trainset_anchor,trainset_Positive,trainset_Negative],y,
                      validation_data=([validation_anchor,validation_Positive,validation_Negative],
                                     test_),
                      batch_size=512, 
                      epochs=1)
     
    #save model to file every epochs
#    trained_model = Model(inputs=anchor_input, outputs=[sub1,sub2,sub3,sub4,sub5,sub6,sub7,sub8,sub9,
#                                sub10,sub11,sub12,sub13,sub14,sub15,sub16])
    model_json = model.to_json()
    with open("trimtl_model_usall.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("trimtl_weights_usall.h5") 
    
    if count%9 == 0:
        gc.collect() 
        
    count+=1
    print('This is the '+str(count)+' iteration')
    #delete those big lists
    anchor_,positive_,negative_,anchor_test,positive_test,negative_test,y_train_,y_val_ =[],[],[],[],[],[],[],[]
    

# serialize model to JSON
model_json = model.to_json()
with open("trimtl_modelusfinalall.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("trimtl_weightsusfinalall.h5")
print("Saved model to disk")
