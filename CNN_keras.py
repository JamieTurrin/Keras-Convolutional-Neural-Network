# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:47:27 2017

@author: James

Convolutional Neural Network (CNN) implemented using keras package.
See https://keras.io/models/model  for documentation on keras.

Keras is a high-level framework for building deep neural networks quickly
and can run on top of TensorFlow if desired

Identify images in which people are 'happy'
Dataset is 750 RGB photos of faces, 600 for training, 150 for test set

Achieves ~98% train accuracy and ~95% test accuracy

"""
import numpy as np
import h5py
import matplotlib.pyplot as plt

################### LOAD DATASET FUNCTION #####################################

def load_dataset():
    train_dataset = h5py.File('C:/users/james/desktop/Neural Networks AI/Conv_NNs/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('C:/users/james/desktop/Neural Networks AI/Conv_NNs/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

################ IMPORT KERAS MODULES ###############################################

#from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D  
#from keras.layers import AveragePooling2D, Dropout, GlobalMaxPooling2D,GlobalAveragePooling2D

from keras.models import Model

import keras.backend as K
K.set_image_data_format('channels_last')


#################### DATA PREPROCESSING #######################################

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape to dim(num_samples, n_H, n_W, n_C)
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

"""
number of training examples = 600
number of test examples = 150
X_train shape: (600, 64, 64, 3)
Y_train shape: (600, 1)
X_test shape: (150, 64, 64, 3)
Y_test shape: (150, 1)
"""

################# DEFINE THE KERAS MODEL ######################################

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset, a tuple (n_H, n_W, n_C)

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    # For 3x3 Conv2D filter below, 'same' padding will be (3-1)/2=1
    X = ZeroPadding2D((1,1))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(16, (3,3), strides = (1, 1), name = 'conv1')(X) # 16 filters of 3x3
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)

    # MAXPOOL over a 2x2 area
    X = MaxPooling2D((2, 2), name='max_pool1')(X)
    
    # CONV -> BN -> RELU 
    X = Conv2D(32, (3,3), strides = (1, 1), name = 'conv2')(X) # 32 filters of 3x3
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X) 
    
    # MAXPOOL over a 2x2 area
    X = MaxPooling2D((2, 2), name='max_pool2')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED 
    X = Flatten()(X)  # flatten images before inserting into fully connected layer
    X = Dense(1, activation='sigmoid', name='fc1')(X) # 1 output node
    

    # Create model. This creates your Keras model instance, 
    # you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')
    
    return model

#################### CALL THE MODEL FUNCTION ##################################
happyModel = HappyModel((64,64,3))

##################### COMPILE THE MODEL #######################################
happyModel.compile(optimizer='Adam',loss='mean_squared_error',metrics=['accuracy'])

################### TRAIN THE MODEL ###########################################
happyHistory = happyModel.fit(x=X_train, y=Y_train, epochs=10, batch_size=16)

##################### TEST THE MODEL ACCURACY ##########################################
happyModel.evaluate(x=X_test, y=Y_test) # returns loss and test accuracy

#################### PREDICTIONS ##############################################
# returns array of dim(num_test_samples, num_classes)
# each row has a decimal between 0 and 1 indicating percent likelihood that
# sample belongs to a given class
predictions = happyModel.predict(X_test)

#################### MODEL SUMMARY ############################################
# print the details of the model layers with sizes of inputs/outputs and
# number of parameters in each layer
happyModel.summary()

########################## PLOTS OF LOSS AND ACCURACY #########################
# print dictionary of history keys, dictionary has 'loss' and 'acc'
print(happyHistory.history.keys()) # print dictionary of history keys

plt.plot(happyHistory.history['acc'])  # plot accuracy
plt.plot(happyHistory.history['loss']) # plot loss
plt.title('Evolution of Accuracy and Loss')
plt.xlabel('Epoch')
plt.legend(['Accuracy','Loss'])
plt.show()



























