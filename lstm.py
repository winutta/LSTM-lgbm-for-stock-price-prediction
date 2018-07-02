import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop,Nadam,Adam
from keras.models import Sequential
from sklearn import preprocessing
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pandas_datareader import data
import datetime
from sklearn import metrics
from sklearn.model_selection import train_test_split
import lightgbm as lgb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

#Function to create the data

def difference(dataset, interval=1):
    diff = list()
    for i in range(len(dataset)-1):
        value =  dataset[i + interval] - dataset[i] 
        diff.append(value)
    return pd.Series(diff)

def get_data(seq_len,gap, split = 0.9,stock = "F", start = datetime.datetime(2017, 9, 1), end = datetime.datetime(2018, 6, 15)):

    print("The lag for prediction is: "+str(gap))
    print("The sequence lenth for inputs is: "+str(seq_len))

    
    # Creates a pandas DataFrame object with stock price and other related data such as Volume and dividends
    df = data.DataReader(stock, 'quandl', start, end)
    print("Predictive Model for stock ticker symbol: "+stock+" from "+str(start)+" until "+str(end)+".")
    print("The amount of total price samples is: "+str(len(df["AdjOpen"])-1-seq_len-gap))

    # Reverse the order of rows in the data so that the first row corresponds to the first time point
    
    #Take a subset of dataset with relevant columns
    relevant_columns = ["AdjOpen", "AdjHigh", "AdjLow", "AdjClose", "AdjVolume"]

    df = df.iloc[::-1]
    df = df.loc[:,relevant_columns]
    df = df[1:]
    base = df
    
    '''
    create a 3d numpy array for the whole dataset of shape(timepoints-sequence-gap,seq_index,feature)
    contains inputs for all the outputs which have a known price and is possible to predict from other
    known data, for train+test sets
    '''
    whole_array = np.zeros((len(df["AdjOpen"])-seq_len-gap,seq_len,1))
    for i in relevant_columns:
        result = []
        d = df[i].tolist()
        for j in range(len(d)-seq_len-gap):
            result.append(d[j:j+seq_len])
        a = np.array(result)
        a = np.reshape(a,(a.shape[0],a.shape[1],1))
        whole_array = np.dstack((whole_array,a))
    whole_array = whole_array[:,:,1:] 

    future_inputs = np.zeros((gap,seq_len,1))
    for i in relevant_columns:
        result = []
        d = df[i].tolist()
        for j in range(gap):
            result.append(d[-gap+j-seq_len:-gap+j])
        a = np.array(result)
        a = np.reshape(a,(a.shape[0],a.shape[1],1))
        future_inputs = np.dstack((future_inputs,a))
    future_inputs = future_inputs[:,:,1:]

    y = np.array(df["AdjOpen"][seq_len+gap:])
    y = np.reshape(y,(y.shape[0],1))
    
    #Split Splits inputs and outputs into train,test sets
    row = round(split * whole_array.shape[0])
    y_train = y[:int(row)]
    y_test = y[int(row):]
    x_train = whole_array[:int(row),:,:]
    x_test = whole_array[int(row):,:,:]

    '''
    Calculate min and max of train set of each feature 
    to scale x_train,x_test, and future_inputs
    Fits minmaxscaler on train set and applies to train, test, and future sets
    '''
    x_min = x_train.min(axis=(0, 1), keepdims=True)
    x_max = x_train.max(axis=(0, 1), keepdims=True)

    x_train = 2*(x_train-x_min)/(x_max-x_min)-1
    x_test = 2*(x_test-x_min)/(x_max-x_min)-1
    future_inputs = 2*(future_inputs-x_min)/(x_max-x_min)-1


    return [x_train, y_train, x_test, y_test, future_inputs]


#Builds RNN with LSTM

def build_model(layers,batch = 1,steps=20,learning_rate = .01):
    model = Sequential()
    model.add(LSTM(
        batch_input_shape = (batch,steps,5),
        output_dim=layers[2],
        return_sequences=False,
        stateful = True))

    model.add(Dense(
        output_dim=layers[3],activation="relu"))
    
    start = time.time()
    
    rms = RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    # nadam = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    # adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=.002, decay=0.0, amsgrad=True)
    model.compile(loss="mse", optimizer=rms)

    return model

def predict_lag(model,data):
    predictions = []
    data_len = len(data[0])
    for i in range(len(data)):
        predictions.append(model.predict(data[i][newaxis,:,:])[0,0])
    return predictions

