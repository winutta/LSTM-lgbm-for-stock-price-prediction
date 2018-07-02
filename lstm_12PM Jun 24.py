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

def get_data(seq_len,gap, split = 0.9, threedarray = True):

    print("The lag for prediction is: "+str(gap))
    print("The sequence lenth for inputs is: "+str(seq_len))

    start = datetime.datetime(2017, 9, 1)
    end = datetime.datetime(2018, 6, 15)
    stock = "F"
    # Creates a pandas DataFrame object with stock price and other related data such as Volume and dividends
    df = data.DataReader(stock, 'quandl', start, end)
    print("Predictive Model for stock ticker symbol: "+stock+" from "+str(start)+" until "+str(end)+".")
    print("The amount of total samples is: "+str(len(df["AdjOpen"])-1+seq_len+gap))

    # Reverse the order of rows in the data so that the first row corresponds to the first time point
    
    #Take a subset of dataset with relevant columns
    relevant_columns = ["AdjOpen", "AdjHigh", "AdjLow", "AdjClose", "AdjVolume"]

    df = df.iloc[::-1]
    df = df.loc[:,relevant_columns]
    base = df
    #df = df.diff().iloc[1:,:]
    
    if threedarray:

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
        row = whole_array.shape[0]-gap
        y_train = y[:int(row)]
        y_test = y[int(row):]
        x_train = whole_array[:int(row),:,:]
        x_test = whole_array[int(row):,:,:]

        from sklearn.utils import shuffle
        x_train, y_train = shuffle(x_train, y_train, random_state=0)

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

    else:
        #Creates lists for each variable for easy concatenation

        AO = df["AdjOpen"].tolist()
        AH = df["AdjHigh"].tolist()
        AL = df["AdjLow"].tolist()
        AC = df["AdjClose"].tolist()
        AV = df["AdjVolume"].tolist()
        AO_target = AO[seq_len+gap:]

        #Creates Sequence Features

        result = []
        for row in range(len(AO)-seq_len-gap):
            result.append(AO[row:row+seq_len]+AH[row:row+seq_len]+AL[row:row+seq_len]+AC[row:row+seq_len]+AV[row:row+seq_len]+[AO_target[row]])

        future_inputs = []
        for i in range(gap):
            future_inputs.append(AO[-gap+i-seq_len:-gap+i]+AH[-gap+i-seq_len:-gap+i]+AL[-gap+i-seq_len:-gap+i]+AC[-gap+i-seq_len:-gap+i]+AV[-gap+i-seq_len:-gap+i])

        result = np.array(result)
        future_inputs = np.array(future_inputs)

        #Split arrays into x and y and then train and test sets for each

        row = round(split * whole_array.shape[0])
        train = result[:int(row), :]
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = result[int(row):, :-1]
        y_test = result[int(row):, -1]

        #Train minmaxscaler on x_train, and apply to x_train, x_test, and future set

        scaler = preprocessing.MinMaxScaler((-1,1))
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        future_inputs = scaler.transform(future_inputs)

        #Reshape data to fit in NN

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
        future_inputs = np.reshape(future_inputs, (future_inputs.shape[0], future_inputs.shape[1],1))
        y_train = np.reshape(y_train, (y_train.shape[0],1))
        y_test = np.reshape(y_test, (y_test.shape[0],1))

        return [x_train, y_train, x_test, y_test, future_inputs]

#Builds RNN with LSTM

def build_model(layers,batch,steps,learning_rate = 0.001,epochs = 500):
    model = Sequential()

    # model.add(Dense(
    #     batch_input_shape = (batch,steps,5),
    #     input_shape=(layers[1], layers[0]),
    #     output_dim = layers[2],activation = "linear"))
    # #     ))
    model.add(LSTM(
        batch_input_shape = (batch,steps,5),
        output_dim=layers[2],
        return_sequences=False,
        stateful = True))
    # model.add(Dropout(0.2))

    # model.add(LSTM(
    #     output_dim = layers[3],
    #     return_sequences=False,
    #     stateful= True))
    # model.add(Dropout(0.2))

    # model.add(LSTM(
    #     2,
    #     return_sequences=False))
    # model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3],activation="relu"))
    # model.add(Activation("relu"))
    start = time.time()
    
    rms = RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    nadam = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss="mse", optimizer=adam)
    print("> Compilation Time : ", time.time() - start)

    return model

#Was used to normalize data, but not currently used
def normalise_windows(window_data, how = "standard"):
    # normalised_data = []
    # multipliers = []
    if how == "standard":
        return preprocessing.MinMaxScaler((-1,1)).fit_transform(window_data)
    normalised_data = []
    multipliers = []
    if how == "percentage":
        for window in window_data:
            normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
            normalised_data.append(normalised_window)
            multipliers.append(float(window[0]))
        return normalised_data



def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def predict_lag(model,data):
    predictions = []
    data_len = len(data[0])
    for i in range(len(data)):
        predictions.append(model.predict(data[i][newaxis,:,:])[0,0])
    return predictions

def recover_prices(data,multipliers):
    prices = []
    for i in range(len(data)):
        prices.append((data[i]+1)*multipliers[i])
    return prices

def lGBM(X,y):
    X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.10, random_state=23)

    lgbm_params =  {
    'task': 'train',
    'boosting_type': 'dart',
    'objective': 'regression',
    'metric': 'rmse',
    'max_depth': 15,
    'num_leaves': 37,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    # 'bagging_freq': 5,
    'learning_rate': 0.019,
    'verbose': 1
    }  
    lgtrain = lgb.Dataset(X_train, y_train)
    lgvalid = lgb.Dataset(X_valid, y_valid)
    print(X_train.shape)
    modelstart = time.time()
    lgb_clf = lgb.train(
    lgbm_params,
    lgtrain,
    num_boost_round=16000,
    valid_sets=[lgtrain, lgvalid],
    valid_names=['train','valid'],
    early_stopping_rounds=1000,
    verbose_eval=1
    )
    print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))

    return lgb_clf

def predict_lgbm(lgb_clf,testing):
    lgpred = lgb_clf.predict(testing,num_iteration=lgb_clf.best_iteration)
    return lgpred
