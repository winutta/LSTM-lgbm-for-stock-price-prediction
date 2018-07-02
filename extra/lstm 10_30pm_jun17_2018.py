import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn import preprocessing
import matplotlib.pyplot as plt

import pandas as pd
from pandas_datareader import data
import datetime
from sklearn import metrics
from sklearn.model_selection import train_test_split
import lightgbm as lgb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def get_data(seq_len,gap):
    start = datetime.datetime(2016, 6, 1)
    mid = datetime.datetime(2017,6,1)
    end = datetime.datetime(2018, 1, 1)
    print("The lag for prediction is: "+str(gap))
    #fm = data.DataReader('F', 'morningstar', start, end)

    # Creates a pandas DataFrame object with stock price and other related data such as Volume and dividends
    df = data.DataReader('F', 'quandl', start, end)


    # Reverse the order of rows in the data so that the first row corresponds to the first time point
    df = df.iloc[::-1]
    # plt.plot(df["AdjOpen"])
    # plt.show()
    relevant_columns = ["AdjOpen", "AdjHigh", "AdjLow", "AdjClose", "AdjVolume"]
    df = df.loc[:,relevant_columns]
    #print(df.head())

    AO = df["AdjOpen"].tolist()
    AH = df["AdjHigh"].tolist()
    AL = df["AdjLow"].tolist()
    AC = df["AdjClose"].tolist()
    AV = df["AdjVolume"].tolist()
    AO_target = AO[seq_len+gap:]
    #print(AO)
    nAO = np.array(df["AdjOpen"])
    
    fin_arr = np.zeros((len(AO)-seq_len-gap,seq_len,1))
    for i in relevant_columns:
        result = []
        d = df[i].tolist()
        for j in range(len(d)-seq_len-gap):
            result.append(d[j:j+seq_len])
        a = np.array(result)
        a = np.reshape(a,(a.shape[0],a.shape[1],1))
        fin_arr = np.dstack((fin_arr,a))

    fin_arr = fin_arr[:,:,1:]   
    print(fin_arr[0])
    print(fin_arr.shape)


    
    result = []
    for row in range(len(AO)-seq_len-gap):
        result.append(AO[row:row+seq_len]+AH[row:row+seq_len]+AL[row:row+seq_len]+AC[row:row+seq_len]+AV[row:row+seq_len]+[AO_target[row]])

    
    ender = []
    for i in range(gap):
        ender.append(AO[-gap+i-seq_len:-gap+i]+AH[-gap+i-seq_len:-gap+i]+AL[-gap+i-seq_len:-gap+i]+AC[-gap+i-seq_len:-gap+i]+AV[-gap+i-seq_len:-gap+i])


    result = np.array(result)
    # print(result.shape)
    # print(type(result))
    # print(result[0])
    # print(result[0][0])
    ender = np.array(ender)
    row = round(0.9 * result.shape[0])

    # train_res_mult = result_mult[:int(row)]
    # test_res_mult = result_mult[int(row):]

    train = result[:int(row), :]
    #np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    # print(x_train.shape)
    # print(x_train[0].shape)

    scaler = preprocessing.MinMaxScaler((-1,1))
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    ender = scaler.transform(ender)

    # x_train = normalise_windows(x_train)
    # x_test = normalise_windows(x_test,t = "test")
    # ender = normalise_windows(ender)

    # x_train = np.array(x_train)
    # x_test = np.array(x_test)
    # ender = np.array(ender)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
    ender = np.reshape(ender, (ender.shape[0], ender.shape[1],1))  
    y_train = np.reshape(y_train, (y_train.shape[0],1))
    y_test = np.reshape(y_test, (y_test.shape[0],1))
    #train_res_mult, test_res_mult, end_mult

    return [x_train, y_train, x_test, y_test, ender]


def gap_load_data(filename, seq_len, normalise_window,gap):
    f = open(filename, 'rb').read()
    data = f.decode().split('\n')
    new = np.array(data[:-1])
    new1 = []
    for i in new:
        new1.append(float(i))
    
    new1 = np.array(new1).reshape(-1,1)

    scalerX = StandardScaler().fit(new1)
    data = scalerX.transform(new1)
    data = data.tolist()
    
    
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - seq_len-(gap)):
        result.append(data[index: index + seq_len]+[data[index+seq_len+(gap)]])
    
    ender = []
    for i in range(gap):
        ender.append(data[-gap+i-seq_len:-gap+i])

    if normalise_window:
        result,result_mult = normalise_windows(result)
        ender,end_mult = normalise_windows(ender)

    result = np.array(result)
    
    ender = np.array(ender)
    row = round(0.7 * result.shape[0])

    # train_res_mult = result_mult[:int(row)]
    # test_res_mult = result_mult[int(row):]

    train = result[:int(row), :]
    #np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    ender = np.reshape(ender, (ender.shape[0], ender.shape[1], 1))  

    #train_res_mult, test_res_mult, end_mult

    return [new1, x_train, y_train, x_test, y_test, ender]

#def load_data(filename, seq_len, normalise_window):
#     f = open(filename, 'rb').read()
#     data = f.decode().split('\n')


#     sequence_length = seq_len + 1
#     result = []
#     for index in range(len(data) - sequence_length):
#         result.append(data[index: index + sequence_length])
    
#     if normalise_window:
#         result = normalise_windows(result)

#     result = np.array(result)

#     row = round(0.9 * result.shape[0])
#     train = result[:int(row), :]
#     np.random.shuffle(train)
#     x_train = train[:, :-1]
#     y_train = train[:, -1]
#     x_test = result[int(row):, :-1]
#     y_test = result[int(row):, -1]


#     x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#     x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

#     return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data,t, how = True,):
    # normalised_data = []
    # multipliers = []

    return preprocessing.MinMaxScaler((-1,1)).fit_transform(window_data)


    # for window in window_data:
    #     normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
    #     normalised_data.append(normalised_window)
    #     multipliers.append(float(window[0]))
    # return normalised_data

def build_model(layers,batch,steps):
    model = Sequential()

    model.add(Dense(
        batch_input_shape = (batch,steps,1),
        input_shape=(layers[1], layers[0]),
        output_dim = layers[1]
        ))
    model.add(LSTM(
        output_dim=layers[2],
        return_sequences=False,
        stateful = True))
    # model.add(Dropout(0.2))

    # model.add(LSTM(
    #     layers[2],
    #     return_sequences=False))
    # model.add(Dropout(0.2))

    # model.add(LSTM(
    #     2,
    #     return_sequences=False))
    # model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    # model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

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
