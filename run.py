import lstm
import time
import matplotlib.pyplot as plt
import keras as keras
from keras.utils.vis_utils import plot_model
from sklearn.metrics import mean_squared_error
import datetime
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
import pandas as pd
from hyperopt import Trials, hp, STATUS_OK, fmin, tpe
import sys
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop,Nadam,Adam,Adagrad
from keras.models import Sequential

def plot_results(predicted_data, true_data,future):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(211)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    padding = [None for p in range(len(true_data))]
    plt.scatter(range(len(padding+future)),padding + future, label='Prediction_f')
    plt.legend()
    plt.show()

def plot_results_no_future(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(211)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_future(predicted_data):
	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(111)
	plt.scatter(range(len(predicted_data)),predicted_data,label='Prediction')
	plt.legend()
	plt.show()

#-------------Control-Panel#
#__________________________#

epochs  = 6000
seq_len = 45
node = 30
batch_sizes = 1
gap = 4
patience = 5
learning_rate = 0.0003
start = datetime.datetime(2017, 9, 1)
end = datetime.datetime(2018, 6, 15)
stock = "F"
layers = [5, seq_len, node, 1]
#__________________________#
#--------------------------#

print("Node :" +str(node),"Patience :"+str(patience))

def send(seq_len=seq_len,layers = layers, gap = gap, batch = batch_sizes,start = start,end = end, stock = stock):
	return [seq_len,layers,gap,batch,start,end,stock]

if __name__=='__main__':
	global_start_time = time.time()

	# print('> Data Loaded. Compiling...')
	X_train, y_train, X_test, y_test, ender = lstm.get_data(seq_len = seq_len,split = .8,gap = gap, start = start, end = end, stock = stock)
	model = lstm.build_model(layers,batch = batch_sizes,steps = seq_len,learning_rate = learning_rate)
	print(model.summary())
	plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
	
	print("neurons: "+str(layers[0]*layers[1])+", "+str(layers[2]))
	print("Learning Rate: "+ str(learning_rate))
	print("X_train shape: "+str(X_train.shape))
	print("X_test shape : "+str(X_test.shape))
	call = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='min')
	# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, min_lr=0.000001, verbose=1) 
	checkpointer = keras.callbacks.ModelCheckpoint(filepath="weights.hdf5",monitor='val_loss', verbose=1, save_best_only=True)
	
	# print(y_train.shape,X_train.shape,y_test.shape,X_test.shape)
	history = model.fit(
	    X_train,
	    y_train,
	    nb_epoch = epochs,
	    batch_size= batch_sizes,
	    validation_split = 0.2,
	    callbacks = [call,checkpointer]
	    )
	plt.plot(history.history['loss'],color = "r")
	plt.plot(history.history['val_loss'],color = "b")
	plt.show()

	model.load_weights('weights.hdf5')
	
	train_pred = lstm.predict_lag(model,X_train)
	plot_results_no_future(train_pred,y_train)
	predictions = lstm.predict_lag(model,X_test)
       
	future = lstm.predict_lag(model, ender)
	print('Training duration (s) : ', time.time() - global_start_time)


	plot_results(predictions,y_test,future)
	plot_future(future)
	
