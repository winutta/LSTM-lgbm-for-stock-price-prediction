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

epochs  = 30
seq_len = 45
node = 30
batch_sizes = 1
gap = 4
patience = 100
learning_rate = 0.000599
start = datetime.datetime(2017, 9, 1)
end = datetime.datetime(2018, 6, 15)
stock = "F"
#__________________________#
#--------------------------#



def build_model(params):
    model = Sequential()
    model.add(LSTM(
        batch_input_shape = (batch_sizes,params["seq_len"],5),
        output_dim=params["node"],
        return_sequences=False,
        stateful = True))
    print(params)
    model.add(Dense(
        output_dim=1,activation="relu"))
    start = datetime.datetime(2017, 9, 1)
    end = datetime.datetime(2018, 6, 15)
    X_train, y_train, X_test, y_test, ender = lstm.get_data(seq_len = params["seq_len"],split = .8,gap = params["gap"], start = start, end = end, stock = stock)
    start = time.time()
    
    rms = RMSprop(lr=params["learning_rate"], rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss="mse", optimizer=rms)
    call = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='min')
    checkpointer = keras.callbacks.ModelCheckpoint(filepath="weights.hdf5",monitor='val_loss', verbose=1, save_best_only=True)
    history = model.fit(X_train, y_train, 
    	nb_epoch=epochs,
    	batch_size=batch_sizes, 
    	verbose = 1, 
    	validation_data = (X_test, y_test),
    	callbacks = [call,checkpointer]
    	)
    # plt.plot(history.history['loss'],color = "r")
    # plt.plot(history.history['val_loss'],color = "b")
    # plt.show()
    model.load_weights('weights.hdf5')
    preds  = model.predict(X_test, batch_size = batch_sizes, verbose = 1)
    acc = mean_squared_error(y_test, preds)
    print('MSE:', acc)
    sys.stdout.flush()
    return {'loss': acc, 'status': STATUS_OK}

print("Node :" +str(node),"Patience :"+str(patience))

def send(seq_len=seq_len,node = node,gap = gap):
	return [seq_len,node,gap]

if __name__=='__main__':
	global_start_time = time.time()

	# "optimizer": hp.choice("optimizer",[ 'Adam', 'Adamax','SGD', 'RMSprop', 'Nadam','Adagrad', 'Adadelta']),

	space = {"gap": 1 + hp.randint("gap",10),
	"learning_rate": hp.uniform("learning_rate", 0.0001, 0.001),
	"node": 21 + hp.randint("node",150),
	"seq_len": 10 +hp.randint("seq_len",50)}
	trials = Trials()
	best = fmin(build_model, space, algo=tpe.suggest, max_evals = 100, trials=trials)
	print(trials.trials)
	print('best: ')
	print(best)

