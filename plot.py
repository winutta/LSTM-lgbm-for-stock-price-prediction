# Print graphs
import lstm
import time
import matplotlib.pyplot as plt
import keras as keras
import run


l = run.send()
seq_len=l[0]
layers = l[1]
gap = l[2]
batch_size = l[3]
start = l[4]
end = l[5]
stock = l[6]

X_train, y_train, X_test, y_test, ender = lstm.get_data(seq_len = seq_len,split = .8, gap = gap, start = start, end = end, stock = stock)

model = lstm.build_model(layers,batch = batch_size,steps = seq_len)
model.load_weights('weights.hdf5')
train_pred = lstm.predict_lag(model,X_train)
run.plot_results_no_future(train_pred,y_train)
predictions = lstm.predict_lag(model,X_test)
 
future = lstm.predict_lag(model, ender)


run.plot_results(predictions,y_test,future)
run.plot_future(future)