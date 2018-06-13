import lstm
import time
import matplotlib.pyplot as plt

def plot_results(predicted_data, true_data,future):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(211)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    padding = [None for p in range(len(true_data))]
    plt.plot(padding + future.tolist(), label='Prediction_f')
    plt.legend()
    plt.show()

def plot_future(predicted_data):
	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(111)
	plt.plot(predicted_data,label='Prediction')
	plt.legend()
	plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

#Main Run Thread
if __name__=='__main__':
	global_start_time = time.time()
	epochs  = 40
	seq_len = 14

	print('> Loading data... ')

	#, train_result_mult, test_result_mult, end_mult
	#new, X_train, y_train, X_test, y_test, ender = lstm.gap_load_data('sp500.csv', seq_len, False,5)
	X_train, y_train, X_test, y_test, ender = lstm.get_data(seq_len,7)
	#plot_future(new)
	print('> Data Loaded. Compiling...')

	# model = lstm.build_model([1, seq_len*5, 50, 1])

	# model.fit(
	#     X_train,
	#     y_train,
	#     batch_size=64,
	#     nb_epoch=epochs,
	#     validation_split=0.05)

	model = lstm.lGBM(X_train,y_train)

	#predictions = lstm.predict_lag(model,X_test)
	predictions = lstm.predict_lgbm(model,X_test)
	future = lstm.predict_lgbm(model,ender)
	#predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
	#predicted = lstm.predict_sequence_full(model, X_test, seq_len)
	#predicted = lstm.predict_point_by_point(model, X_test)        
	#future = lstm.predict_lag(model, ender)
	print('Training duration (s) : ', time.time() - global_start_time)

	# true_pred = lstm.recover_prices(predictions,test_result_mult,a = True)
	# true_future = lstm.recover_prices(future,end_mult,a = True)
	# true_prices = lstm.recover_prices(y_test,test_result_mult,a = True)
	#plot_results_multiple(predictions, y_test, 50)
	plot_results(predictions,y_test,future)
	plot_future(future)
	
