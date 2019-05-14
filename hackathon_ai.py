import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
from collections import Counter
import time
from sklearn.model_selection import TimeSeriesSplit
np.random.seed(1)
import random as rn
rn.seed(1)
import tensorflow as tf
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K
tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.layers import Dense
from keras.models import Sequential
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers
from keras import metrics
from sklearn.preprocessing import  StandardScaler
from sklearn.decomposition import PCA
import numpy.lib.recfunctions as rf
from sklearn import utils
from keras.callbacks import EarlyStopping
import scipy as sp
import keras as k
from keras.initializers import RandomNormal
from sklearn.utils.class_weight import compute_class_weight
import keras_metrics as km



#Plot all features
bigdata_t = nf_data.append(af_data, ignore_index=True)
bigdata=bigdata_t.sort_values(by='Time')
bigdata=bigdata.reset_index (drop = True)
bigdata.set_index('Time', inplace=True)
bigdata.index=pd.to_datetime(bigdata.index)

final_dataset = bigdata
final_dataset['Anomaly'][final_dataset['Anomaly']>1]=1
#final_dataset=final_dataset.astype('float16')

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

hour_prior = 12
X_sts = series_to_supervised(final_dataset, 6*hour_prior, 1, dropnan=True)
lagged = [col for col in X_sts.columns if '(t-' in col] #all those values are .shifted by some time
X_lagged = X_sts[lagged]
targets = [col for col in X_sts.columns if '(t)' in col] #our original values
target_original = X_sts[targets]
y_original_class = target_original.iloc[:,-1] #I am focusing on predicting only the fault
X=X_lagged.values
y=y_original_class.values
print (X.shape)
print (y_original_class.shape)

train_size = int(0.70*X.shape[0])
X_train, X_test, y_train, y_test = X[0:train_size], X[train_size:],y[0:train_size], y[train_size:]

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
X_train_pt = pt.fit_transform(X_train)
X_test_pt = pt.transform(X_test)

X_train,X_test = X_train_pt,X_test_pt

model = Sequential()
model.add(Dense(35, activation='relu', input_shape=(X_train.shape[1],),
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=42),
                bias_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=42)))

model.add(Dense(35, activation='relu',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=42),
                bias_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=42)))

model.add(Dense(1, activation='sigmoid'))

model.compile (loss = 'sparse_categorical_crossentropy', optimizer=k.optimizers.Adam(lr=1e-4))

early_stopping_monitor = EarlyStopping(monitor='val_loss', mode='min', patience=5)

history = model.fit(X_train, y_train, epochs=1000, class_weight=class_weights, batch_size=32,
                    validation_data=(X_test, y_test), verbose = 1, callbacks=[early_stopping_monitor])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

y_pred = model.predict(X_test)
y_pred = y_pred.round()
y_pred = y_pred.argmax(axis=1)

from sklearn.metrics import r2_score
print (r2_score(y_test, y_pred))
