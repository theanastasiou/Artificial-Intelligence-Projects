import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

import matplotlib as mpl
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping
import seaborn as sns
from math import sqrt
from numpy import mean


from keras.preprocessing.sequence import TimeseriesGenerator

def multivariate_data(dataset, target, look_back,
                      target_size, step, single_step=False):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        indices = range(i-look_back, i, step)
        X.append(dataset[indices])
        if single_step:
            Y.append(target[i+target_size])
        else:
            Y.append(target[i:i+target_size])
    return np.array(X), np.array(Y)

# def multivariate_data(dataset, target, start_index, end_index, history_size,
#                       target_size, step, single_step=False):
#   data = []
#   labels = []
#   print('IN')
#   print(dataset)
#   print(target)
#   start_index = start_index + history_size
#   print(start_index)
#   if end_index is None:
#     end_index = len(dataset) - target_size
#     print(end_index)
#   for i in range(start_index, end_index):
#     indices = range(i-history_size, i, step)
#     print(indices)
#     data.append(dataset[indices])
#     if single_step:
#       labels.append(target[i+target_size])
#     else:
#       labels.append(target[i:i+target_size])
#   print(data)
#   print(labels)
 # return np.array(data), np.array(labels)

#dimioyrgia time steps g ta plots
def create_time_steps(length):
  return list(range(-length, 0))

#plot for train history
def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()

#plot for univariate
def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

#plot for multistep
def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 1]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()


#set columns
cols = ['Global_active_power','Global_reactive_power','Sub_metering_1','Sub_metering_2','Sub_metering_3']

path = r'/home/at/Desktop/Texniti_2/' # use your path'
filename = path + "3MonthGrouping.csv"
dataset1 = pd.read_csv(filename, header=0, low_memory=False)
# print(len(dataset1))
#dataset['DateTime'] =pd.to_datetime(dataset['DateTime'])
# dataset.set_index(pd.DatetimeIndex(dataset['DateTime']),inplace=True)
# dataset.set_index(dataset['Month'],inplace=True)

tf.random.set_seed(13)
# print(dataset1.isna().sum())

features_considered = ['Sub_metering_1','Sub_metering_2','Sub_metering_3'] #stiles stis opoies tha ginei to train k to predict - i se mia i se polles
features = dataset1[features_considered]
features.index = dataset1['DateTime'] #set indext to datetime

# dataset = dataset.astype('float32')

# features.plot(subplots=True)
# plt.show()


scaler = MinMaxScaler(feature_range=(0, 1)) #normalize ta dedomena me xrisi tu minmaxscaler
dataset = scaler.fit_transform(features) 
# train_size = int(len(dataset) * 0.80)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# TRAIN_SPLIT = 1344
# dataset = features.values
# data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
# data_std = dataset[:TRAIN_SPLIT].std(axis=0)
# dataset = (dataset-data_mean)/data_std

#print(len(dataset))

past_history =  1440 #arxika 51855 grammes ara peripou 36 mines
STEP = 1 #12 mines
EPOCHS = 40  #epochs
EVALUATION_INTERVAL = 40
future_target = 60 #auta pou thelw n predict meta - tosa samples
BATCH_SIZE = 5 #batch size g to training 
BUFFER_SIZE = 100 #buffer size g tin cache()


#an einai gia univarite tote perniete to dataset sto multivariate_data alliws to datasett[:,1]
if(len(features_considered)>1):
    data = dataset[:, 1]
else:
  data = dataset

#create data
x_train_multi, y_train_multi = multivariate_data(dataset, data , past_history,future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, data,  past_history, future_target, STEP)

print ('Single window of past history : {}'.format(x_train_multi[0].shape)) #print shape tu past history
print ('\n Target data to predict : {}'.format(y_train_multi[0].shape)) #to shape tu target p tha ginei predict


#cache() & repeat gia ligoteri mnimi - amesi prosvasi 
train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

# print(train_data_multi)

if(len(features_considered)>1):
  for x, y in train_data_multi.take(1):
    multi_step_plot(x[0], y[0], np.array([0]))
else:
  print('univariateplot')
  # show_plot([x_train_multi[0], y_train_multi[0]], 0, 'Sample Example')


#MODEL
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(16,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu',dropout=0.2))
if(len(features_considered)==1):
  multi_step_model.add(tf.keras.layers.Dense(1))
else:
  multi_step_model.add(tf.keras.layers.Dense(future_target))


#compile model with SGD Optimizer
multi_step_model.compile(optimizer=tf.keras.optimizers.SGD(
    learning_rate=0.01), loss='mae')


for x, y in val_data_multi.take(1):
  print (multi_step_model.predict(x).shape)

print("hi")
# print(train_data_multi.shape[1])
multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS, steps_per_epoch=20,validation_data=val_data_multi, validation_steps=10,verbose=1)


plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')




if(len(features_considered)>1):
  #plot for MULTIVARIATE_MULTISTEP 
  print(x[0])
  for x, y in val_data_multi.take(3):
    multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
  plt.show()
else:
  #plot for UNIVARIATE_MULTISTEP
  print("univariateplot")
  if(STEP==1 & future_target ==1):
    for x, y in val_data_multi.take(3):
      plot = show_plot([x[0].numpy(), y[0].numpy(),
                        multi_step_model.predict(x)[0]], 0, 'Simple LSTM model')
      plot.show()
  else:
    for x, y in val_data_multi.take(3):
      multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
    plt.show()