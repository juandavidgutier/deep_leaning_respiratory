# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:48:44 2021

@author: David
"""
import os, warnings, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_score
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, \
    DateOffset, MO, next_monday, next_monday_or_tuesday, GoodFriday, EasterMonday
from pandas.tseries.holiday import *
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score


# Set seeds to make the results more reproducible.
def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 123
seed_everything(seed)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)


#important variables
d_parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
df = pd.read_csv('D:/jd/clases/UDES/articulo IA y AQ/archivos_trabajo/bogota/github/df_final.csv', parse_dates=['Fecha'], date_parser=d_parser)
df.set_index('Fecha', inplace=True)
df

#plot all variables vs. Attentions
for col in df.columns[1:]:
    df.plot.scatter(x=col, y='Attentions')
    
#all figure vs time
df.columns

plt.rcParams['figure.figsize'] = (15, 8)   # Increases the Plot Size
df['DMSSMASS_accum2d'].plot(grid = True)

plt.rcParams['figure.figsize'] = (15, 8)   # Increases the Plot Size
df['SO2SMASS_accum3d'].plot(grid = True)

plt.rcParams['figure.figsize'] = (15, 8)   # Increases the Plot Size
df['DUSMASS_accum7d'].plot(grid = True)

plt.rcParams['figure.figsize'] = (15, 8)   # Increases the Plot Size
df['SSSMASS_accum4d'].plot(grid = True)

plt.rcParams['figure.figsize'] = (15, 8)   # Increases the Plot Size
df['SO4SMASS_accum4d'].plot(grid = True)

plt.rcParams['figure.figsize'] = (15, 8)   # Increases the Plot Size
df['OCSMASS_accum3d'].plot(grid = True)

plt.rcParams['figure.figsize'] = (15, 8)   # Increases the Plot Size
df['TPRECMAX_accum3d'].plot(grid = True)

plt.rcParams['figure.figsize'] = (15, 8)   # Increases the Plot Size
df['T2MMEAN_lag0d'].plot(grid = True)

#obtain the month of attention
df['month'] = df.index.month

#workday
class CoBusinessCalendar(AbstractHolidayCalendar):
  rules = [
        Holiday('Año nuevo', month=1, day=1),
        Holiday('Reyes', month=1, day=6),
        Holiday('San José', month=3, day=19, offset=DateOffset(weekday=MO(1))),
        Holiday('Jueves Santo', month=1, day=1, offset=[Easter(), Day(-3)]),
        Holiday('Viernes Santo', month=1, day=1, offset=[Easter(), Day(-2)]),
        Holiday('Día del trabajo', month=5, day=1), 
        Holiday('Ascensión', month=5, day=21, offset=DateOffset(weekday=MO(1))),
        Holiday('Corpus Christi', month=6, day=11, offset=DateOffset(weekday=MO(1))),
        Holiday('Sagrado Corazón', month=6, day=19, offset=DateOffset(weekday=MO(1))),
        Holiday('San Pedro y San Pablo', month=6, day=25, offset=DateOffset(weekday=MO(1))),
        Holiday('20 de julio', month=7, day=20),
        Holiday('7 de agosto', month=8, day=7),
        Holiday('día de la raza', month=10, day=12, offset=DateOffset(weekday=MO(1))),
        Holiday('los santos', month=11, day=1, offset=DateOffset(weekday=MO(1))),         
        Holiday('independencia Cartagena', month=11, day=11, offset=DateOffset(weekday=MO(1))),           
        Holiday('velitas', month=12, day=8),              
        Holiday('Navidad', month=12, day=25)
     ]

co_BD = CustomBusinessDay(calendar=CoBusinessCalendar())
s = pd.date_range('2009-01-01', end='2019-12-31', freq=co_BD)
workdays = pd.DataFrame(s, columns=['Date'])

totaldays = pd.date_range(start='2009-01-01', end='2019-12-31', freq ='1D')
totaldays = pd.DataFrame(totaldays)
totaldays['Date'] = totaldays.iloc[:,0]
totaldays = totaldays[['Date']]

noworkdays = pd.concat([totaldays,workdays]).drop_duplicates(keep=False)

noworkdays['workday'] = 0
noworkdays.shape

workdays['workday'] = 1
workdays.shape    

workday = pd.concat([noworkdays, workdays])
workday.set_index('Date', inplace=True)
workday.sort_index()


df = pd.concat([df, workday], axis=1)
df = df.dropna()

#exclude these variable to best performance
df = df.drop(['SSSMASS_accum4d', 'SO2SMASS_accum3d'], axis=1)
df
list(df.columns.values) 
 
X = df.drop(['Attentions'], axis=1)
y = df[['Attentions']]

X = pd.DataFrame(X)
y = pd.DataFrame(y)
y_col='Attentions'


#fiugre with sets
test_set = df.iloc[3610:4010]
#validation_set = df.iloc[3210:3610]
train_set = df.iloc[0:3610]

test_set = test_set.Attentions
#validation_set = validation_set.Attentions
train_set = train_set.Attentions

plt.figure(figsize=(20, 5))
plt.plot(train_set)
#plt.plot(validation_set, color='g')
plt.plot(test_set, color='g')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('Attentions', fontsize=18)
plt.show()


test_size = int(len(df) * 0.1) # the test data will be 10% (0.1) of the entire data
train = df.iloc[:-test_size,:].copy() 

x_train = train.drop(['Attentions'], axis=1)
y_train = train[['Attentions']]

# the copy() here is important, it will prevent us from getting: SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame. Try using .loc[row_index,col_indexer] = value instead
test = df.iloc[-test_size:,:].copy()

x_test = test.drop(['Attentions'], axis=1)
y_test = test[['Attentions']]

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

training_set_shape = x_train.shape
print(training_set_shape)

Xscaler = MinMaxScaler(feature_range=(0, 1)) # scale so that all the X data will range from 0 to 1
Xscaler.fit(x_train)
scaled_x_train = Xscaler.transform(x_train)
print(x_train.shape)
Yscaler = MinMaxScaler(feature_range=(0, 1))
Yscaler.fit(y_train)
scaled_y_train = Yscaler.transform(y_train)
print(scaled_y_train.shape)
scaled_y_train = scaled_y_train.reshape(-1) # remove the second dimention from y so the shape changes from (n,1) to (n,)
print(scaled_y_train.shape)

scaled_y_train = np.insert(scaled_y_train, 0, 0)
scaled_y_train = np.delete(scaled_y_train, -1)

b_size = 1000 # Number of timeseries samples in each batch
n_input = 1  #how many samples/rows/timesteps to look in the past in order to forecast the next sample
n_features= x_train.shape[1] # how many predictors/Xs/features we have to predict y

#generator = TimeseriesGenerator(scaled_x_train, scaled_y_train, length=n_input, batch_size=scaled_x_train.shape[0])
#X, y = generator[0]  

generator = TimeseriesGenerator(scaled_x_train, scaled_y_train, length=n_input, batch_size=b_size)
for i in range(len(generator)):
    X, y = generator[i]


# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)


#architecture of the network
model = Sequential()

#1st LSTM and Dropout
model.add(LSTM(8,
               activation='selu', input_shape=(n_input, n_features), return_sequences=True))
model.add(Dropout(0.1))

#2nd LSTM and Dropout
model.add(LSTM(5, kernel_regularizer=l2(0.01),
               activation='selu', return_sequences=True))
model.add(Dropout(0.1))

#3rd LSTM and Dropout
model.add(LSTM(3, kernel_regularizer=l1(0.000087),
               activation='selu', return_sequences=True))
model.add(Dropout(0.1))

#4th LSTM and Dropout
model.add(LSTM(2))
model.add(Dropout(0.1))

#5th layer (output layer)
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse')     
 
#mape metric
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#variables to save the result of each instance
val_predictions = np.arange(0,400,1)
val_predictions = pd.DataFrame(val_predictions)
val_metrics = np.arange(0,8,1)
val_metrics = pd.DataFrame(val_metrics)


for i in range(1,21):
    random_state = 123
    model.fit(X, y, epochs=5000, validation_split=0.1, callbacks=[es], use_multiprocessing=True, shuffle=False)
    scaled_x_test = Xscaler.transform(x_test)
    test_generator = TimeseriesGenerator(scaled_x_test, np.zeros(len(x_test)), length=n_input, batch_size=x_test.shape[0])
    y_pred_scaled = model.predict(test_generator)
    y_pred = Yscaler.inverse_transform(y_pred_scaled)
    results = pd.DataFrame({'y_true':test[y_col].values[n_input:],'y_pred':y_pred.ravel()})
    fecha_y = pd.date_range(start='2018-11-27', end='2019-12-31', freq ='1D')
    fecha_y.shape
    results['Fecha'] = fecha_y 
    results.set_index('Fecha', inplace=True)
    results.y_pred = pd.DataFrame(results.y_pred)
    val_predictions['instance' + str(i)] = y_pred
    ev_test = explained_variance_score(results.y_true, results.y_pred)
    me_test = max_error(results.y_true, results.y_pred)
    mae_test = mean_absolute_error(results.y_true, results.y_pred)
    mse_test = mean_squared_error(results.y_true, results.y_pred)
    rmse_test = mean_squared_error(results.y_true, results.y_pred, squared=False)
    mdae_test = median_absolute_error(results.y_true, results.y_pred)
    r2_test= r2_score(results.y_true, results.y_pred)
    mape_test = mean_absolute_percentage_error(results.y_true, results.y_pred)
    metrics = [ev_test, me_test, mae_test, mse_test, rmse_test, mdae_test, r2_test, mape_test]
    val_metrics['instance' + str(i)] = metrics
    
    #test_generator = TimeseriesGenerator(scaled_x_test, np.zeros(len(x_test)), length=n_input, batch_size=x_test.shape[0])
    #y_pred_scaled = model.predict(test_generator[0][0])

val_predictions
val_predi = val_predictions.iloc[:,1:21]
val_predi['mean'] = val_predi.mean(axis=1)
val_predi


fecha_y = pd.date_range(start='2018-11-27', end='2019-12-31', freq ='1D')

val_predi['Fecha'] = fecha_y 
val_predi.set_index('Fecha', inplace=True)

#plot of the mean of 20 instances
plt.figure(figsize=(20, 5))
plt.plot(df.index, df['Attentions'])
plt.plot(val_predi.index, val_predi['mean'], color='r')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('Attentions', fontsize=18)
plt.show()

#histogram
y_pred = val_predi['mean'].to_numpy()
y_true = results.y_true.to_numpy()

plt.figure(figsize=(15, 4))
plt.hist(y_pred, bins=100, color='orange', alpha=0.5, label='test pred')
plt.hist(y_true, bins=100, color='green', alpha=0.5, label='test true')
plt.legend()
plt.xlabel('Daily attentions', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.show()

val_metrics
val_metrics = pd.DataFrame(val_metrics)
val_metrics = val_metrics.iloc[:,1:21]
val_metrics

val_metrics = val_metrics.values
val_metri = np.transpose(val_metrics)
val_metri = pd.DataFrame(val_metri)
val_metri.columns = ['ev_test', 'me_test', 'mae_test', 'mse_test', 'rmse_test', 'mdae_test', 'r2_test', 'mape_test']

#figs
fig1, ax1 = plt.subplots()
ax1.set_title('a)')# explained_variance_score
ax1.boxplot(val_metri.ev_test)

fig1, ax1 = plt.subplots()
ax1.set_title('b)')#max_error
ax1.boxplot(val_metri.me_test)

fig1, ax1 = plt.subplots()
ax1.set_title('c)') #mean_absolute_error
ax1.boxplot(val_metri.mae_test)

fig1, ax1 = plt.subplots()
ax1.set_title('d)')# mean_squared_error
ax1.boxplot(val_metri.mse_test)

fig1, ax1 = plt.subplots()
ax1.set_title('e)') #root_mean_squared_error
ax1.boxplot(val_metri.rmse_test)

fig1, ax1 = plt.subplots()
ax1.set_title('f)') #median_absolute_error
ax1.boxplot(val_metri.mdae_test)

fig1, ax1 = plt.subplots()
ax1.set_title('g)') # r2_score
ax1.boxplot(val_metri.r2_test)

fig1, ax1 = plt.subplots()
ax1.set_title('h)')# mean absolute percentage error 
ax1.boxplot(val_metri.mape_test)

print(val_metri.mean(axis = 0))
val_metri.shape

