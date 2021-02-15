# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:47:23 2021

@author: David
"""
import os, warnings, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random 
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import STL
from BorutaShap import BorutaShap
from catboost import CatBoostRegressor


# Set seeds to make the results more reproducible.
def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    #tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 123
seed_everything(seed)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)


#import dataset
d_parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
df = pd.read_csv('D:/jd/clases/UDES/articulo IA y AQ/archivos_trabajo/bogota/QA_respiratatory_attentions.csv')
fecha = pd.date_range(start='2009-01-01', end='2019-12-31', freq ='1D')
df['Fecha'] = fecha
df.set_index('Fecha', inplace=True)
df.head()

##############################################borutashape
####borutashap for accumm of aerosols
#STL
df2 = df.iloc[:,18:89]
#df2 = df2.dropna()
df2.head()

#microg/m3
for i in range(0,71):
    x = df2.iloc[:,i]*1000000
    df2.iloc[:,i] = x
 
df2.head()

df2['Attentions'] = df['Attentions']
df2.head()
df2.shape

#trends
df2 = df2.dropna()

for i in range(0,72):
    x = df2.iloc[:,i]
    stl = STL(x, robust=True)
    result = stl.fit()
    seasonal, trend, resid = result.seasonal, result.trend, result.resid
    df2.iloc[:,i] = trend 
    
df2.head()    
    
X = df2.drop(['Attentions'], axis=1)
y = df2[['Attentions']]
X = pd.DataFrame(X)
y = pd.DataFrame(y)
Y = y.to_numpy()


model = CatBoostRegressor()
Feature_Selector = BorutaShap(model=model,
                              importance_measure='shap',
                              classification=False)

Feature_Selector.fit(X=X, y=Y, n_trials=100, sample=False, random_state = 123,
                normalize=True)

# Returns Boxplot of features
Feature_Selector.plot(which_features='all')

# Returns a subset of the original data with the selected features
subset = Feature_Selector.Subset()
subset  
subset_acummAES = subset #subset of acumulated values

#correlation subset of important variables
# Compute the correlation matrix
corr = subset_acummAES.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmin=-1, vmax=1)

#remove highly correlated ones
# Create correlation matrix
corr_matrix = subset_acummAES.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]

# Drop features 
subset_acummAES.drop(to_drop, axis=1, inplace=True)
subset_acummAES.shape
subset_acummAES



#####borutashape for acumulated of rainfall
df3 = df.iloc[:,90:97]
df3.head()
df3['Attentions'] = df['Attentions']
df3.head()
df3.shape

#trends
df3 = df3.dropna()

for i in range(0,8):
    x = df3.iloc[:,i]
    stl = STL(x, robust=True)
    result = stl.fit()
    seasonal, trend, resid = result.seasonal, result.trend, result.resid
    df3.iloc[:,i] = trend 
    
df3.head()
   
X = df3.drop(['Attentions'], axis=1)
y = df3[['Attentions']]
X = pd.DataFrame(X)
y = pd.DataFrame(y)
Y = y.to_numpy()


model = CatBoostRegressor()

# no model selected default is Random Forest, if classification is False it is a Regression problem
Feature_Selector = BorutaShap(model=model,
                              importance_measure='shap',
                              classification=False)

Feature_Selector.fit(X=X, y=Y, n_trials=100, sample=False, random_state = 123,
                normalize=True)

# Returns Boxplot of features
Feature_Selector.plot(which_features='all')

# Returns a subset of the original data with the selected features
subset = Feature_Selector.Subset()
subset  
    
subset_acummPreci = subset #subset de acumulados de aerosoles

#correlation subset of important variables
# Compute the correlation matrix
corr = subset_acummPreci.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmin=-1, vmax=1)

#remove highly correlated ones
# Create correlation matrix
corr_matrix = subset_acummPreci.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]

# Drop features 
subset_acummPreci.drop(to_drop, axis=1, inplace=True)
subset_acummPreci.shape
subset_acummPreci


#####borutashape for lags of aerosols
df4 = df.iloc[:,106:177]
df4.head()

#micrograms/m3
for i in range(0,71):
    x = df4.iloc[:,i]*1000000
    df4.iloc[:,i] = x
 
df4.head()
df4['Attentions'] = df['Attentions']
df4.head()
df4.shape

#trend
df4 = df4.dropna()

for i in range(0,72):
    x = df4.iloc[:,i]
    stl = STL(x, robust=True)
    result = stl.fit()
    seasonal, trend, resid = result.seasonal, result.trend, result.resid
    df4.iloc[:,i] = trend 
    
df4.head()    
    
    
X = df4.drop(['Attentions'], axis=1)
y = df4[['Attentions']]
X = pd.DataFrame(X)
y = pd.DataFrame(y)
Y = y.to_numpy()


model = CatBoostRegressor()

Feature_Selector = BorutaShap(model=model,
                              importance_measure='shap',
                              classification=False)

Feature_Selector.fit(X=X, y=Y, n_trials=100, sample=False, random_state = 123,
                normalize=True)

# Returns Boxplot of features
Feature_Selector.plot(which_features='all')

# Returns a subset of the original data with the selected features
subset = Feature_Selector.Subset()
subset  
    
subset_lagAES = subset #subset de acumulados de aerosoles
#important: 

#correlation subset of important variables
# Compute the correlation matrix
corr = subset_lagAES.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmin=-1, vmax=1)

#remove highly correlated ones
# Create correlation matrix
corr_matrix = subset_lagAES.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]

# Drop features 
subset_lagAES.drop(to_drop, axis=1, inplace=True)
subset_lagAES.shape
subset_lagAES


#####borutashap for lags of rainfall and temperature
df5 = df.iloc[:,178:193]
df5.head()

df5['Attentions'] = df['Attentions']
df5.head()
df5.shape

#trends
df5 = df5.dropna()

for i in range(0,16):
    x = df5.iloc[:,i]
    stl = STL(x, robust=True)
    result = stl.fit()
    seasonal, trend, resid = result.seasonal, result.trend, result.resid
    df5.iloc[:,i] = trend 
    
df5.head()

   
X = df5.drop(['Attentions'], axis=1)
y = df5[['Attentions']]
X = pd.DataFrame(X)
y = pd.DataFrame(y)
Y = y.to_numpy()


model = CatBoostRegressor()

# no model selected default is Random Forest, if classification is False it is a Regression problem
Feature_Selector = BorutaShap(model=model,
                              importance_measure='shap',
                              classification=False)

Feature_Selector.fit(X=X, y=Y, n_trials=100, sample=False, random_state = 123,
                normalize=True)

# Returns Boxplot of features
Feature_Selector.plot(which_features='all')

# Returns a subset of the original data with the selected features
subset = Feature_Selector.Subset()
subset  
    
subset_lagPreciTemp = subset #subset de acumulados de aerosoles

#correlation subset of important variables
# Compute the correlation matrix
corr = subset_lagPreciTemp.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmin=-1, vmax=1)

#remove highly correlated ones
# Create correlation matrix
corr_matrix = subset_lagPreciTemp.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]

# Drop features 
subset_lagPreciTemp.drop(to_drop, axis=1, inplace=True)
subset_lagPreciTemp.shape
subset_lagPreciTemp


#final_data
df_final = pd.concat([subset_acummAES, subset_acummPreci, subset_lagAES, subset_lagPreciTemp, df2['Attentions']], axis=1)
df_final = df_final.dropna()
df_final.shape

#remove highly correlated ones
# Create correlation matrix
corr_matrix = df_final.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.8
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]

# Drop features 
df_final.drop(to_drop, axis=1, inplace=True)
df_final.shape
df_final

#plots
sns.pairplot(df_final)

#save final df
df_final.to_csv('D:/jd/clases/UDES/articulo IA y AQ/archivos_trabajo/bogota/df_final_new.csv')
