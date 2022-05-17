import torch
import pandas as pd
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import itertools
import scipy
from scipy import io


# setting seaborn style
sns.set(style='whitegrid', context='notebook')

random_seed=4

# load lte data
lte_data_numpy = scipy.io.loadmat('data_lte.mat')['dataset_LTE']
lte_data_numpy = lte_data_numpy.flatten()
lte_data_numpy = lte_data_numpy.T


lte_data=pd.DataFrame({'real':np.real(lte_data_numpy)[:],'img':np.imag(lte_data_numpy)[:],
                        'phase':np.angle(lte_data_numpy)[:], 'amp':np.abs(lte_data_numpy)[:],
                       })

lte_data=lte_data.loc[~(lte_data==0).any(axis=1)]
lte_data['label']=0
#lte_data=lte_data.sample(frac=1)
lte_data.reset_index(drop=True, inplace=True)
lte_data.head()

# load nr data
nr_data_numpy = scipy.io.loadmat('data_nr.mat')['dataset_NR']
nr_data_numpy = nr_data_numpy.flatten()
nr_data_numpy = nr_data_numpy.T


nr_data=pd.DataFrame({'real':np.real(nr_data_numpy)[:],'img':np.imag(nr_data_numpy)[:],
                        'phase':np.angle(nr_data_numpy)[:], 'amp':np.abs(nr_data_numpy)[:],
                       })

nr_data=nr_data.loc[~(nr_data==0).any(axis=1)]
nr_data['label']=0
#lte_data=lte_data.sample(frac=1)
nr_data.reset_index(drop=True, inplace=True)
nr_data.head()

# Create train and test data
training_size = 350000

# train data. dropping labels on training data since we don't need it for training
x_train = lte_data[:training_size].drop(['label'], axis=1)

# test data
x_test=lte_data[training_size:]

# # inserting Wi-Fi data in between
x_test = x_test.copy()
nr_data = nr_data.copy()

x_test.iloc[:10000,:] = nr_data.iloc[:10000,:].values
x_test.iloc[25000:35000,:] = nr_data.iloc[20000:30000,:].values
x_test.iloc[35000:45000,:] = nr_data.iloc[40000:50000,:].values
x_test.iloc[70000:80000,:] = nr_data.iloc[60000:70000,:].values
x_test.iloc[80000:90000,:] = nr_data.iloc[70000:80000,:].values
x_test.iloc[90000:100000,:] = nr_data.iloc[80000:90000,:].values

print("Length of train data:",len(x_train))
print("Length of test data:",len(x_test))

# creating training and validation dataset
x_train,x_validate = train_test_split(x_train,test_size=0.2,random_state=random_seed)

# separating labels from test dataset for plotting later
x_test,labels = x_test.drop('label',axis=1).values,x_test.label.values




# creating a pipeline for normalizing

pipeline=Pipeline([('normalizer',Normalizer()), ('scaler',MinMaxScaler())])
x_train_transformed=pipeline.fit_transform(x_train)
x_validate_transformed=pipeline.fit_transform(x_validate)


# plotting the relation between the 1st 3 features after transformation

# storing the column names because the transformed data is a numpy array and does not contain column name. But the  \
# iloc function needs a dataframe and so we are converting the numpy array to dataframe which needs the column name.
column_names=list(x_train.columns)

# one way of doing it
g=sns.PairGrid(pd.DataFrame(x_train_transformed, columns=column_names).iloc[:,:5].sample(600))   

# another way of doing the above without converting into dataframe. Note: .sample(600) is returning a random 600 examples not the first 600 examples
#g=sns.PairGrid(pd.DataFrame(x_train_transformed[:600,:3]))
plt.subplots_adjust(top=0.9)
g.fig.suptitle('After:')
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot)
g.figure.savefig("output.png")

#shape of training and testing data

print(x_train_transformed.shape)
print(x_validate_transformed.shape)

