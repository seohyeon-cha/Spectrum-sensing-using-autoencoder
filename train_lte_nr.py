
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataset.create_dataset import CustomDataset

import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

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
import itertools
import scipy
from scipy import io
from train.train_val_test import get_device
import csv
################################################################################################################
def return_loader(lte_file, nr_file):
    
    sns.set(style='whitegrid', context='notebook')
    # load lte data
    lte_data_numpy = scipy.io.loadmat(lte_file)['dataset_LTE']
    lte_data_numpy = lte_data_numpy.T
    lte_data_numpy = lte_data_numpy[np.nonzero(lte_data_numpy)] #(1xframe)

    # check LTE waveform
    _, ax = plt.subplots(figsize=(10, 8))
    ax.plot(np.real(lte_data_numpy), label='magnitude', color='b', linewidth=1)
    ax.set_title('LTE waveform', fontsize=16)
    plt.xlabel('frame')
    plt.ylabel('magnitude')
    plt.savefig('results/lte waveform.png')

    # Input feature shape
    lte_data = np.ndarray(shape=(len(lte_data_numpy),4))
    lte_data[:,0] = np.real(lte_data_numpy).T
    lte_data[:,1] = np.imag(lte_data_numpy).T
    lte_data[:,2] = np.angle(lte_data_numpy).T
    lte_data[:,3] = np.abs(lte_data_numpy).T

    lte_label = np.ndarray(shape=(len(lte_data_numpy),1))
    lte_label[:, 0] = 0


    # load nr data
    nr_data_numpy = scipy.io.loadmat(nr_file)['dataset_NR']
    nr_data_numpy = nr_data_numpy.T
    nr_data_numpy = nr_data_numpy[np.nonzero(nr_data_numpy)] #(1xframe)


    # check nr waveform
    _, ax = plt.subplots(figsize=(10, 8))
    ax.plot(np.real(nr_data_numpy), label='magnitude', color='b', linewidth=1)
    ax.set_title('5G NR waveform', fontsize=16)
    plt.xlabel('frame')
    plt.ylabel('magnitude')
    plt.savefig('results/nr waveform.png')


    # Input feature shape
    nr_data = np.ndarray(shape=(len(nr_data_numpy),4))
    nr_data[:,0] = np.real(nr_data_numpy).T
    nr_data[:,1] = np.imag(nr_data_numpy).T
    nr_data[:,2] = np.angle(nr_data_numpy).T
    nr_data[:,3] = np.abs(nr_data_numpy).T

    nr_label = np.ndarray(shape=(len(nr_data_numpy),1))
    nr_label[:, 0] = 1

    # Training data
    training_size = 350000
    train_data = lte_data[:training_size]
    train_label = lte_label[:training_size]

    # Test data
    test_size = 1000000
    test_data = lte_data[training_size:training_size+test_size]
    test_label = lte_label[training_size:training_size+test_size]

    test_data[150000:250000] = nr_data[:100000]
    test_data[250000:350000] = nr_data[200000:300000]
    test_data[350000:450000] = nr_data[400000:500000]
    test_data[700000:800000] = nr_data[600000:700000]
    test_data[800000:900000] = nr_data[700000:800000]
    test_data[900000:1000000] = nr_data[800000:900000]

    test_label[150000:250000] = nr_label[:100000]
    test_label[250000:350000] = nr_label[200000:300000]
    test_label[350000:450000] = nr_label[400000:500000]
    test_label[700000:800000] = nr_label[600000:700000]
    test_label[800000:900000] = nr_label[700000:800000]
    test_label[900000:1000000] = nr_label[800000:900000]
    


    # check test_data waveform
    _, ax = plt.subplots(figsize=(10, 8))
    ax.plot(test_data[:,0], label='magnitude', color='b', linewidth=1)
    ax.set_title('test data waveform', fontsize=16)
    plt.xlabel('frame')
    plt.ylabel('magnitude')
    plt.savefig('results/test data waveform.png')

    x_train_transformed = train_data
    x_test_transformed = test_data
    ######################################################################################################################
    # Transforms

    pipeline=Pipeline([('normalizer',Normalizer()), ('scaler',MinMaxScaler())])
    # pipeline=Pipeline([('scaler',MinMaxScaler((-1,1)))])
    
    x_train_transformed=pipeline.fit_transform(train_data)
    x_test_transformed = pipeline.transform(test_data)
    
    # check transformed data
    _, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x_train_transformed[:,0], label='magnitude', color='b', linewidth=1)
    ax.set_title('train data waveform', fontsize=16)
    plt.xlabel('frame')
    plt.ylabel('magnitude')
    plt.savefig('results/transformed train data waveform.png')
    
    # check transformed data
    _, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x_test_transformed[:,0], label='magnitude', color='b', linewidth=1)
    ax.set_title('test data waveform', fontsize=16)
    plt.xlabel('frame')
    plt.ylabel('magnitude')
    plt.savefig('results/transformed test data waveform.png')

    # plotting the relation between the 1st 3 features after transformation
    column_names=['real','imag','angle','abs']

    # one way of doing it
    # g=sns.PairGrid(pd.DataFrame(x_train_transformed[:600,:4]))

    # plt.subplots_adjust(top=0.9)
    # g.fig.suptitle('After:')
    # g.map_diag(sns.kdeplot)
    # g.map_offdiag(sns.kdeplot)
    # g.figure.savefig("results/transformation_output.png")


    #########################################################################################################
    # Training dataset
    random_seed = 4

    train_dataset = CustomDataset(x_train_transformed, train_label)
    train_dataset,val_dataset = train_test_split(train_dataset,test_size=0.2,random_state=random_seed)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    # train_features, train_labels = next(iter(train_dataloader))

    # validation dataset
    valid_dataloader = DataLoader(val_dataset, batch_size = 64, shuffle=False)

    # Test dataset 
    test_dataset = CustomDataset(x_test_transformed, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return (train_dataloader, valid_dataloader, test_dataloader, x_train_transformed, x_test_transformed, train_label, test_label)

from model.autoencoder import deepAE
from train.train_val_test import train, test

def train_sample():
    
    lte_file = 'dataset/data_lte.mat'
    nr_file = 'dataset/data_nr.mat'
    
    (train_dataloader, valid_dataloader, test_dataloader, _, _, _, _) = return_loader(lte_file, nr_file)
    

    device = get_device()
    network = deepAE().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    num_epochs = 15
        
    train_loss = train(network, train_dataloader, valid_dataloader, num_epochs, optimizer, criterion)
    
    return train_loss

def test():
    
    lte_file = 'dataset/data_lte.mat'
    nr_file = 'dataset/data_nr.mat'
    
    x_test_transformed = return_loader(lte_file, nr_file)[4]
    test_dataloader = return_loader(lte_file, nr_file)[2]
    criterion = nn.MSELoss()

    device = get_device()
    network = deepAE().to(device)
    path = 'model_save/pretrain_model2.pth'
    network.load_state_dict(torch.load(path))

    reconstructed_data, latent_vector = test(network, test_dataloader, criterion)
    recon_numpy = reconstructed_data.to('cpu').numpy()
    
    # Write on csv file 
    f = open('result_data/reconstruct', 'w')
    f.truncate()
    writer = csv.writer(f)
    writer.writerows(recon_numpy)
    f.close()

    # read from csv file 
    f = open('result_data/reconstruct.csv')
    reader = csv.reader(f)
    rows = []
    for i, row in enumerate(reader):
        rows.append(float(row[0]))
    f.close()

    # Plot reconstructed data
    _, ax = plt.subplots(figsize=(10, 8))
    ax.plot(rows, label='magnitude', color='b', linewidth=1)
    ax.set_title('reconstructed waveform', fontsize=16)
    plt.xlabel('frame')
    plt.ylabel('magnitude')
    plt.savefig('results/reconstructed_tr.png')
    print('Finish')
    mse = np.mean(np.power((recon_numpy-x_test_transformed),2),axis=1)
   
    return mse, latent_vector

