import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import warnings
import seaborn as sns
import pickle

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten,Reshape
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tensorflow_model_optimization as tfmot

import tempfile
import n_cut.n_cut as nc
import n_cut.MNIST_helper as mh
from collections import OrderedDict
from scipy import stats

# count the number of nodes in a model including inputs
def count_nodes(model, start=1):
    # num input nodes
    count = model.layers[start].get_weights()[0].shape[0]
    # num nodes in each layer from 2nd dimension of weight matrix
    for layer in model.layers[start:]:
        num =  layer.get_weights()[0].shape[1]
        count = count + num 
    return count

# fill adjacency matrix with weights from a single layer
def fill_from_layer(layer, A_matrix, offset, identity=False, shuffle_weights='No'):
    A = A_matrix.copy()
    W = layer.get_weights()[0]
    if shuffle_weights == 'Yes':
        rng = np.random.default_rng()
        rng.shuffle(W, axis=0)
        rng.shuffle(W, axis=1)
    num_nodes_in = W.shape[0]
    num_nodes_out = W.shape[1]
    for i in range(num_nodes_in):
        for j in range(num_nodes_out):
            if identity == False:
                A[i + offset, j + num_nodes_in + offset] = W[i,j]
                A[ j + num_nodes_in + offset, i + offset] = W[i,j]
            else:
                A[i + offset, j + num_nodes_in + offset] = 1
                A[ j + num_nodes_in + offset, i + offset] = 1
    return A, num_nodes_in  

# Make and fill adjacency matrix for a given model
def make_adjacency_matrix(model, start=1, identity=False, shuffle_weights='No'):
    num_nodes = count_nodes(model, start=start)
    A = np.zeros([num_nodes, num_nodes])
    offset = 0
    for layer in model.layers[start:]:
        A, off = fill_from_layer(layer, A, offset, identity=identity, shuffle_weights=shuffle_weights)
        offset = offset + off
    return A

from sklearn.cluster import SpectralClustering
def get_clusters(model, num_clusters, random_state=None, start=1, identity=False, shuffle_weights='No'):
    A = make_adjacency_matrix(model, shuffle_weights=shuffle_weights, identity=identity, start=start)
    A = np.abs(A)
    sc = SpectralClustering(num_clusters, affinity='precomputed', n_init=100,
                        assign_labels='kmeans', eigen_solver = 'arpack', random_state=random_state)
    sc.fit_predict(A)
    return sc.labels_, A 

def calculate_volume(group, labels, degrees):
    volume = []
    for i, label in enumerate(labels):
        if label == group:
            volume.append(degrees[i])
    return np.array(volume).sum()

def make_group_to_volumes_dict(A,labels):
    degrees = A.sum(axis=1)
    volumes = {}
    for group in set(labels):
        volumes[group] = calculate_volume(group, labels, degrees)
    return volumes

def make_group_to_members_dict(labels):
    groups = set(labels)
    dict = {}
    for group in groups:
        members = []
        for i, label in enumerate(labels):
            if label == group:
                members.append(i)
        dict[group] = members
    return dict 

def calc_W_between_two_clusters(group_label0, group_label1, group_members_dict, A):
    W = []
    group0 = group_members_dict[group_label0]
    group1 = group_members_dict[group_label1]
    for m0 in group0:
        for m1 in group1:
            if A[m0, m1] !=0:
                W.append(A[m0, m1])
    return np.array(W).sum()

def calculate_W_one_to_all_clusters(group0, group_to_members_dict, group_to_volumes_dict, A):
    Ws = []
    for key in group_to_members_dict.keys():
        if key != group0:
            Ws.append(calc_W_between_two_clusters(group0, key, group_to_members_dict, A))
    W = np.array(Ws).sum()
    W = W/group_to_volumes_dict[group0]
    return W

def calculate_n_cut(labels, A):
    group_to_members = make_group_to_members_dict(labels)
    group_to_volumes = make_group_to_volumes_dict(A,labels)
    W_over_Vs = []
    for group in group_to_members.keys():
        W_over_Vs.append(calculate_W_one_to_all_clusters(group, group_to_members, group_to_volumes, A))
    return np.array(W_over_Vs).sum()


def make_train_simultaneous_model(x_train,
                                  y_train,
                                  x_test,
                                  y_test,
                                  series,
                                  num_hidden_layers=4,
                                  num_hidden_nodes=256,
                                  num_output_nodes=4,
                                  pruning=None,
                                  epochs=20,
                                  verbose=2,
                                  summary=True):
    
    # Make data with just numerals in series
    x_tr_data, y_tr_data = mh.make_repeating_series(x_train, y_train, series, 10000)
    x_te_data, y_te_data = mh.make_repeating_series(x_test, y_test, series, 10000)
    
    # Make one-hot encoded y vectors for series just made
    y_tr_data_one_hot = mh.make_one_hot_y_for_series(y_tr_data, series)
    y_te_data_one_hot = mh.make_one_hot_y_for_series(y_te_data, series)
    
    # Create a neural network model and show summary
    num_hidden_layers = num_hidden_layers
    num_hidden_nodes = num_hidden_nodes
    num_output_nodes = num_output_nodes 
    model = Sequential()
    model.add(Flatten(input_shape= (28,28)))
    for i in range(num_hidden_layers):
        model.add(Dense(num_hidden_nodes , activation='relu'))
    model.add(Dense(num_output_nodes , activation='softmax'))

    callbacks = None
    if pruning != None:
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        pruning_params = {'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=pruning, begin_step=0)}
        model = prune_low_magnitude(model, **pruning_params)

        logdir = tempfile.mkdtemp()
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
        ]

    if summary == True:
        model.summary()
    

    # Set loss and optimizer and compile model
    loss = tf.keras.losses.categorical_crossentropy
    optimizer = Adam(lr=0.001)
    metrics = ['accuracy'] 
    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    
    # Fit model
    history = model.fit(x_tr_data, y_tr_data_one_hot, 
                                      epochs=epochs, 
                                      batch_size=32, 
                                      verbose=verbose,
                                      validation_data=(x_te_data, y_te_data_one_hot),
                                      callbacks=callbacks, 
                                      shuffle=True)
    
    # Plot loss and accuracy
    mh.plot_loss_and_accuracy(history)
    
    return model, history

def make_train_sequential_model(x_train,
                                  y_train,
                                  x_test,
                                  y_test,
                                  series,
                                  num_hidden_layers=4,
                                  num_hidden_nodes=256,
                                  num_output_nodes=4,
                                  pruning=None,
                                  epochs_per_numeral=5,
                                  num_cycles = 3, 
                                  verbose=2,
                                  summary=True):
    
    # Make data with just numerals in series
    x_tr_data, y_tr_data = mh.make_repeating_series(x_train, y_train, series, 10000)
    x_te_data, y_te_data = mh.make_repeating_series(x_test, y_test, series, 10000)
    
    # Make one-hot encoded y vectors for series just made
    y_tr_data_one_hot = mh.make_one_hot_y_for_series(y_tr_data, series)
    y_te_data_one_hot = mh.make_one_hot_y_for_series(y_te_data, series)
    
    # Create a neural network model and show summary
    num_hidden_layers = num_hidden_layers
    num_hidden_nodes = num_hidden_nodes
    num_output_nodes = num_output_nodes 
    model = Sequential()
    model.add(Flatten(input_shape= (28,28)))
    for i in range(num_hidden_layers):
        model.add(Dense(num_hidden_nodes , activation='relu'))
    model.add(Dense(1 , activation='sigmoid'))
    
    callbacks = None
    if pruning != None:
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        pruning_params = {'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=pruning, begin_step=0)}
        model = prune_low_magnitude(model, **pruning_params)

        logdir = tempfile.mkdtemp()
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
        ]

    if summary == True:
        model.summary()
        
    # Set loss and optimizer and compile model
    loss = tf.keras.losses.binary_crossentropy
    optimizer = Adam(lr=0.001)
    metrics = ['accuracy'] 
    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    
    for i in range(num_cycles):
        mutated_series = np.random.permutation(series)
        print('round' + str(i), mutated_series)
        for num in mutated_series:
            print(num)
            y_train_binary = mh.make_y_for_logistic(y_tr_data, num)
            y_test_binary = mh.make_y_for_logistic(y_te_data, num)
            history = model.fit(x_tr_data, y_train_binary, 
                                          epochs=epochs_per_numeral, 
                                          batch_size=32, 
                                          verbose=verbose,
                                          validation_data=(x_te_data, y_test_binary),
                                          callbacks=callbacks,
                                          shuffle=True)

            mh.plot_loss_and_accuracy(history)
        
    model_B = Sequential(model.layers[:-1])
    model_B.add(Dense(num_output_nodes, activation='softmax'))
    model_B.summary()
    
    for layer in model_B.layers[:-1]:
        layer.trainable=False
    
    # Set loss and optimizer and compile model
    loss = tf.keras.losses.categorical_crossentropy
    optimizer = Adam(lr=0.001)
    metrics = ['accuracy'] 
    model_B.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    
    history_B = model_B.fit(x_tr_data, y_tr_data_one_hot, 
                                      epochs=10, 
                                      batch_size=32, 
                                      verbose=2,
                                      validation_data=(x_te_data, y_te_data_one_hot), 
                                      shuffle=True)

    
    return model_B, history_B



from scipy import stats
def get_n_cuts_and_p_value(model, model_history, num_clusters, random_state=None, identity=False, start=1, num_shuffle_trials=100):
    shuffle_weights = 'No'
    labels, A = nc.get_clusters(model=model, num_clusters=num_clusters, 
                                random_state=random_state, identity=identity,
                                start=start, shuffle_weights=shuffle_weights)
    n_cut =  nc.calculate_n_cut(labels, A)
    
    shuffle_weights = 'Yes'
    list_random_n_cuts = []
    for i in range(num_shuffle_trials):
        labels_rand, A_rand = nc.get_clusters(model=model, num_clusters=num_clusters,
                                    random_state=random_state,
                                    identity=identity, start=start,
                                    shuffle_weights=shuffle_weights)
        n_cut_rand =  nc.calculate_n_cut(labels_rand, A_rand)
        list_random_n_cuts.append(n_cut_rand)
        
    mean_rand_n_cut = np.array(list_random_n_cuts).mean()
    std_rand_n_cut = np.array(list_random_n_cuts).std()
    ste_rand_n_cut = std_rand_n_cut/(num_shuffle_trials)**(1/2)
    percentile = stats.percentileofscore(np.array(list_random_n_cuts), n_cut)
    
    if percentile >= 97.5 or percentile <= 2.5:
        sig = 'Yes'
    else:
        sig = 'No'
    
    if percentile < 50:
        p = 2 * percentile * 0.01
        direction ='smaller'
    else:
        p = 2 * (100 - percentile) * 0.01
        direction ='larger'
    
    accuracy = model_history.history['val_accuracy'][-1]
    
    return model, n_cut,  mean_rand_n_cut , std_rand_n_cut, ste_rand_n_cut, percentile, p, sig,  direction, accuracy,


def process_df(df):
    dict = OrderedDict()
    
    dict['Series'] = df.index.str.slice(0,4)[0]
    dict['Train_type'] = df.index.str.slice(5,8)[0]
    
    num_nodes_per_hidden_layer = df.index.str.slice(-8,-5)[0]
    if num_nodes_per_hidden_layer == '_32':
         hn = 32
    elif num_nodes_per_hidden_layer == '_64':
         hn= 64
    elif num_nodes_per_hidden_layer == '128':
         hn = 128
    elif num_nodes_per_hidden_layer == '256':
         hn = 256
    
    if hn >=100:
        dict['H_layers'] = int(df.index.str.slice(-10,-9)[0])
    else:
        dict['H_layers'] = int(df.index.str.slice(-9,-8)[0])
    
    dict['H_nodes'] = hn
    
    prune = df.index.str.slice(15,17)[0]
    if prune == '0_':
        dict['prunning'] = 0
    elif prune == '04':
        dict['prunning'] = 0.4
    elif prune == '08':
        dict['prunning'] = 0.8
    
    dict['Mean_N_cut'] = mean_n_cut  = np.round(df.N_cut.mean(),3)
    dict['N_cut_std'] = std_n_cut = np.round(df.N_cut.std(),3)
    dict['N_cut_ste'] = ste_n_cut = np.round(df.N_cut.std()/10**(1/2),3)
    dict['N_cut_rand'] = mean_n_cut_random = np.round(df.Mean_n_cut_rand.mean(),3)
    dict['N_cut_rand_std'] = std_n_cut_random = np.round(df.Mean_n_cut_rand.std(),3)
    dict['N_cut_diff'] = np.round(mean_n_cut - mean_n_cut_random, 3)
    
    if dict['N_cut_diff'] < 0:
        dict['Direction'] = 'smaller'
    elif dict['N_cut_diff'] > 0:
        dict['Direction'] = 'larger'
    else:
        dict['Direction'] = 'no change'
        
    t, p = stats.ttest_rel(df.N_cut, df.Mean_n_cut_rand)
    dict['P_value'] = np.round(p, 3)
    if p <= 0.05:
        dict['Sig'] = 'Yes'
    else:
        dict['Sig'] = 'No'

    dict['Accuracy'] = df.Accuracy.mean()

    #dict['N_cut_diff_std'] = np.round(( (std_n_cut)**2 + (std_n_cut_random)**2 ) **(1/2),3)
    
    #smaller = df.Direction[df.Direction == 'smaller'].count()
    #larger = df.Direction[df.Direction == 'larger'].count()
    #dict['Frac_smaller'] = smaller/(larger + smaller)
    #dict['Frac_smaller'] = larger/(larger + smaller)

    #dict['Frac_sig_smaller'] = np.round(df[ (df.Significance == 'Yes') & (df.Direction == 'smaller')].shape[0]/smaller, 3)
    #dict['Frac_sig_larger'] = np.round(df[ (df.Significance == 'Yes') & (df.Direction == 'larger')].shape[0]/larger, 3) 
    
    return dict

def make_table(dfs_list):
    rows = []
    for i, df in enumerate(dfs_list):
        rows.append(nc.process_df(df))
    df_out = pd.DataFrame(rows)
    return df_out

def load_dfs_to_list(path, file_list):
    dfs = []
    for f in file_list:
        file = path + f
        with open(file, 'rb') as fp:
            dfs.append(pickle.load(fp))
    return dfs