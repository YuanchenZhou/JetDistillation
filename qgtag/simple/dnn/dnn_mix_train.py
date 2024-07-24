# standard library imports
from __future__ import absolute_import, division, print_function
import argparse

# Data I/O and numerical imports
#import h5py
import numpy as np

# ML imports
import tensorflow as tf
import keras
from keras.models import load_model
from keras.callbacks import EarlyStopping,ModelCheckpoint

tf.experimental.numpy.experimental_enable_numpy_behavior()

from tensorflow.data import Dataset
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# energyflow is not available by default
import energyflow as ef

from energyflow.archs.efn import PFN
from energyflow.archs.dnn import DNN
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, remap_pids, to_categorical

# Plotting imports
import matplotlib.pyplot as plt

import time

###########

class PredictionTimeHistory(tf.keras.callbacks.Callback):
    def on_predict_begin(self, logs=None):
        self.times = []
        self.total_time = 0
    def on_predict_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.perf_counter()
    def on_predict_batch_end(self, batch, logs=None):
        self.batch_end_time = time.perf_counter()
        batch_time = self.batch_end_time - self.batch_start_time
        self.times.append(batch_time)
        self.total_time += batch_time

def print_gpu_info():
    gpus = tf.config.list_physical_devices('GPU')
    name = []
    if not gpus:
        print("No GPUs found.")
    else:
        for gpu in gpus:
            print(f"Device: {gpu.device_type}, Name: {gpu.name}")
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                gpu_name = gpu_details.get('device_name', 'Unknown GPU Name')
                name.append(gpu_name)
                pci_bus_id = gpu_details.get('pci_bus_id', 'Unknown PCI Bus ID')
                print(f"Details - Name: {gpu_name}, PCI bus ID: {pci_bus_id}")
            except Exception as e:
                print(f"Could not retrieve details for GPU: {gpu.name}. Error: {str(e)}")
    return name

###########
# Main script
parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-nEpochs", dest='nEpochs', default=200, type=int, required=False,
                    help="How many epochs to train for?")
parser.add_argument("-batchSize", dest='batchSize', default=500, type=int, required=False,
                    help="How large should the batch size be?")
parser.add_argument("-doEarlyStopping", dest='doEarlyStopping', action='store_true', required=False,
                    help="Do early stopping?")
parser.add_argument("-patience", dest='patience', default=20, type=int, required=False,
                    help="How patient")
parser.add_argument("-usePIDs", dest='usePIDs', action='store_false', required=False,
                    help="Use PIDs? If True, this script will currently break!")
parser.add_argument("-nLayers",dest='nLayers', default=2, type=int, required=False,
                    help="How many layers for the dnn")
parser.add_argument("-layerSize",dest='layerSize', default=100, type=int,required=False,
                    help="How large should the layer dense size be for the simple and student model")
args = parser.parse_args()

if(args.nEpochs==0 and args.doEarlyStopping==False):
    raise Exception("You need to specify a number of epochs to train for, or to use early stopping!")
# nice : https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two
#if(args.nEpochs>0 and args.doEarlyStopping):
#    raise Exception("Either specify early stopping, **or** a number of epochs to train for!")
#print("pfn_example.py\tWelcome!")

################################################################################
# data controls, can go up to 2000000 for full Pythia dataset, 1500000 for full Herwig dataset
pythia_num = 2000000
herwig_num = 1500000

#train, val, test = 75000, 10000, 15000 # small
#train, val, test = 150000, 20000, 30000 # medium (2x small, ~0.1x complete)
train_ratio, val_ratio, test_ratio = 0.75, 0.125, 0.125
train_pythia, val_pythia, test_pythia = int(pythia_num*train_ratio), int(pythia_num*val_ratio), int(pythia_num*test_ratio) # complete
train_herwig, val_herwig, test_herwig = int(herwig_num*train_ratio), int(herwig_num*val_ratio), int(herwig_num*test_ratio)
use_pids = args.usePIDs

# network architecture parameters
# DNN
nLayers = args.nLayers
layerSize = args.layerSize
dense_sizes = (layerSize,)*nLayers

# network training parameters
num_epoch = args.nEpochs
if(args.doEarlyStopping):
    num_epoch = 500
batch_size = args.batchSize
patience = args.patience
################################################################################

# load Pythia training data
print('Loading the Pythia training dataset ...')
X_pythia, y_pythia = qg_jets.load(train_pythia + val_pythia + test_pythia, generator='pythia')
n_pythia_pad = 148 - X_pythia.shape[1]
X_pythia = np.lib.pad(X_pythia, ((0,0), (0,n_pythia_pad), (0,0)), mode='constant', constant_values=0)
print('Dataset loaded!')
# convert labels to categorical
Y_pythia = to_categorical(y_pythia, num_classes=2)
print('Loaded quark and gluon jets')
# preprocess by centering jets and normalizing pts
for x in X_pythia:
    mask = x[:,0] > 0
    yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
    x[mask,1:3] -= yphi_avg
    x[mask,0] /= x[:,0].sum()
# handle particle id channel
if use_pids:
    remap_pids(X_pythia, pid_i=3)
else:
    X_pythia = X_pythia[:,:,:3]
print('Finished preprocessing')
# do train/val/test split
(X_pythia_train, X_pythia_val, X_pythia_test,
 Y_pythia_train, Y_pythia_val, Y_pythia_test) = data_split(X_pythia, Y_pythia, val=val_pythia, test=test_pythia)
print('Done pythia train/val/test split')


# load Herwig training data
print('Loading the Herwig training dataset ...')
X_herwig, y_herwig = qg_jets.load(train_herwig + val_herwig + test_herwig, generator='herwig')
n_herwig_pad =  148 - X_herwig.shape[1]
X_herwig = np.lib.pad(X_herwig, ((0,0), (0,n_herwig_pad), (0,0)), mode='constant', constant_values=0)
print('Dataset loaded!')
# convert labels to categorical
Y_herwig = to_categorical(y_herwig, num_classes=2)
print('Loaded quark and gluon jets')
# preprocess by centering jets and normalizing pts
for x in X_herwig:
    mask = x[:,0] > 0
    yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
    x[mask,1:3] -= yphi_avg
    x[mask,0] /= x[:,0].sum()
# handle particle id channel
if use_pids:
    remap_pids(X_herwig, pid_i=3)
else:
    X_herwig = X_herwig[:,:,:3]
print('Finished preprocessing')
# do train/val/test split
(X_herwig_train, X_herwig_val, X_herwig_test,
 Y_herwig_train, Y_herwig_val, Y_herwig_test) = data_split(X_herwig, Y_herwig, val=val_herwig, test=test_herwig)
print('Done herwig train/val/test split')


X_mix = np.concatenate((X_pythia,X_herwig), axis=0)
X_mix_train = np.concatenate((X_pythia_train,X_herwig_train),axis=0)
X_mix_val = np.concatenate((X_pythia_val,X_herwig_val),axis=0)
X_mix_test = np.concatenate((X_pythia_test,X_herwig_test),axis=0)
Y_mix = np.concatenate((Y_pythia,Y_herwig),axis=0)
Y_mix_train = np.concatenate((Y_pythia_train,Y_herwig_train),axis=0)
Y_mix_val = np.concatenate((Y_pythia_val,Y_herwig_val),axis=0)
Y_mix_test = np.concatenate((Y_pythia_test,Y_herwig_test),axis=0)
print('Done Mixing Pythia and Herwig')


print('Pythia Shape:',X_pythia.shape)
print('Herwig Shape:',X_herwig.shape)
print('Mix Shape:', X_mix.shape)


############################################

# build architecture
# dense_sizes above
dnn_mix_simple  = DNN(input_dim=X_mix_train.shape[1]*X_mix_train.shape[2], dense_sizes=dense_sizes)

# train the simple mix model
if(args.doEarlyStopping):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    mc = ModelCheckpoint(filepath =f'/users/yzhou276/work/qgtag/simple/dnn/model/best_{dense_sizes}_dnn_mix.keras', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    dnn_mix_simple.fit(X_mix_train.reshape(-1,X_mix_train.shape[1]*X_mix_train.shape[2]), Y_mix_train,
                   epochs=num_epoch,
                   batch_size=batch_size,
                   validation_data=(X_mix_val.reshape(-1,X_mix_val.shape[1]*X_mix_val.shape[2]), Y_mix_val),
                   verbose=1,
                   callbacks=[es,mc])
else:
    dnn_mix_simple.fit(X_mix_train.reshape(-1,X_mix_train.shape[1]*X_mix_train.shape[2]), Y_mix_train,
                   epochs=num_epoch,
                   batch_size=batch_size,
                   validation_data=(X_mix_val.reshape(-1,X_mix_val.shape[1]*X_mix_val.shape[2]), Y_mix_val),
                   verbose=1)
    dnn_mix_simple.save(f'/users/yzhou276/work/qgtag/simple/dnn/model/best_{dense_sizes}_dnn_mix.keras')

############################################

gpu = print_gpu_info()

# get Mix simple predictions on pythia test data and ROC curve
preds_mix_simple_pythia = dnn_mix_simple.predict(X_pythia_test.reshape(-1,X_pythia_val.shape[1]*X_pythia_val.shape[2]), batch_size=1000)
dnn_fp_mix_simple_pythia, dnn_tp_mix_simple_pythia, threshs_mix_simple_pythia = roc_curve(Y_pythia_test[:,1], preds_mix_simple_pythia[:,1])
auc_mix_simple_pythia  = roc_auc_score(Y_pythia_test[:,1], preds_mix_simple_pythia[:,1])
# Get Prediction Time
mix_simple_pred_time_on_pythia = []
print('Mix/P Simple Prediction Time:')
for i in range(6):
    pred_time_callback = PredictionTimeHistory()
    predictions = dnn_mix_simple.predict(X_pythia_test.reshape(-1,X_pythia_val.shape[1]*X_pythia_val.shape[2]), batch_size=1000, verbose=1, callbacks=[pred_time_callback])
    #print(pred_time_callback.times)
    if i>0:
        mix_simple_pred_time_on_pythia.append(pred_time_callback.times)
    i=i+1
mix_simple_pred_time_on_pythia = np.array(mix_simple_pred_time_on_pythia)
MixP_simple_avg_pred_time = np.mean(mix_simple_pred_time_on_pythia)
print()
print('Mix/Pythia Simple DNN AUC:', auc_mix_simple_pythia)
print()


# get Mix simple predictions on herwig test data and ROC curve
preds_mix_simple_herwig = dnn_mix_simple.predict(X_herwig_test.reshape(-1,X_herwig_val.shape[1]*X_herwig_val.shape[2]), batch_size=1000)
dnn_fp_mix_simple_herwig, dnn_tp_mix_simple_herwig, threshs_mix_simple_herwig = roc_curve(Y_herwig_test[:,1], preds_mix_simple_herwig[:,1])
auc_mix_simple_herwig  = roc_auc_score(Y_herwig_test[:,1], preds_mix_simple_herwig[:,1])
# Get Prediction Time
mix_simple_pred_time_on_herwig = []
print('Mix/H Simple Prediction Time:')
for i in range(6):
    pred_time_callback = PredictionTimeHistory()
    predictions = dnn_mix_simple.predict(X_herwig_test.reshape(-1,X_herwig_val.shape[1]*X_herwig_val.shape[2]), batch_size=1000, verbose=1, callbacks=[pred_time_callback])
    #print(pred_time_callback.times)
    if i>0:
        mix_simple_pred_time_on_herwig.append(pred_time_callback.times)
    i=i+1
mix_simple_pred_time_on_herwig = np.array(mix_simple_pred_time_on_herwig)
MixH_simple_avg_pred_time = np.mean(mix_simple_pred_time_on_herwig)
print()
print('Mix/Herwig Simple DNN AUC:', auc_mix_simple_herwig)
print()


### Mix Simple Pareto ###
with open(f'/users/yzhou276/work/qgtag/simple/dnn/auc/best_mix_dnn_nlayers{nLayers}_dense{layerSize}.txt', 'w') as f:
    f.write(f'P8A {auc_mix_simple_pythia}\n')
    f.write(f'H7A {auc_mix_simple_herwig}\n')
    f.write(f'UNC {np.abs(auc_mix_simple_pythia-auc_mix_simple_herwig)/auc_mix_simple_pythia}\n')
    f.write(f'P8A Pred Time {MixP_simple_avg_pred_time}\n')
    f.write(f'H7A Pred Time {MixH_simple_avg_pred_time}\n')
    f.write(f'GPU {gpu}')
