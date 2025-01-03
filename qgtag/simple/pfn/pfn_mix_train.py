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

####################
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

# Add Arguments
parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-nEpochs", dest='nEpochs', default=200, type=int, required=False,
                    help="How many epochs to train for?")
parser.add_argument("-batchSize", dest='batchSize', default=500, type=int, required=False,
                    help="How large should the batch size be?")
parser.add_argument("-latentSize", dest='latentSize', default=128, type=int, required=False,
                    help="What is the dimension of the per-particle embedding? n.b. must be a power of 2!")
parser.add_argument("-phiSizes", dest='phiSizes', default=100, type=int, required=False,
                    help="What is the dimension of each layer in phi network")
parser.add_argument("-doEarlyStopping", dest='doEarlyStopping', action='store_true', required=False,
                    help="Do early stopping?")
parser.add_argument("-patience", dest='patience', default=20, type=int, required=False,
                    help="How patient")
parser.add_argument("-usePIDs", dest='usePIDs', action='store_false', required=False,
                    help="Use PIDs? If True, this script will currently break!")
parser.add_argument("-ModelNum", dest='ModelNum', default=0, type=int, required=False,
                    help="label each model")
args = parser.parse_args()

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
Phi_sizes_teacher, F_sizes_teacher = (args.phiSizes, args.phiSizes, args.latentSize), (args.phiSizes, args.phiSizes, args.phiSizes)

# network training parameters
num_epoch = args.nEpochs
if(args.doEarlyStopping):
    num_epoch = 500
batch_size = args.batchSize
patience = args.patience
model_num=args.ModelNum
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
 Y_pythia_train, Y_pythia_val, Y_pythia_test) = data_split(X_pythia, Y_pythia, val=val_pythia, test=test_pythia, shuffle=False)
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
 Y_herwig_train, Y_herwig_val, Y_herwig_test) = data_split(X_herwig, Y_herwig, val=val_herwig, test=test_herwig, shuffle=False)
print('Done herwig train/val/test split')


mix_num = 1500000
train_mix, val_mix, test_mix = int(mix_num*train_ratio), int(mix_num*val_ratio), int(mix_num*test_ratio)

pythia_ratio = 0.5
herwig_ratio = 1 - pythia_ratio

X_mix = np.concatenate((X_pythia[0:int(mix_num*pythia_ratio)],X_herwig[0:int(mix_num*herwig_ratio)]),axis=0)
X_mix_train = np.concatenate((X_pythia_train[0:int(train_mix*pythia_ratio)],X_herwig_train[0:int(train_mix*herwig_ratio)]),axis=0)
X_mix_val = np.concatenate((X_pythia_val[0:int(val_mix*pythia_ratio)],X_herwig_val[0:int(val_mix*herwig_ratio)]),axis=0)
X_mix_test = np.concatenate((X_pythia_test[0:int(test_mix*pythia_ratio)],X_herwig_test[0:int(test_mix*herwig_ratio)]),axis=0)
Y_mix = np.concatenate((Y_pythia[0:int(mix_num*pythia_ratio)],Y_herwig[0:int(mix_num*herwig_ratio)]),axis=0)
Y_mix_train = np.concatenate((Y_pythia_train[0:int(train_mix*pythia_ratio)],Y_herwig_train[0:int(train_mix*herwig_ratio)]),axis=0)
Y_mix_val = np.concatenate((Y_pythia_val[0:int(val_mix*pythia_ratio)],Y_herwig_val[0:int(val_mix*herwig_ratio)]),axis=0)
Y_mix_test = np.concatenate((Y_pythia_test[0:int(test_mix*pythia_ratio)],Y_herwig_test[0:int(test_mix*herwig_ratio)]),axis=0)
print('Done Mixing Pythia and Herwig')


print('Pythia Shape:',X_pythia.shape)
print('Herwig Shape:',X_herwig.shape)
print('Mix Shape:', X_mix.shape)


####################
print('Model summary:')
# build architecture
pfn_mix_teacher = PFN(input_dim=X_pythia.shape[-1], Phi_sizes=Phi_sizes_teacher, F_sizes=F_sizes_teacher)

# train the pythia teacher model
if(args.doEarlyStopping):
    from keras.callbacks import EarlyStopping,ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    mc = ModelCheckpoint(filepath =f'/users/yzhou276/work/qgtag/simple/pfn/model/best_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_mix_{pythia_ratio}Pythia_{herwig_ratio}Herwig_{model_num}.keras', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    print("Training mix teacher:")
    pfn_mix_teacher.fit(X_mix_train, Y_mix_train,
                    epochs=num_epoch,
                    batch_size=batch_size,
                    validation_data=(X_mix_val, Y_mix_val),
                    verbose=1,
                    callbacks=[es,mc])
else:
    print("Training mix teacher:")
    pfn_mix_teacher.fit(X_mix_train, Y_mix_train,
                    epochs=num_epoch,
                    batch_size=batch_size,
                    validation_data=(X_mix_val, Y_mix_val),
                    verbose=1)
    pfn_mix_teacher.save(f'/users/yzhou276/work/qgtag/simple/pfn/model/best_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_mix_{pythia_ratio}Pythia_{herwig_ratio}Herwig_{model_num}.keras')


####################

gpu = print_gpu_info()

# get Mix teacher predictions on pythia test data and ROC curve
preds_mix_teacher_pythia = pfn_mix_teacher.predict(X_pythia_test, batch_size=1000)
pfn_fp_mix_teacher_pythia, pfn_tp_mix_teacher_pythia, threshs_mix_teacher_pythia = roc_curve(Y_pythia_test[:,1], preds_mix_teacher_pythia[:,1])
auc_mix_teacher_pythia  = roc_auc_score(Y_pythia_test[:,1], preds_mix_teacher_pythia[:,1])
# Get Prediction Time
mix_teacher_pred_time_on_pythia = []
print('Mix/P Teacher Prediction Time:')
for i in range(6):
    pred_time_callback = PredictionTimeHistory()
    predictions = pfn_mix_teacher.predict(X_pythia_test, batch_size=1000, verbose=1, callbacks=[pred_time_callback])
    #print(pred_time_callback.times)
    if i>0:
        mix_teacher_pred_time_on_pythia.append(pred_time_callback.times)
    i=i+1
mix_teacher_pred_time_on_pythia = np.array(mix_teacher_pred_time_on_pythia)
MixP_teacher_avg_pred_time = np.mean(mix_teacher_pred_time_on_pythia)
print()
print('Mix/Pythia PFN AUC:', auc_mix_teacher_pythia)
print()


# get Mix teacher predictions on herwig test data and ROC curve
preds_mix_teacher_herwig = pfn_mix_teacher.predict(X_herwig_test, batch_size=1000)
pfn_fp_mix_teacher_herwig, pfn_tp_mix_teacher_herwig, threshs_mix_teacher_herwig = roc_curve(Y_herwig_test[:,1], preds_mix_teacher_herwig[:,1])
auc_mix_teacher_herwig  = roc_auc_score(Y_herwig_test[:,1], preds_mix_teacher_herwig[:,1])
# Get Prediction Time
mix_teacher_pred_time_on_herwig = []
print('Mix/H Teacher Prediction Time:')
for i in range(6):
    pred_time_callback = PredictionTimeHistory()
    predictions = pfn_mix_teacher.predict(X_herwig_test, batch_size=1000, verbose=1, callbacks=[pred_time_callback])
    #print(pred_time_callback.times)
    if i>0:
        mix_teacher_pred_time_on_herwig.append(pred_time_callback.times)
    i=i+1
mix_teacher_pred_time_on_herwig = np.array(mix_teacher_pred_time_on_herwig)
MixH_teacher_avg_pred_time = np.mean(mix_teacher_pred_time_on_herwig)
print()
print('Mix/Herwig PFN AUC:', auc_mix_teacher_herwig)
print()


### Mix Pareto ###
with open(f'/users/yzhou276/work/qgtag/simple/pfn/auc/best_mix_pfn_latent{args.latentSize}_phi{args.phiSizes}_{pythia_ratio}Pythia_{herwig_ratio}Herwig_{model_num}.txt', 'w') as f:
    f.write(f'P8A {auc_mix_teacher_pythia}\n')
    f.write(f'H7A {auc_mix_teacher_herwig}\n')
    f.write(f'UNC {np.abs(auc_mix_teacher_pythia-auc_mix_teacher_herwig)/auc_mix_teacher_pythia}\n')
    f.write(f'P8A Pred Time {MixP_teacher_avg_pred_time}\n')
    f.write(f'H7A Pred Time {MixH_teacher_avg_pred_time}\n')
    f.write(f'GPU {gpu}')
