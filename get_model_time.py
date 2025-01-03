from __future__ import absolute_import, division, print_function
import numpy as np
import energyflow as ef
import tensorflow as tf
from energyflow.archs.efn import PFN
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, remap_pids, to_categorical
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import sys
import argparse

from keras.models import load_model

import time


class EvaluateTimeHistory(tf.keras.callbacks.Callback):
    def on_test_begin(self, logs=None):
        self.times = []
        self.total_time = 0
    def on_test_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.perf_counter()
    def on_test_batch_end(self, batch, logs=None):
        self.batch_end_time = time.perf_counter()
        batch_time = self.batch_end_time - self.batch_start_time
        self.times.append(batch_time)
        self.total_time += batch_time


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
    if not gpus:
        print("No GPUs found.")
    else:
        for gpu in gpus:
            print(f"Device: {gpu.device_type}, Name: {gpu.name}")
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                gpu_name = gpu_details.get('device_name', 'Unknown GPU Name')
                pci_bus_id = gpu_details.get('pci_bus_id', 'Unknown PCI Bus ID')
                print(f"Details - Name: {gpu_name}, PCI bus ID: {pci_bus_id}")
            except Exception as e:
                print(f"Could not retrieve details for GPU: {gpu.name}. Error: {str(e)}")

print('Loading Data')
# Pythia 2000000 max
# Herwig 1500000 max
num_load_pythia = 5000
num_load_herwig = 5000

X_pythia, y_pythia = qg_jets.load(num_load_pythia, generator='pythia')
Y_pythia = to_categorical(y_pythia, num_classes=2)
n_pythia_pad = 148 - X_pythia.shape[1]
X_pythia = np.lib.pad(X_pythia, ((0,0), (0,n_pythia_pad), (0,0)), mode='constant', constant_values=0)

X_herwig, y_herwig = qg_jets.load(num_load_herwig, generator='herwig')
Y_herwig = to_categorical(y_herwig, num_classes=2)
n_herwig_pad =  148 - X_herwig.shape[1]
X_herwig = np.lib.pad(X_herwig, ((0,0), (0,n_herwig_pad), (0,0)), mode='constant', constant_values=0)

use_pids = False

print('Loaded quark and gluon jets')
for x in X_pythia:
    mask = x[:,0] > 0
    yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
    x[mask,1:3] -= yphi_avg
    x[mask,0] /= x[:,0].sum()
if use_pids:
    remap_pids(X_pythia, pid_i=3)
else:
    X_pythia = X_pythia[:,:,:3]
print('Finished preprocessing pythia data')

for x in X_herwig:
    mask = x[:,0] > 0
    yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
    x[mask,1:3] -= yphi_avg
    x[mask,0] /= x[:,0].sum()
if use_pids:
    remap_pids(X_herwig, pid_i=3)
else:
    X_herwig = X_herwig[:,:,:3]
print('Finished preprocessing herwig data')

print('Pythia Shape:',X_pythia.shape)
print('Herwig Shape:',X_herwig.shape)

print_gpu_info()


# Teacher Model, (250, 250, 128) (250, 250, 250) PFN
pfn_save_path = '/users/yzhou276/work/QGtag/teacher_save/pfn_model/'
pfn_latentSize = 128
pfn_FSizes = (250, 250, pfn_latentSize)
pfn_phiSizes = (250, 250, 250)
pythia_teacher_model = load_model(pfn_save_path+f'best_{pfn_FSizes}_{pfn_phiSizes}_pfn_pythia.keras',safe_mode=False)
#herwig_teacher_model = load_model(pfn_save_path+f'best_{pfn_FSizes}_{pfn_phiSizes}_pfn_herwig.keras',safe_mode=False)
print(pythia_teacher_model.summary())
#print(herwig_teacher_model.summary())


batchsize = 1000
# Pythia Teacher on Pythia
# Get Evaluation Time
pythia_teacher_eva_time_on_pythia = []
print('P/P Teacher Evaluation Time:')
for i in range(6):
    evaluate_time_callback = EvaluateTimeHistory()
    loss, acc = pythia_teacher_model.evaluate(X_pythia, Y_pythia,  batch_size=batchsize, verbose=1, callbacks=[evaluate_time_callback])
#    print(evaluate_time_callback.times)
    if i>0:
        pythia_teacher_eva_time_on_pythia.append(evaluate_time_callback.times)
    i=i+1
pythia_teacher_eva_time_on_pythia = np.array(pythia_teacher_eva_time_on_pythia)
# Get Prediction Time
pythia_teacher_pred_time_on_pythia = []
print('P/P Teacher Prediction Time:')
for i in range(6):
    pred_time_callback = PredictionTimeHistory()
    predictions = pythia_teacher_model.predict(X_pythia, batch_size=batchsize, verbose=1, callbacks=[pred_time_callback])
#    print(pred_time_callback.times)
    if i>0:
        pythia_teacher_pred_time_on_pythia.append(pred_time_callback.times)
    i=i+1
pythia_teacher_pred_time_on_pythia = np.array(pythia_teacher_pred_time_on_pythia)
PP_teacher_avg_eva_time = np.mean(pythia_teacher_eva_time_on_pythia)
PP_teacher_avg_pred_time = np.mean(pythia_teacher_pred_time_on_pythia)
print(pythia_teacher_eva_time_on_pythia)
print(PP_teacher_avg_eva_time)
print(pythia_teacher_pred_time_on_pythia)
print(PP_teacher_avg_pred_time)
