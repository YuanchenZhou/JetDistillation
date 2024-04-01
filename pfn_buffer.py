from __future__ import absolute_import, division, print_function

import numpy as np
import energyflow as ef

import tensorflow as tf
#from keras import __version__
#tf.keras.__version__=__version__

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

# Clear session
#from tensorflow.keras import backend as K
# Garbage Collection
#import gc

'''
num_model = os.getenv('num_model')
if len(sys.argv)>1:
    i = int(sys.argv[1])
else:
    i = None
num_epoch = os.getenv('num_epoch')
'''

parser = argparse.ArgumentParser()
parser.add_argument('-num_model', type=int, help='Total Model Number')
parser.add_argument('-num_epoch', type=int, help='Epoch Number')
parser.add_argument('-model_number', type=int, help='Current model Number')
parser.add_argument('-n_power', type=int, help='layer size, 2^n')
parser.add_argument('-batch_size', type=int, help='batch size')
args = parser.parse_args()
num_model = args.num_model
num_epoch = args.num_epoch
n = args.n_power
l = 2**n
i = args.model_number+1
batch = args.batch_size

# Total add up to 2000000
train, val, test = 1500000, 200000, 300000
use_pids = False
Phi_sizes, F_sizes = (100, 100,l), (100, 100, 100)
batch_size = batch


X, y = qg_jets.load(train + val + test)
Y = to_categorical(y, num_classes=2)
print('Loaded quark and gluon jets')
for x in X:
    mask = x[:,0] > 0
    yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
    x[mask,1:3] -= yphi_avg
    x[mask,0] /= x[:,0].sum()
if use_pids:
    remap_pids(X, pid_i=3)
else:
    X = X[:,:,:3]
print('Finished preprocessing')
#Do train/val/test split
(X_train, X_val, X_test, Y_train, Y_val, Y_test) = data_split(X, Y, val=val, test=test)
print('Done train/val/test split')
print('Model summary:')

#Set up Early Stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint(filepath=f'/users/yzhou276/work/JetDistillation/buffer_save/buffer_model/{i}_of_{num_model}_{num_epoch}epoch_phi(100,100,{l})_es_best_pfn.keras', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
# Build architecture
pfn = PFN(input_dim=X.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes)
# train model
return_val=pfn.fit(X_train, Y_train,
                   epochs=num_epoch,
                   batch_size=batch_size,
                   validation_data=(X_val, Y_val),
                   verbose=1,callbacks=[es,mc]) # verbose: how much information about the training progress is displayed, callbacks=([es,mc])


# get predictions on test data
preds = pfn.predict(X_test, batch_size=1000)


# get ROC curve
pfn_fp, pfn_tp, threshs = roc_curve(Y_test[:,1], preds[:,1])
# get area under the ROC curve
auc = roc_auc_score(Y_test[:,1], preds[:,1])
print()
print('PFN AUC:', auc)
print()
# get multiplicity and mass for comparison
masses = np.asarray([ef.ms_from_p4s(ef.p4s_from_ptyphims(x).sum(axis=0)) for x in X])
mults = np.asarray([np.count_nonzero(x[:,0]) for x in X])
mass_fp, mass_tp, threshs = roc_curve(Y[:,1], -masses)
mult_fp, mult_tp, threshs = roc_curve(Y[:,1], -mults)
# some nicer plot settings
plt.rcParams['figure.figsize'] = (4,4)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.autolayout'] = True
# plot the ROC curves
plt.plot(pfn_tp, 1-pfn_fp, '-', color='black', label='PFN')
plt.plot(mass_tp, 1-mass_fp, '-', color='blue', label='Jet Mass')
plt.plot(mult_tp, 1-mult_fp, '-', color='red', label='Multiplicity')
# axes labels
plt.xlabel('Quark Jet Efficiency')
plt.ylabel('Gluon Jet Rejection')
# axes limits
plt.xlim(0, 1)
plt.ylim(0, 1)
# make legend and show plot
plt.legend(loc='lower left', frameon=False)
plt.show()
plt.savefig(f'/users/yzhou276/work/JetDistillation/buffer_save/buffer_roc_image/{i}_of_{num_model}_{num_epoch}epoch_phi(100,100,{l})_roc_curve')
with open('/users/yzhou276/work/JetDistillation/buffer_save/roc_auc.txt', 'a') as f:
    f.write(f"{num_epoch}epoch_phi(100,100,{l})_roc_curve_auc:{auc}\n")

# Plot Confusion Matrix
binary_predictions = (preds[:, 1] > 0.5).astype(int)
cm = confusion_matrix(Y_test[:, 1], binary_predictions)
tn, fp, fn, tp = cm.ravel()
# TP: Correctly identifying 1 as 1
# TN: Correctly identifying 0 as 0
# FP: Incorrectly identifying 0 as 1
# FN: Incorrectly identifying 1 as 0
# 0: gluon jet 1:quark jet
ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f')
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.xaxis.set_ticklabels(['Gluon', 'Quark'])
ax.yaxis.set_ticklabels(['Gluon', 'Quark'])
plt.show()
plt.savefig(f'/users/yzhou276/work/JetDistillation/buffer_save/buffer_cm_image/{i}_of_{num_model}_{num_epoch}epoch_phi(100,100,{l})_confusion_matrix')
# Normalize the confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
ax = sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2f')
ax.set_title('Normalized Confusion Matrix')
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.xaxis.set_ticklabels(['Gluon', 'Quark'])
ax.yaxis.set_ticklabels(['Gluon', 'Quark'])
plt.show()
plt.savefig(f'/users/yzhou276/work/JetDistillation/buffer_save/buffer_cm_image/{i}_of_{num_model}_{num_epoch}epoch_phi(100,100,{l})_normalized_confusion_matrix')

# Save the model ...
pfn.model.save(f'/users/yzhou276/work/JetDistillation/buffer_save/buffer_model/{i}_of_{num_model}_{num_epoch}epoch_phi(100,100,{l})_best_pfn.keras') # .h5 / .keras
    
#plt.clf()
#del X, y, Y
#del X_train, X_val, X_test, Y_train, Y_val, Y_test
#del pfn, return_val, preds, binary_predictions
#del cm, tn, fp, fn, tp, cm_normalized, ax
#K.clear_session()
#gc.collect()
