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
################################################################################

# load Pythia training data
print('Loading the Pythia training dataset ...')
X_pythia, y_pythia = qg_jets.load(train_pythia + val_pythia + test_pythia, generator='pythia')
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

print('Pythia Shape:',X_pythia.shape)
print('Herwig Shape:',X_herwig.shape)

print('Model summary:')
# build architecture
pfn_pythia_teacher = PFN(input_dim=X_pythia.shape[-1], Phi_sizes=Phi_sizes_teacher, F_sizes=F_sizes_teacher)
pfn_herwig_teacher = PFN(input_dim=X_herwig.shape[-1], Phi_sizes=Phi_sizes_teacher, F_sizes=F_sizes_teacher)

# train the pythia teacher model
if(args.doEarlyStopping):
    from keras.callbacks import EarlyStopping,ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    mc = ModelCheckpoint(filepath =f'/users/yzhou276/work/QGtag/teacher_save/pfn_model/best_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_pythia.keras', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    print("Training pythia teacher:")
    pfn_pythia_teacher.fit(X_pythia_train, Y_pythia_train,
                    epochs=num_epoch,
                    batch_size=batch_size,
                    validation_data=(X_pythia_val, Y_pythia_val),
                    verbose=1,
                    callbacks=[es,mc])
else:
    print("Training teacher:")
    pfn_pythia_teacher.fit(X_pythia_train, Y_pythia_train,
                    epochs=num_epoch,
                    batch_size=batch_size,
                    validation_data=(X_pythia_val, Y_pythia_val),
                    verbose=1)
    pfn_pythia_teacher.save(f'/users/yzhou276/work/QGtag/teacher_save/pfn_model/best_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_pythia.keras')


# train the herwig teacher model
if(args.doEarlyStopping):
    from keras.callbacks import EarlyStopping,ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    mc = ModelCheckpoint(filepath =f'/users/yzhou276/work/QGtag/teacher_save/pfn_model/best_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_herwig.keras', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    print("Training herwig teacher:")
    pfn_herwig_teacher.fit(X_herwig_train, Y_herwig_train,
                    epochs=num_epoch,
                    batch_size=batch_size,
                    validation_data=(X_herwig_val, Y_herwig_val),
                    verbose=1,
                    callbacks=[es,mc])
else:
    print("Training herewig teacher:")
    pfn_herwig_teacher.fit(X_herwig_train, Y_herwig_train,
                    epochs=num_epoch,
                    batch_size=batch_size,
                    validation_data=(X_herwig_val, Y_herwig_val),
                    verbose=1)
    pfn_pythia_teacher.save(f'/users/yzhou276/work/QGtag/teacher_save/pfn_model/best_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_herwig.keras')


with open('/users/yzhou276/work/QGtag/teacher_save/pfn_auc.txt', 'a') as f:
    f.write(f"{Phi_sizes_teacher}, {F_sizes_teacher} PFN Patience:{patience}\n")

# get Pythia teacher predictions on pythia test data and ROC curve
preds_pythia_teacher_pythia = pfn_pythia_teacher.predict(X_pythia_test, batch_size=1000)
pfn_fp_pythia_teacher_pythia, pfn_tp_pythia_teacher_pythia, threshs_pythia_teacher_pythia = roc_curve(Y_pythia_test[:,1], preds_pythia_teacher_pythia[:,1])
auc_pythia_teacher_pythia  = roc_auc_score(Y_pythia_test[:,1], preds_pythia_teacher_pythia[:,1])
print()
print('Pythia/Pythia PFN AUC:', auc_pythia_teacher_pythia)
print()
with open('/users/yzhou276/work/QGtag/teacher_save/pfn_auc.txt', 'a') as f:
    f.write(f"Pythia/Pythia: {auc_pythia_teacher_pythia}\n")
# get Pythia teacher predictions on herwig test data and ROC curve
preds_pythia_teacher_herwig = pfn_pythia_teacher.predict(X_herwig_test, batch_size=1000)
pfn_fp_pythia_teacher_herwig, pfn_tp_pythia_teacher_herwig, threshs_pythia_teacher_herwig = roc_curve(Y_herwig_test[:,1], preds_pythia_teacher_herwig[:,1])
auc_pythia_teacher_herwig  = roc_auc_score(Y_herwig_test[:,1], preds_pythia_teacher_herwig[:,1])
print()
print('Pythia/Herwig PFN AUC:', auc_pythia_teacher_herwig)
print()
with open('/users/yzhou276/work/QGtag/teacher_save/pfn_auc.txt', 'a') as f:
    f.write(f"Pythia/Herwig: {auc_pythia_teacher_herwig}\n")


# get Herwig teacher predictions on herwig test data and ROC curve
preds_herwig_teacher_herwig = pfn_herwig_teacher.predict(X_herwig_test, batch_size=1000)
pfn_fp_herwig_teacher_herwig, pfn_tp_herwig_teacher_herwig, threshs_herwig_teacher_herwig = roc_curve(Y_herwig_test[:,1], preds_herwig_teacher_herwig[:,1])
auc_herwig_teacher_herwig  = roc_auc_score(Y_herwig_test[:,1], preds_herwig_teacher_herwig[:,1])
print()
print('Herwig/Herwig PFN AUC:', auc_herwig_teacher_herwig)
print()
with open('/users/yzhou276/work/QGtag/teacher_save/pfn_auc.txt', 'a') as f:
    f.write(f"Herwig/Herwig: {auc_herwig_teacher_herwig}\n")
# get Herwig teacher predictions on pythia test data and ROC curve
preds_herwig_teacher_pythia = pfn_herwig_teacher.predict(X_pythia_test, batch_size=1000)
pfn_fp_herwig_teacher_pythia, pfn_tp_herwig_teacher_pythia, threshs_herwig_teacher_pythia = roc_curve(Y_pythia_test[:,1], preds_herwig_teacher_pythia[:,1])
auc_herwig_teacher_pythia  = roc_auc_score(Y_pythia_test[:,1], preds_herwig_teacher_pythia[:,1])
print()
print('Herwig/Pythia PFN AUC:', auc_herwig_teacher_pythia)
print()
with open('/users/yzhou276/work/QGtag/teacher_save/pfn_auc.txt', 'a') as f:
    f.write(f"Herwig/Pythia: {auc_herwig_teacher_pythia}\n\n")
