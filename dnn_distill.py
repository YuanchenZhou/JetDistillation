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

###########

# Distillation based on keras example by Kenneth Borup
# https://keras.io/examples/vision/knowledge_distillation/
# Adapted to PFN

class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.loss_tracker = keras.metrics.Mean(name="distillation_loss")

    @property
    def metrics(self):
        metrics = super().metrics
        metrics.append(self.loss_tracker)
        return metrics

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """
        Configure the distiller.
        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    #def compute_loss(
    def train_step(
        self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
    ):
        # Forward pass of teacher
        #teacher_predictions = self.teacher.model(x, training=False)
        teacher_predictions = self.teacher(x[0], training=False)
        #student_loss = self.student_loss_fn(y, y_pred)

        with tf.GradientTape() as tape:
            # Forward pass of student

            student_predictions = self.student.model(x[0].reshape(-1,X_train.shape[1]*X_train.shape[2]),
                                                     training=True)
            '''
            student_predictions = self.student(x[0].reshape(-1,X_train.shape[1]*X_train.shape[2]),
                                                     training=True)
            '''
            # Compute loss
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
        """
        # MLB Note: This kind of combined-loss seems to break things at the moment...
        # just using distillation loss for now, but it works well!

        #student_loss  = self.student_loss_fn(y, y_pred) # Also compute the loss of training the student directly

        # alpha determines how much the student listens to the teacher or trusts itself
        #combined_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        """

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        #gradients = tape.gradient(combined_loss, trainable_vars)
        gradients = tape.gradient(distillation_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Report progress
        #self.loss_tracker.update_state(combined_loss)
        self.loss_tracker.update_state(distillation_loss)
        return {"distillation_loss": self.loss_tracker.result()}
        #return loss

    def call(self, x):
        #return self.student(x.reshape(-1,X_train.shape[1]*X_train.shape[2]))
        return self.student.model(x.reshape(-1,X_train.shape[1]*X_train.shape[2]))


###########
# Main script
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
parser.add_argument("-nLayers",dest='nLayers', default=2, type=int, required=False,
                    help="How many layers for the dnn")
parser.add_argument("-layerSize",dest='layerSize', default=100, type=int,required=False,
                    help="How large should the layer dense size be for the simple and student model")
args = parser.parse_args()


if(args.nEpochs==0 and args.doEarlyStopping==False):
    raise Exception("You need to specify a number of epochs to train for, or to use early stopping!")
# nice : https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two
if( not(args.latentSize & (args.latentSize-1))==0 and args.latentSize!=0):
    raise Exception("The dimension of the per-particle embedding has to be a power of 2!")
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
# PFN
Phi_sizes_teacher, F_sizes_teacher = (args.phiSizes, args.phiSizes, args.latentSize), (args.phiSizes, args.phiSizes, args.phiSizes)
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

print('Pythia Shape:',X_pythia.shape)
print('Herwig Shape:',X_herwig.shape)

# Load Teacher Models
model_save_path = '/users/yzhou276/work/QGtag/teacher_save/pfn_model/'

# Load pythia pfn teacher model
pfn_teacher_pythia = load_model(model_save_path+f'best_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_pythia.keras', safe_mode=False)

# Load herwig pfn teacher model
pfn_teacher_herwig = load_model(model_save_path+f'best_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_herwig.keras', safe_mode=False)


############################################

# Set up the 'simple' and 'student' models, which will be simple fully-connected DNNs

# build architecture
print(X_pythia_train.shape)
print(X_pythia_train.shape[1:])
print(X_pythia_train.shape[1:2])
# dense_sizes above
dnn_pythia_simple  = DNN(input_dim=X_pythia_train.shape[1]*X_pythia_train.shape[2], dense_sizes=dense_sizes)
dnn_pythia_student = DNN(input_dim=X_pythia_train.shape[1]*X_pythia_train.shape[2], dense_sizes=dense_sizes)
# train the simple pythia model
if(args.doEarlyStopping):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    mc = ModelCheckpoint(filepath =f'/users/yzhou276/work/QGtag/distill_save/dnn_models/simple_{dense_sizes}_dnn_by_pythia.keras', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    dnn_pythia_simple.fit(X_pythia_train.reshape(-1,X_pythia_train.shape[1]*X_pythia_train.shape[2]), Y_pythia_train,
                   epochs=num_epoch,
                   batch_size=batch_size,
                   validation_data=(X_pythia_val.reshape(-1,X_pythia_val.shape[1]*X_pythia_val.shape[2]), Y_pythia_val),
                   verbose=1,
                   callbacks=[es,mc])
else:
    dnn_pythia_simple.fit(X_pythia_train.reshape(-1,X_pythia_train.shape[1]*X_pythia_train.shape[2]), Y_pythia_train,
                   epochs=num_epoch,
                   batch_size=batch_size,
                   validation_data=(X_pythia_val.reshape(-1,X_pythia_val.shape[1]*X_pythia_val.shape[2]), Y_pythia_val),
                   verbose=1)
    dnn_pythia_simple.save(f'/users/yzhou276/work/QGtag/distill_save/dnn_models/simple_{dense_sizes}_dnn_by_pythia.keras')

############################################

# train the pythia student model
X_train = X_pythia_train
distiller_pythia = Distiller(student=dnn_pythia_student, teacher=pfn_teacher_pythia)

distiller_pythia.compile(
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.CategoricalCrossentropy()],
    student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=0.5, # was 0.1 but doesn't do anything right now
    temperature=3.0,)

print("Training pythia student:")
distiller_pythia.fit(X_pythia_train,
              Y_pythia_train,#,_for_distiller,
              epochs=20, #num_epoch,
              batch_size=batch_size,
              validation_data=(X_pythia_val, Y_pythia_val),#_for_distiller),
              verbose=1)

dnn_pythia_student.save(f'/users/yzhou276/work/QGtag/distill_save/dnn_models/student_{dense_sizes}_dnn_by_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_pythia.keras')
#########################################################################
# Set up the 'simple' and 'student' models, which will be simple fully-connected DNNs
# build architecture
print(X_herwig_train.shape)
print(X_herwig_train.shape[1:])
print(X_herwig_train.shape[1:2])
# dense_sizes above
dnn_herwig_simple  = DNN(input_dim=X_herwig_train.shape[1]*X_herwig_train.shape[2], dense_sizes=dense_sizes)
dnn_herwig_student = DNN(input_dim=X_herwig_train.shape[1]*X_herwig_train.shape[2], dense_sizes=dense_sizes)
# train the simple herwig model
if(args.doEarlyStopping):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    mc = ModelCheckpoint(filepath =f'/users/yzhou276/work/QGtag/distill_save/dnn_models/simple_{dense_sizes}_dnn_by_herwig.keras', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    dnn_herwig_simple.fit(X_herwig_train.reshape(-1,X_herwig_train.shape[1]*X_herwig_train.shape[2]), Y_herwig_train,
                   epochs=num_epoch,
                   batch_size=batch_size,
                   validation_data=(X_herwig_val.reshape(-1,X_herwig_val.shape[1]*X_herwig_val.shape[2]), Y_herwig_val),
                   verbose=1,
                   callbacks=[es,mc])
else:
    dnn_herwig_simple.fit(X_herwig_train.reshape(-1,X_herwig_train.shape[1]*X_herwig_train.shape[2]), Y_herwig_train,
                   epochs=num_epoch,
                   batch_size=batch_size,
                   validation_data=(X_herwig_val.reshape(-1,X_herwig_val.shape[1]*X_herwig_val.shape[2]), Y_herwig_val),
                   verbose=1)
    dnn_herwig_simple.save(f'/users/yzhou276/work/QGtag/distill_save/dnn_models/simple_{dense_sizes}_dnn_by_herwig.keras')

############################################
# train the herwig student model
X_train = X_herwig_train
distiller_herwig = Distiller(student=dnn_herwig_student, teacher=pfn_teacher_herwig)
distiller_herwig.compile(
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.CategoricalCrossentropy()],
    student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=0.5, # was 0.1 but doesn't do anything right now
    temperature=3.0,)

print("Training herwig student:")
distiller_herwig.fit(X_herwig_train,
                     Y_herwig_train,#,_for_distiller,
                     epochs=20, #num_epoch,
                     batch_size=batch_size,
                     validation_data=(X_herwig_val, Y_herwig_val),#_for_distiller),
                     verbose=1)

dnn_herwig_student.save(f'/users/yzhou276/work/QGtag/distill_save/dnn_models/student_{dense_sizes}_dnn_by_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_herwig.keras')

#########################################################################

with open('/users/yzhou276/work/QGtag/distill_save/dnn_auc.txt', 'a') as f:
    f.write(f"{Phi_sizes_teacher} {F_sizes_teacher} PFN, {dense_sizes} DNN, Patience:{patience}\n")

# get Pythia student predictions on pythia test data and ROC curve
preds_pythia_student_pythia = dnn_pythia_student.predict(X_pythia_test.reshape(-1,X_pythia_val.shape[1]*X_pythia_val.shape[2]), batch_size=1000)
dnn_fp_pythia_student_pythia, dnn_tp_pythia_student_pythia, threshs_pythia_student_pythia = roc_curve(Y_pythia_test[:,1], preds_pythia_student_pythia[:,1])
auc_pythia_student_pythia  = roc_auc_score(Y_pythia_test[:,1], preds_pythia_student_pythia[:,1])
print()
print('Pythia/Pythia Student DNN AUC:', auc_pythia_student_pythia)
print()
with open('/users/yzhou276/work/QGtag/distill_save/dnn_auc.txt', 'a') as f:
    f.write(f"Student DNN AUC Pythia/Pythia: {auc_pythia_student_pythia}\n")
# get Pythia student predictions on herwig test data and ROC curve
preds_pythia_student_herwig = dnn_pythia_student.predict(X_herwig_test.reshape(-1,X_herwig_val.shape[1]*X_herwig_val.shape[2]), batch_size=1000)
dnn_fp_pythia_student_herwig, dnn_tp_pythia_student_herwig, threshs_pythia_student_herwig = roc_curve(Y_herwig_test[:,1], preds_pythia_student_herwig[:,1])
auc_pythia_student_herwig  = roc_auc_score(Y_herwig_test[:,1], preds_pythia_student_herwig[:,1])
print()
print('Pythia/Herwig Student DNN AUC:', auc_pythia_student_herwig)
print()
with open('/users/yzhou276/work/QGtag/distill_save/dnn_auc.txt', 'a') as f:
    f.write(f"Student DNN AUC Pythia/Herwig: {auc_pythia_student_herwig}\n")

# get Pythia simple predictions on pythia test data and ROC curve
preds_pythia_simple_pythia = dnn_pythia_simple.predict(X_pythia_test.reshape(-1,X_pythia_val.shape[1]*X_pythia_val.shape[2]), batch_size=1000)
dnn_fp_pythia_simple_pythia, dnn_tp_pythia_simple_pythia, threshs_pythia_simple_pythia = roc_curve(Y_pythia_test[:,1], preds_pythia_simple_pythia[:,1])
auc_pythia_simple_pythia  = roc_auc_score(Y_pythia_test[:,1], preds_pythia_simple_pythia[:,1])
print()
print('Pythia/Pythia Simple DNN AUC:', auc_pythia_simple_pythia)
print()
with open('/users/yzhou276/work/QGtag/distill_save/dnn_auc.txt', 'a') as f:
    f.write(f"Simple DNN AUC Pythia/Pythia: {auc_pythia_simple_pythia}\n")
# get Pythia simple predictions on herwig test data and ROC curve
preds_pythia_simple_herwig = dnn_pythia_simple.predict(X_herwig_test.reshape(-1,X_herwig_val.shape[1]*X_herwig_val.shape[2]), batch_size=1000)
dnn_fp_pythia_simple_herwig, dnn_tp_pythia_simple_herwig, threshs_pythia_simple_herwig = roc_curve(Y_herwig_test[:,1], preds_pythia_simple_herwig[:,1])
auc_pythia_simple_herwig  = roc_auc_score(Y_herwig_test[:,1], preds_pythia_simple_herwig[:,1])
print()
print('Pythia/Herwig Simple DNN AUC:', auc_pythia_simple_herwig)
print()
with open('/users/yzhou276/work/QGtag/distill_save/dnn_auc.txt', 'a') as f:
    f.write(f"Simple DNN AUC Pythia/Herwig: {auc_pythia_simple_herwig}\n")

# get Herwig student predictions on herwig test data and ROC curve
preds_herwig_student_herwig = dnn_herwig_student.predict(X_herwig_test.reshape(-1,X_herwig_val.shape[1]*X_herwig_val.shape[2]), batch_size=1000)
dnn_fp_herwig_student_herwig, dnn_tp_herwig_student_herwig, threshs_herwig_student_herwig = roc_curve(Y_herwig_test[:,1], preds_herwig_student_herwig[:,1])
auc_herwig_student_herwig  = roc_auc_score(Y_herwig_test[:,1], preds_herwig_student_herwig[:,1])
print()
print('Herwig/Herwig Student DNN AUC:', auc_herwig_student_herwig)
print()
with open('/users/yzhou276/work/QGtag/distill_save/dnn_auc.txt', 'a') as f:
    f.write(f"Student DNN AUC Herwig/Herwig: {auc_herwig_student_herwig}\n")
# get Herwig student predictions on pythia test data and ROC curve
preds_herwig_student_pythia = dnn_herwig_student.predict(X_pythia_test.reshape(-1,X_pythia_val.shape[1]*X_pythia_val.shape[2]), batch_size=1000)
dnn_fp_herwig_student_pythia, dnn_tp_herwig_student_pythia, threshs_herwig_student_pythia = roc_curve(Y_pythia_test[:,1], preds_herwig_student_pythia[:,1])
auc_herwig_student_pythia  = roc_auc_score(Y_pythia_test[:,1], preds_herwig_student_pythia[:,1])
print()
print('Herwig/Pythia Student DNN AUC:', auc_herwig_student_pythia)
print()
with open('/users/yzhou276/work/QGtag/distill_save/dnn_auc.txt', 'a') as f:
    f.write(f"Student DNN AUC Herwig/Pythia: {auc_herwig_student_pythia}\n")
    
# get Herwig simple predictions on herwig test data and ROC curve
preds_herwig_simple_herwig = dnn_herwig_simple.predict(X_herwig_test.reshape(-1,X_herwig_val.shape[1]*X_herwig_val.shape[2]), batch_size=1000)
dnn_fp_herwig_simple_herwig, dnn_tp_herwig_simple_herwig, threshs_herwig_simple_herwig = roc_curve(Y_herwig_test[:,1], preds_herwig_simple_herwig[:,1])
auc_herwig_simple_herwig  = roc_auc_score(Y_herwig_test[:,1], preds_herwig_simple_herwig[:,1])
print()
print('Herwig/Herwig Simple DNN AUC:', auc_herwig_simple_herwig)
print()
with open('/users/yzhou276/work/QGtag/distill_save/dnn_auc.txt', 'a') as f:
    f.write(f"Simple DNN AUC Herwig/Herwig: {auc_herwig_simple_herwig}\n")
# get Herwig simple predictions on pythia test data and ROC curve
preds_herwig_simple_pythia = dnn_herwig_simple.predict(X_pythia_test.reshape(-1,X_pythia_val.shape[1]*X_pythia_val.shape[2]), batch_size=1000)
dnn_fp_herwig_simple_pythia, dnn_tp_herwig_simple_pythia, threshs_herwig_simple_pythia = roc_curve(Y_pythia_test[:,1], preds_herwig_simple_pythia[:,1])
auc_herwig_simple_pythia  = roc_auc_score(Y_pythia_test[:,1], preds_herwig_simple_pythia[:,1])
print()
print('Herwig/Pythia Simple DNN AUC:', auc_herwig_simple_pythia)
print()
with open('/users/yzhou276/work/QGtag/distill_save/dnn_auc.txt', 'a') as f:
    f.write(f"Simple DNN AUC Herwig/Pythia: {auc_herwig_simple_pythia}\n\n")
