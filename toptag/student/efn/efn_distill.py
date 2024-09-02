# standard library imports
from __future__ import absolute_import, division, print_function
import argparse

# Data I/O and numerical imports
#import h5py
import numpy as np

# ML imports
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
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
from energyflow.archs.efn import EFN
#from energyflow.archs.dnn import DNN
#from energyflow.datasets import qg_jets
from energyflow.datasets import ttag_jets
from energyflow.utils import data_split, remap_pids, to_categorical

# Plotting imports
import matplotlib.pyplot as plt

import time

###########

# Function borrowed from Rikab
# https://github.com/rikab/GaussianAnsatz
from keras.layers import Dense, Dropout, Input, Concatenate
from keras.models import Model

def efn_input_converter(model_efn, shape=None, num_global_features=0):
    if num_global_features == 0:
        input_layer = Input(shape=shape)
        output = model_efn([input_layer[:, :, 0], input_layer[:, :, 1:]])
        return Model(input_layer, output)
    else:
        input_layer_1 = Input(shape=shape)
        input_layer_2 = Input(shape=(num_global_features,))
        output = model_efn([input_layer_1[:, :, 0], input_layer_1[:, :, 1:], input_layer_2])
        return Model([input_layer_1, input_layer_2], output)

# Distillation based on keras example by Kenneth Borup
# https://keras.io/examples/vision/knowledge_distillation/
# Adapted to PFN


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


class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.loss_tracker = keras.metrics.Mean(name="distillation_loss")

    def get_config(self):
        config = super().get_config()
        config.update({
            'student': self.student,
            'teacher': self.teacher
        })
        return config

    def from_config(cls, config):
        return cls(**config)

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
        # n.b. in train_step x is a tuple of the information in X & y

        # Forward pass of teacher
        #teacher_predictions = self.teacher.model(x[0], training=False)
        teacher_predictions = self.teacher(x[0], training=False)
        #student_loss = self.student_loss_fn(x[1], y_pred)

        with tf.GradientTape() as tape:
            # Forward pass of student
            #student_predictions = self.student.model([z,p], training=True)
            student_predictions = self.student(x[0], training=True)

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
        return self.student(x)


class CustomModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto', save_format='tf'):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_format = save_format
        if mode not in ['min', 'max', 'auto']:
            raise ValueError("Mode must be 'min', 'max', or 'auto'")
        
        if mode == 'min':
            self.best = np.Inf
        elif mode == 'max':
            self.best = -np.Inf
        else:
            self.best = None

    def on_train_begin(self, logs=None):
        if self.mode == 'auto':
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.mode = 'max'
                self.best = -np.Inf
            else:
                self.mode = 'min'
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            return
        
        if self.mode == 'min':
            is_improvement = monitor_value < self.best
        else:
            is_improvement = monitor_value > self.best

        if self.save_best_only:
            if is_improvement:
                self.best = monitor_value
                if self.verbose > 0:
                    print(f'\nEpoch {epoch+1}: {self.monitor} improved to {monitor_value}, saving model to {self.filepath}')
                self.model.student.save(self.filepath, save_format=self.save_format)
        else:
            if self.verbose > 0:
                print(f'\nEpoch {epoch+1}: saving model to {self.filepath}')
            self.model.student.save(self.filepath, save_format=self.save_format)


class ModelSnapshot(Callback):
    def __init__(self, patience):
        super(ModelSnapshot, self).__init__()
        self.patience = patience
        self.models = []

    def on_train_begin(self, logs=None):
        self.models = []

    def on_epoch_end(self, epoch, logs=None):
        if len(self.models) > self.patience:
            self.models.pop(0)

        model_copy = self.model.student.get_weights()
        self.models.append(model_copy)

    def on_train_end(self, logs=None):
        print("Training completed. Student models from last epochs stored.")


###########
# Main script

parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-nEpochs", dest='nEpochs', default=200, type=int, required=False,
                    help="How many epochs to train for?")
parser.add_argument("-batchSize", dest='batchSize', default=500, type=int, required=False,
                    help="How large should the batch size be?")
parser.add_argument("-latentSizeTeacher", dest='latentSizeTeacher', default=128, type=int, required=False,
                    help="What is the dimension of the per-particle embedding? n.b. must be a power of 2!")
parser.add_argument("-phiSizesTeacher", dest='phiSizesTeacher', default=100, type=int, required=False,
                    help="What is the dimension of each layer in phi network for the Teacher")
parser.add_argument("-latentSizeStudent", dest='latentSizeStudent', default=128, type=int, required=False,
                    help="What is the dimension of the per-particle embedding? n.b. must be a power of 2!")
parser.add_argument("-phiSizesStudent", dest='phiSizesStudent', default=100, type=int, required=False,
                    help="What is the dimension of each layer in phi network for the Student")
parser.add_argument("-doEarlyStopping", dest='doEarlyStopping', action='store_true', required=False,
                    help="Do early stopping?")
parser.add_argument("-patience", dest='patience', default=10, type=int, required=False,
                    help="How patient?")
parser.add_argument("-usePIDs", dest='usePIDs', action='store_false', required=False,
                    help="Use PIDs? If True, this script will currently break!")
args = parser.parse_args()

if(args.nEpochs==0 and args.doEarlyStopping==False):
    raise Exception("You need to specify a number of epochs to train for, or to use early stopping!")

# nice : https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two
if( not(args.latentSizeTeacher & (args.latentSizeTeacher-1))==0 and args.latentSizeTeacher!=0):
    raise Exception("The dimension of the per-particle embedding has to be a power of 2!")

if( not(args.latentSizeStudent & (args.latentSizeStudent-1))==0 and args.latentSizeStudent!=0):
    raise Exception("The dimension of the per-particle embedding has to be a power of 2!")

#if(args.nEpochs>0 and args.doEarlyStopping):
#    raise Exception("Either specify early stopping, **or** a number of epochs to train for!")

#print("pfn_example.py\tWelcome!")

################################################################################
# data controls, can go up to 2000000 for full Pythia dataset, 1500000 for full Herwig dataset
pythia_num = 1000000
herwig_num = 1000000

#train, val, test = 75000, 10000, 15000 # small
#train, val, test = 150000, 20000, 30000 # medium (2x small, ~0.1x complete)
train_ratio, val_ratio, test_ratio = 0.75, 0.125, 0.125
train_pythia, val_pythia, test_pythia = int(pythia_num*train_ratio), int(pythia_num*val_ratio), int(pythia_num*test_ratio) # complete
train_herwig, val_herwig, test_herwig = int(herwig_num*train_ratio), int(herwig_num*val_ratio), int(herwig_num*test_ratio)
use_pids = args.usePIDs

# network architecture parameters
# PFN Teacher
Phi_sizes_teacher, F_sizes_teacher = (args.phiSizesTeacher, args.phiSizesTeacher, args.latentSizeTeacher), (args.phiSizesTeacher, args.phiSizesTeacher, args.phiSizesTeacher)
# EFN Student
Phi_sizes_student, F_sizes_student = (args.phiSizesStudent, args.phiSizesStudent, args.latentSizeStudent), (args.phiSizesStudent, args.phiSizesStudent, args.phiSizesStudent)

# network training parameters
num_epoch = args.nEpochs
if(args.doEarlyStopping):
    num_epoch = 500
batch_size = args.batchSize
patience = args.patience
################################################################################

# load Pythia training data
print('Loading the Pythia training dataset ...')
X_pythia, y_pythia = ttag_jets.load(train_pythia + val_pythia + test_pythia, generator='pythia')
n_pythia_pad = 200 - X_pythia.shape[1]
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
# For EFN
z_pythia_train = X_pythia_train[:,:,0]
z_pythia_val   = X_pythia_val[:,:,0]
z_pythia_test  = X_pythia_test[:,:,0]
p_pythia_train = X_pythia_train[:,:,1:]
p_pythia_val   = X_pythia_val[:,:,1:]
p_pythia_test  = X_pythia_test[:,:,1:]
print(p_pythia_train.shape, z_pythia_train.shape, X_pythia_train.shape)
print('Done pythia train/val/test split')

# load Herwig training data
print('Loading the Herwig training dataset ...')
X_herwig, y_herwig = ttag_jets.load(train_herwig + val_herwig + test_herwig, generator='herwig', cache_dir='~/.energyflow/herwig')
n_herwig_pad =  200 - X_herwig.shape[1]
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
# For EFN
z_herwig_train = X_herwig_train[:,:,0]
z_herwig_val   = X_herwig_val[:,:,0]
z_herwig_test  = X_herwig_test[:,:,0]
p_herwig_train = X_herwig_train[:,:,1:]
p_herwig_val   = X_herwig_val[:,:,1:]
p_herwig_test  = X_herwig_test[:,:,1:]
print(p_herwig_train.shape, z_herwig_train.shape, X_herwig_train.shape)
print('Done herwig train/val/test split')

print('Pythia Shape:',X_pythia.shape)
print('Herwig Shape:',X_herwig.shape)


# Load Teacher Models
model_save_path = '/users/yzhou276/work/toptag/simple/pfn/model/'

# Load pythia pfn teacher model
pfn_teacher_pythia = load_model(model_save_path+f'best_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_pythia.keras', safe_mode=False)

# Load herwig pfn teacher model
pfn_teacher_herwig = load_model(model_save_path+f'best_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_herwig.keras', safe_mode=False)
############################################

# build architecture, input_dim=2 (y,phi) for pythia EFN
_efn_pythia_student = EFN(input_dim=2, Phi_sizes=Phi_sizes_student, F_sizes=F_sizes_student).model

max_pythia_particles=X_pythia.shape[1]
efn_pythia_student = efn_input_converter(_efn_pythia_student,shape=(max_pythia_particles, 3))
efn_pythia_student.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(),
                    #metrics=["val_acc"]
                    )

############################################

# train the pythia student model

pythia_distiller = Distiller(student=efn_pythia_student, teacher=pfn_teacher_pythia)

pythia_distiller.compile(
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.CategoricalCrossentropy()],
    student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=0.5, # was 0.1 but doesn't do anything right now
    temperature=3.0,
)

print("Training Pythia student:")

if(args.doEarlyStopping):
    #es_d = EarlyStopping(monitor='val_distillation_loss', mode='min', verbose=1, patience=args.patience)
    es_d = EarlyStopping(monitor='val_categorical_crossentropy', mode='auto', verbose=1, patience=args.patience)
    '''
    mc_d = ModelCheckpoint(filepath = f'/users/yzhou276/work/toptag/student/efn/model/student_{Phi_sizes_student}_{F_sizes_student}_efn_by_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_pythia.keras',
                           monitor='val_categorical_crossentropy',
                           mode='auto',
                           verbose=1,
                           save_best_only=True,
                           save_format="tf")
    '''
    mc_d = CustomModelCheckpoint(filepath = f'/users/yzhou276/work/toptag/student/efn/model/student_{Phi_sizes_student}_{F_sizes_student}_efn_by_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_pythia.keras',
                                 monitor='val_categorical_crossentropy',
                                 mode='auto',
                                 verbose=1,
                                 save_best_only=True,
                                 save_format="tf")

    #ms_d = ModelSnapshot(patience = args.patience)

    #print("Training student:")
    hist = pythia_distiller.fit(X_pythia_train,
                  Y_pythia_train,#_for_distiller,
                  epochs=num_epoch,
                  batch_size=batch_size,
                  validation_data=(X_pythia_val, Y_pythia_val),#_for_distiller),
                  verbose=1,
                  callbacks=[es_d,mc_d]) #ms_d])

    #best_weights = ms_d.models[0]
    #efn_pythia_student.set_weights(best_weights)
    #efn_pythia_student.save(f'/users/yzhou276/work/toptag/student/efn/model/student_{Phi_sizes_student}_{F_sizes_student}_efn_by_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_pythia.keras')

else:
    pythia_distiller.fit(X_pythia_train,
              Y_pythia_train,
              epochs=20, #num_epoch,
              batch_size=batch_size,
              validation_data=(X_pythia_val, Y_pythia_val),#_for_distiller),
              verbose=1)
    efn_pythia_student.save(f'/users/yzhou276/work/toptag/student/efn/model/student_{Phi_sizes_student}_{F_sizes_student}_efn_by_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_pythia.keras')

#########################################################################

# build architecture, input_dim=2 (y,phi) for herwig EFN
_efn_herwig_student = EFN(input_dim=2, Phi_sizes=Phi_sizes_student, F_sizes=F_sizes_student).model

max_herwig_particles=X_herwig.shape[1]
efn_herwig_student = efn_input_converter(_efn_herwig_student,shape=(max_herwig_particles, 3))
efn_herwig_student.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(),
                    #metrics=["val_acc"]
                    )

############################################

# train the herwig student model

herwig_distiller = Distiller(student=efn_herwig_student, teacher=pfn_teacher_herwig)

herwig_distiller.compile(
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.CategoricalCrossentropy()],
    student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=0.5, # was 0.1 but doesn't do anything right now
    temperature=3.0,
)

print("Training Herwig student:")

if(args.doEarlyStopping):
    #es_d = EarlyStopping(monitor='val_distillation_loss', mode='min', verbose=1, patience=args.patience)
    es_d = EarlyStopping(monitor='val_categorical_crossentropy', mode='auto', verbose=1, patience=args.patience)
    '''
    mc_d = ModelCheckpoint(filepath = f'/users/yzhou276/work/toptag/student/efn/model/student_{Phi_sizes_student}_{F_sizes_student}_efn_by_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_herwig.keras',
                           monitor='val_categorical_crossentropy',
                           mode='auto',
                           verbose=1,
                           save_best_only=True,
                           save_format="tf")
    '''
    mc_d = CustomModelCheckpoint(filepath = f'/users/yzhou276/work/toptag/student/efn/model/student_{Phi_sizes_student}_{F_sizes_student}_efn_by_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_herwig.keras',
                                 monitor='val_categorical_crossentropy',
                                 mode='auto',
                                 verbose=1,
                                 save_best_only=True,
                                 save_format="tf")

    #ms_d = ModelSnapshot(patience = args.patience)

    #print("Training student:")
    hist = herwig_distiller.fit(X_herwig_train,
                  Y_herwig_train,#_for_distiller,
                  epochs=num_epoch,
                  batch_size=batch_size,
                  validation_data=(X_herwig_val, Y_herwig_val),#_for_distiller),
                  verbose=1,
                  callbacks=[es_d,mc_d]) #ms_d])

    #best_weights = ms_d.models[0]
    #efn_herwig_student.set_weights(best_weights)
    #efn_herwig_student.save(f'/users/yzhou276/work/toptag/student/efn/model/student_{Phi_sizes_student}_{F_sizes_student}_efn_by_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_herwig.keras')

else:
    herwig_distiller.fit(X_herwig_train,
              Y_herwig_train,
              epochs=20, #num_epoch,
              batch_size=batch_size,
              validation_data=(X_herwig_val, Y_herwig_val),#_for_distiller),
              verbose=1)
    efn_herwig_student.save(f'/users/yzhou276/work/toptag/student/efn/model/student_{Phi_sizes_student}_{F_sizes_student}_efn_by_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_herwig.keras')

#########################################################################

gpu = print_gpu_info()

# get Pythia student predictions on pythia test data and ROC curve
preds_pythia_student_pythia = efn_pythia_student.predict(X_pythia_test,
                                    batch_size=1000)
efn_fp_pythia_student_pythia, dnn_tp_pythia_student_pythia, threshs_pythia_student_pythia = roc_curve(Y_pythia_test[:,1], preds_pythia_student_pythia[:,1])
auc_pythia_student_pythia = roc_auc_score(Y_pythia_test[:,1], preds_pythia_student_pythia[:,1])
# Get Prediction Time
pythia_student_pred_time_on_pythia = []
print('P/P Student Prediction Time:')
for i in range(6):
    pred_time_callback = PredictionTimeHistory()
    predictions = efn_pythia_student.predict(X_pythia_test, batch_size=1000, verbose=1, callbacks=[pred_time_callback])
    print(pred_time_callback.times)
    if i>0:
        pythia_student_pred_time_on_pythia.append(pred_time_callback.times)
    i=i+1
pythia_student_pred_time_on_pythia = np.array(pythia_student_pred_time_on_pythia)
PP_student_avg_pred_time = np.mean(pythia_student_pred_time_on_pythia)
print()
print('Pythia/Pythia Student EFN AUC:', auc_pythia_student_pythia)
print()


# get Pythia student predictions on herwig test data and ROC curve
preds_pythia_student_herwig = efn_pythia_student.predict(X_herwig_test,
                                    batch_size=1000)
efn_fp_pythia_student_herwig, dnn_tp_pythia_student_herwig, threshs_pythia_student_herwig = roc_curve(Y_herwig_test[:,1], preds_pythia_student_herwig[:,1])
auc_pythia_student_herwig = roc_auc_score(Y_herwig_test[:,1], preds_pythia_student_herwig[:,1])
# Get Prediction Time
pythia_student_pred_time_on_herwig = []
print('P/H Student Prediction Time:')
for i in range(6):
    pred_time_callback = PredictionTimeHistory()
    predictions = efn_pythia_student.predict(X_herwig_test, batch_size=1000, verbose=1, callbacks=[pred_time_callback])
    print(pred_time_callback.times)
    if i>0:
        pythia_student_pred_time_on_herwig.append(pred_time_callback.times)
    i=i+1
pythia_student_pred_time_on_herwig = np.array(pythia_student_pred_time_on_herwig)
PH_student_avg_pred_time = np.mean(pythia_student_pred_time_on_herwig)
print()
print('Pythia/Herwig Student EFN AUC:', auc_pythia_student_herwig)
print()


### Pythia Student Pareto ###
with open(f'/users/yzhou276/work/toptag/student/efn/auc/best_pythia_efn_student_latent{args.latentSizeStudent}_phi{args.phiSizesStudent}.txt', 'w') as f:
    f.write(f'P8A {auc_pythia_student_pythia}\n')
    f.write(f'H7A {auc_pythia_student_herwig}\n')
    f.write(f'UNC {np.abs(auc_pythia_student_pythia-auc_pythia_student_herwig)/auc_pythia_student_pythia}\n')
    f.write(f'P8A Pred Time {PP_student_avg_pred_time}\n')
    f.write(f'H7A Pred Time {PH_student_avg_pred_time}\n')
    f.write(f'GPU {gpu}\n')
    f.write(f'Trained by best Pythia {Phi_sizes_teacher} {F_sizes_teacher} pfn')


# get Herwig student predictions on herwig test data and ROC curve
preds_herwig_student_herwig = efn_herwig_student.predict(X_herwig_test,
                                    batch_size=1000)
efn_fp_herwig_student_herwig, dnn_tp_herwig_student_herwig, threshs_herwig_student_herwig = roc_curve(Y_herwig_test[:,1], preds_herwig_student_herwig[:,1])
auc_herwig_student_herwig = roc_auc_score(Y_herwig_test[:,1], preds_herwig_student_herwig[:,1])
# Get Prediction Time
herwig_student_pred_time_on_herwig = []
print('H/H Student Prediction Time:')
for i in range(6):
    pred_time_callback = PredictionTimeHistory()
    predictions = efn_herwig_student.predict(X_herwig_test, batch_size=1000, verbose=1, callbacks=[pred_time_callback])
    print(pred_time_callback.times)
    if i>0:
        herwig_student_pred_time_on_herwig.append(pred_time_callback.times)
    i=i+1
herwig_student_pred_time_on_herwig = np.array(herwig_student_pred_time_on_herwig)
HH_student_avg_pred_time = np.mean(herwig_student_pred_time_on_herwig)
print()
print('Herwig/Herwig Student EFN AUC:', auc_herwig_student_herwig)
print()


# get Herwig student predictions on pythia test data and ROC curve
preds_herwig_student_pythia = efn_herwig_student.predict(X_pythia_test,
                                    batch_size=1000)
efn_fp_herwig_student_pythia, dnn_tp_herwig_student_pythia, threshs_herwig_student_pythia = roc_curve(Y_pythia_test[:,1], preds_herwig_student_pythia[:,1])
auc_herwig_student_pythia = roc_auc_score(Y_pythia_test[:,1], preds_herwig_student_pythia[:,1])
# Get Prediction Time
herwig_student_pred_time_on_pythia = []
print('H/P Student Prediction Time:')
for i in range(6):
    pred_time_callback = PredictionTimeHistory()
    predictions = efn_herwig_student.predict(X_pythia_test, batch_size=1000, verbose=1, callbacks=[pred_time_callback])
    print(pred_time_callback.times)
    if i>0:
        herwig_student_pred_time_on_pythia.append(pred_time_callback.times)
    i=i+1
herwig_student_pred_time_on_pythia = np.array(herwig_student_pred_time_on_pythia)
HP_student_avg_pred_time = np.mean(herwig_student_pred_time_on_pythia)
print()
print('Herwig/Pythia Student EFN AUC:', auc_herwig_student_pythia)
print()


### Herwig Student Pareto ###
with open(f'/users/yzhou276/work/toptag/student/efn/auc/best_herwig_efn_student_latent{args.latentSizeStudent}_phi{args.phiSizesStudent}.txt', 'w') as f:
    f.write(f'P8A {auc_herwig_student_pythia}\n')
    f.write(f'H7A {auc_herwig_student_herwig}\n')
    f.write(f'UNC {np.abs(auc_herwig_student_pythia-auc_herwig_student_herwig)/auc_herwig_student_pythia}\n')
    f.write(f'P8A Pred Time {HP_student_avg_pred_time}\n')
    f.write(f'H7A Pred Time {HH_student_avg_pred_time}\n')
    f.write(f'GPU {gpu}\n')
    f.write(f'Trained by best Herwig {Phi_sizes_teacher} {F_sizes_teacher} pfn')
