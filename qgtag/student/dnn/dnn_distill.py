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
from energyflow.archs.dnn import DNN
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, remap_pids, to_categorical

# Plotting imports
import matplotlib.pyplot as plt

import time

###########

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

    def train_step(
        self, data, y_pred=None, sample_weight=None, allow_empty=False
    ):
        x, y = data
        # Forward pass of teacher
        #teacher_predictions = self.teacher.model(x, training=False)
        teacher_predictions = self.teacher(x, training=False)
        #student_loss = self.student_loss_fn(y, y_pred)

        #tf.print(y_pred)
        
        with tf.GradientTape() as tape:
            # Forward pass of student

            student_predictions = self.student.model(x.reshape(-1,X_train.shape[1]*X_train.shape[2]),
                                                     training=True)

            # Compute distillation loss
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )

            # Compute student loss
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute total loss
            combined_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
            
        """
        # MLB Note: This kind of combined-loss seems to break things at the moment...
        # just using distillation loss for now, but it works well!

        #student_loss  = self.student_loss_fn(y, y_pred) # Also compute the loss of training the student directly

        # alpha determines how much the student listens to the teacher or trusts itself
        #combined_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        """

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(combined_loss, trainable_vars)
        #gradients = tape.gradient(distillation_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Report progress
        self.loss_tracker.update_state(combined_loss)
        #self.loss_tracker.update_state(distillation_loss)
        return{"combined_loss": self.loss_tracker.result()}
        #return {"distillation_loss": self.loss_tracker.result()}
        #return loss

    def call(self, x):
        #return self.student(x.reshape(-1,X_train.shape[1]*X_train.shape[2]))
        return self.student.model(x.reshape(-1,X_train.shape[1]*X_train.shape[2]))


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
parser.add_argument("-ModelNum", dest='ModelNum', default=0, type=int, required=False,
                    help="label each model")
parser.add_argument("-alpha",dest='alpha', default=0.0, type=float, required=False,
                    help="How much contribution do you want from student training loss of true labels?")
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
model_num=args.ModelNum
alpha_value = args.alpha
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

print('Pythia Shape:',X_pythia.shape)
print('Herwig Shape:',X_herwig.shape)

# Load Teacher Models
model_save_path = '/users/yzhou276/work/qgtag/simple/pfn/model/'

# Load pythia pfn teacher model
pfn_teacher_pythia = load_model(model_save_path+f'best_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_pythia_{model_num}.keras', safe_mode=False)

# Load herwig pfn teacher model
pfn_teacher_herwig = load_model(model_save_path+f'best_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_herwig_{model_num}.keras', safe_mode=False)


############################################

# build architecture
# dense_sizes above
dnn_pythia_student = DNN(input_dim=X_pythia_train.shape[1]*X_pythia_train.shape[2], dense_sizes=dense_sizes)

############################################

# train the pythia student model
X_train = X_pythia_train
distiller_pythia = Distiller(student=dnn_pythia_student, teacher=pfn_teacher_pythia)

distiller_pythia.compile(
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.CategoricalCrossentropy()],
    student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=alpha_value, # was 0.1 but doesn't do anything right now
    temperature=3.0,)

print("Training pythia student:")

if(args.doEarlyStopping):
    #es_d = EarlyStopping(monitor='val_distillation_loss', mode='min', verbose=1, patience=args.patience)
    es_d = EarlyStopping(monitor='val_categorical_crossentropy', mode='auto', verbose=1, patience=args.patience)
    '''
    mc_d = ModelCheckpoint(filepath = f'/users/yzhou276/work/qgtag/student/dnn/model/student_{dense_sizes}_dnn_by_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_pythia_{alpha_value}alpha_{model_num}.keras',
                           monitor='val_categorical_crossentropy',
                           mode='auto',
                           verbose=1,
                           save_best_only=True,
                           save_format="tf")
    '''
    mc_d = CustomModelCheckpoint(filepath = f'/users/yzhou276/work/qgtag/student/dnn/model/student_{dense_sizes}_dnn_by_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_pythia_{alpha_value}alpha_{model_num}.keras',
                                 monitor='val_categorical_crossentropy',
                                 mode='auto',
                                 verbose=1,
                                 save_best_only=True,
                                 save_format='tf'
                                 )

    #ms_d = ModelSnapshot(patience = args.patience)

    #print("Training student:")
    hist = distiller_pythia.fit(X_pythia_train,
                  Y_pythia_train,#_for_distiller,
                  epochs=num_epoch,
                  batch_size=batch_size,
                  validation_data=(X_pythia_val, Y_pythia_val),#_for_distiller),
                  verbose=1,
                  callbacks=[es_d, mc_d]) #ms_d])

    #best_weights = ms_d.models[0]
    #dnn_pythia_student.set_weights(best_weights)
    #dnn_pythia_student.save(f'/users/yzhou276/work/qgtag/student/dnn/model/student_{dense_sizes}_dnn_by_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_pythia_{alpha_value}alpha_{model_num}.keras')

else:
    distiller_pythia.fit(X_pythia_train,
              Y_pythia_train,#,_for_distiller,
              epochs=20, #num_epoch,
              batch_size=batch_size,
              validation_data=(X_pythia_val, Y_pythia_val),#_for_distiller),
              verbose=1)
    dnn_pythia_student.save(f'/users/yzhou276/work/qgtag/student/dnn/model/student_{dense_sizes}_dnn_by_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_pythia_{alpha_value}alpha_{model_num}.keras')

#########################################################################

# build architecture
# dense_sizes above
dnn_herwig_student = DNN(input_dim=X_herwig_train.shape[1]*X_herwig_train.shape[2], dense_sizes=dense_sizes)

############################################

# train the herwig student model
X_train = X_herwig_train
distiller_herwig = Distiller(student=dnn_herwig_student, teacher=pfn_teacher_herwig)
distiller_herwig.compile(
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.CategoricalCrossentropy()],
    student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=alpha_value, # was 0.1 but doesn't do anything right now
    temperature=3.0,)

print("Training herwig student:")

if(args.doEarlyStopping):
    #es_d = EarlyStopping(monitor='val_distillation_loss', mode='min', verbose=1, patience=args.patience)
    es_d = EarlyStopping(monitor='val_categorical_crossentropy', mode='auto', verbose=1, patience=args.patience)
    '''
    mc_d = ModelCheckpoint(filepath = f'/users/yzhou276/work/qgtag/student/dnn/model/student_{dense_sizes}_dnn_by_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_herwig_{alpha_value}alpha_{model_num}.keras',
                           monitor='val_categorical_crossentropy',
                           mode='auto',
                           verbose=1,
                           save_best_only=True,
                           save_format="tf")
    '''
    mc_d = CustomModelCheckpoint(filepath = f'/users/yzhou276/work/qgtag/student/dnn/model/student_{dense_sizes}_dnn_by_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_herwig_{alpha_value}alpha_{model_num}.keras',
                                 monitor='val_categorical_crossentropy',
                                 mode='auto',
                                 verbose=1,
                                 save_best_only=True,
                                 save_format='tf'
                                 )

    #ms_d = ModelSnapshot(patience = args.patience)

    #print("Training student:")
    hist = distiller_herwig.fit(X_herwig_train,
                  Y_herwig_train,#_for_distiller,
                  epochs=num_epoch,
                  batch_size=batch_size,
                  validation_data=(X_herwig_val, Y_herwig_val),#_for_distiller),
                  verbose=1,
                  callbacks=[es_d,mc_d]) #ms_d])
    
    #best_weights = ms_d.models[0]
    #dnn_herwig_student.set_weights(best_weights)
    #dnn_herwig_student.save(f'/users/yzhou276/work/qgtag/student/dnn/model/student_{dense_sizes}_dnn_by_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_herwig_{alpha_value}alpha_{model_num}.keras')
    
else:
    distiller_herwig.fit(X_herwig_train,
                     Y_herwig_train,#,_for_distiller,
                     epochs=20, #num_epoch,
                     batch_size=batch_size,
                     validation_data=(X_herwig_val, Y_herwig_val),#_for_distiller),
                     verbose=1)
    dnn_herwig_student.save(f'/users/yzhou276/work/qgtag/student/dnn/model/student_{dense_sizes}_dnn_by_{Phi_sizes_teacher}_{F_sizes_teacher}_pfn_herwig_{alpha_value}alpha_{model_num}.keras')

#########################################################################

gpu = print_gpu_info()

# get Pythia student predictions on pythia test data and ROC curve
preds_pythia_student_pythia = dnn_pythia_student.predict(X_pythia_test.reshape(-1,X_pythia_val.shape[1]*X_pythia_val.shape[2]), batch_size=1000)
dnn_fp_pythia_student_pythia, dnn_tp_pythia_student_pythia, threshs_pythia_student_pythia = roc_curve(Y_pythia_test[:,1], preds_pythia_student_pythia[:,1])
auc_pythia_student_pythia  = roc_auc_score(Y_pythia_test[:,1], preds_pythia_student_pythia[:,1])
# Get Prediction Time
pythia_student_pred_time_on_pythia = []
print('P/P Student Prediction Time:')
for i in range(6):
    pred_time_callback = PredictionTimeHistory()
    predictions = dnn_pythia_student.predict(X_pythia_test.reshape(-1,X_pythia_val.shape[1]*X_pythia_val.shape[2]), batch_size=1000, verbose=1, callbacks=[pred_time_callback])
    #print(pred_time_callback.times)
    if i>0:
        pythia_student_pred_time_on_pythia.append(pred_time_callback.times)
    i=i+1
pythia_student_pred_time_on_pythia = np.array(pythia_student_pred_time_on_pythia)
PP_student_avg_pred_time = np.mean(pythia_student_pred_time_on_pythia)
print()
print('Pythia/Pythia Student DNN AUC:', auc_pythia_student_pythia)
print()

plt.figure()
plt.rcParams['figure.figsize'] = (4,4)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.autolayout'] = True
#plt.plot(dnn_tp_pythia_student_pythia, 1-dnn_fp_pythia_student_pythia, '-', label=f'DNN AUC:{auc_pythia_student_pythia}')
plt.plot(dnn_tp_pythia_student_pythia, 1/dnn_fp_pythia_student_pythia, '-', label=f'DNN AUC:{auc_pythia_student_pythia}')
plt.xlabel('Quark Jet Efficiency')
plt.ylabel('Gluon Jet Rejection')
plt.xlim(0, 1)
#plt.ylim(0, 1)
plt.legend()#loc='lower left', frameon=False)
plt.savefig(f'/users/yzhou276/work/qgtag/student/dnn/roc_curves/best_dnn_student_nlayers{nLayers}_dense{layerSize}_{alpha_value}alpha_{model_num}_PP_ROC.jpg')


# get Pythia student predictions on herwig test data and ROC curve
preds_pythia_student_herwig = dnn_pythia_student.predict(X_herwig_test.reshape(-1,X_herwig_val.shape[1]*X_herwig_val.shape[2]), batch_size=1000)
dnn_fp_pythia_student_herwig, dnn_tp_pythia_student_herwig, threshs_pythia_student_herwig = roc_curve(Y_herwig_test[:,1], preds_pythia_student_herwig[:,1])
auc_pythia_student_herwig  = roc_auc_score(Y_herwig_test[:,1], preds_pythia_student_herwig[:,1])
# Get Prediction Time
pythia_student_pred_time_on_herwig = []
print('P/H Student Prediction Time:')
for i in range(6):
    pred_time_callback = PredictionTimeHistory()
    predictions = dnn_pythia_student.predict(X_herwig_test.reshape(-1,X_herwig_val.shape[1]*X_herwig_val.shape[2]), batch_size=1000, verbose=1, callbacks=[pred_time_callback])
    #print(pred_time_callback.times)
    if i>0:
        pythia_student_pred_time_on_herwig.append(pred_time_callback.times)
    i=i+1
pythia_student_pred_time_on_herwig = np.array(pythia_student_pred_time_on_herwig)
PH_student_avg_pred_time = np.mean(pythia_student_pred_time_on_herwig)
print()
print('Pythia/Herwig Student DNN AUC:', auc_pythia_student_herwig)
print()

plt.figure()
plt.rcParams['figure.figsize'] = (4,4)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.autolayout'] = True
#plt.plot(dnn_tp_pythia_student_herwig, 1-dnn_fp_pythia_student_herwig, '-', label=f'DNN AUC:{auc_pythia_student_herwig}')
plt.plot(dnn_tp_pythia_student_herwig, 1/dnn_fp_pythia_student_herwig, '-', label=f'DNN AUC:{auc_pythia_student_herwig}')
plt.xlabel('Quark Jet Efficiency')
plt.ylabel('Gluon Jet Rejection')
plt.xlim(0, 1)
#plt.ylim(0, 1)
plt.legend()#loc='lower left', frameon=False)
plt.savefig(f'/users/yzhou276/work/qgtag/student/dnn/roc_curves/best_dnn_student_nlayers{nLayers}_dense{layerSize}_{alpha_value}alpha_{model_num}_PH_ROC.jpg')


### Pythia Student Pareto ###
with open(f'/users/yzhou276/work/qgtag/student/dnn/auc/best_pythia_dnn_student_nlayers{nLayers}_dense{layerSize}_{alpha_value}alpha_{model_num}.txt', 'w') as f:
    f.write(f'P8A {auc_pythia_student_pythia}\n')
    f.write(f'H7A {auc_pythia_student_herwig}\n')
    f.write(f'UNC {np.abs(auc_pythia_student_pythia-auc_pythia_student_herwig)/auc_pythia_student_pythia}\n')
    f.write(f'P8A Pred Time {PP_student_avg_pred_time}\n')
    f.write(f'H7A Pred Time {PH_student_avg_pred_time}\n')
    f.write(f'GPU {gpu}\n')
    f.write(f'Trained by best Pythia {Phi_sizes_teacher} {F_sizes_teacher} pfn')


# get Herwig student predictions on herwig test data and ROC curve
preds_herwig_student_herwig = dnn_herwig_student.predict(X_herwig_test.reshape(-1,X_herwig_val.shape[1]*X_herwig_val.shape[2]), batch_size=1000)
dnn_fp_herwig_student_herwig, dnn_tp_herwig_student_herwig, threshs_herwig_student_herwig = roc_curve(Y_herwig_test[:,1], preds_herwig_student_herwig[:,1])
auc_herwig_student_herwig  = roc_auc_score(Y_herwig_test[:,1], preds_herwig_student_herwig[:,1])
# Get Prediction Time
herwig_student_pred_time_on_herwig = []
print('H/H Student Prediction Time:')
for i in range(6):
    pred_time_callback = PredictionTimeHistory()
    predictions = dnn_herwig_student.predict(X_herwig_test.reshape(-1,X_herwig_val.shape[1]*X_herwig_val.shape[2]), batch_size=1000, verbose=1, callbacks=[pred_time_callback])
    #print(pred_time_callback.times)
    if i>0:
        herwig_student_pred_time_on_herwig.append(pred_time_callback.times)
    i=i+1
herwig_student_pred_time_on_herwig = np.array(herwig_student_pred_time_on_herwig)
HH_student_avg_pred_time = np.mean(herwig_student_pred_time_on_herwig)
print()
print()#'Herwig/Herwig Student DNN AUC:', auc_herwig_student_herwig)
print()

plt.figure()
plt.rcParams['figure.figsize'] = (4,4)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.autolayout'] = True
#plt.plot(dnn_tp_herwig_student_herwig, 1-dnn_fp_herwig_student_herwig, '-', label=f'DNN AUC:{auc_herwig_student_herwig}')
plt.plot(dnn_tp_herwig_student_herwig, 1/dnn_fp_herwig_student_herwig, '-', label=f'DNN AUC:{auc_herwig_student_herwig}')
plt.xlabel('Quark Jet Efficiency')
plt.ylabel('Gluon Jet Rejection')
plt.xlim(0, 1)
#plt.ylim(0, 1)
plt.legend()#loc='lower left', frameon=False)
plt.savefig(f'/users/yzhou276/work/qgtag/student/dnn/roc_curves/best_dnn_student_nlayers{nLayers}_dense{layerSize}_{alpha_value}alpha_{model_num}_HH_ROC.jpg')


# get Herwig student predictions on pythia test data and ROC curve
preds_herwig_student_pythia = dnn_herwig_student.predict(X_pythia_test.reshape(-1,X_pythia_val.shape[1]*X_pythia_val.shape[2]), batch_size=1000)
dnn_fp_herwig_student_pythia, dnn_tp_herwig_student_pythia, threshs_herwig_student_pythia = roc_curve(Y_pythia_test[:,1], preds_herwig_student_pythia[:,1])
auc_herwig_student_pythia  = roc_auc_score(Y_pythia_test[:,1], preds_herwig_student_pythia[:,1])
# Get Prediction Time
herwig_student_pred_time_on_pythia = []
print('H/P Student Prediction Time:')
for i in range(6):
    pred_time_callback = PredictionTimeHistory()
    predictions = dnn_herwig_student.predict(X_pythia_test.reshape(-1,X_pythia_val.shape[1]*X_pythia_val.shape[2]), batch_size=1000, verbose=1, callbacks=[pred_time_callback])
    #print(pred_time_callback.times)
    if i>0:
        herwig_student_pred_time_on_pythia.append(pred_time_callback.times)
    i=i+1
herwig_student_pred_time_on_pythia = np.array(herwig_student_pred_time_on_pythia)
HP_student_avg_pred_time = np.mean(herwig_student_pred_time_on_pythia)
print()
print('Herwig/Pythia Student DNN AUC:', auc_herwig_student_pythia)
print()

plt.figure()
plt.rcParams['figure.figsize'] = (4,4)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.autolayout'] = True
#plt.plot(dnn_tp_herwig_student_pythia, 1-dnn_fp_herwig_student_pythia, '-', label=f'DNN AUC:{auc_herwig_student_pythia}')
plt.plot(dnn_tp_herwig_student_pythia, 1/dnn_fp_herwig_student_pythia, '-', label=f'DNN AUC:{auc_herwig_student_pythia}')
plt.xlabel('Quark Jet Efficiency')
plt.ylabel('Gluon Jet Rejection')
plt.xlim(0, 1)
#plt.ylim(0, 1)
plt.legend()#loc='lower left', frameon=False)
plt.savefig(f'/users/yzhou276/work/qgtag/student/dnn/roc_curves/best_dnn_student_nlayers{nLayers}_dense{layerSize}_{alpha_value}alpha_{model_num}_HP_ROC.jpg')


### Herwig Student Pareto ###
with open(f'/users/yzhou276/work/qgtag/student/dnn/auc/best_herwig_dnn_student_nlayers{nLayers}_dense{layerSize}_{alpha_value}alpha_{model_num}.txt', 'w') as f:
    f.write(f'P8A {auc_herwig_student_pythia}\n')
    f.write(f'H7A {auc_herwig_student_herwig}\n')
    f.write(f'UNC {np.abs(auc_herwig_student_pythia-auc_herwig_student_herwig)/auc_herwig_student_pythia}\n')
    f.write(f'P8A Pred Time {HP_student_avg_pred_time}\n')
    f.write(f'H7A Pred Time {HH_student_avg_pred_time}\n')
    f.write(f'GPU {gpu}\n')
    f.write(f'Trained by best Herwig {Phi_sizes_teacher} {F_sizes_teacher} pfn')
