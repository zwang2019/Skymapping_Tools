# -*- coding: utf-8 -*-

import os
import platform
from copy import deepcopy
import re
import csv
import time
import configparser
import math

import scanpy as sc
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import rcParams

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, AveragePooling2D, BatchNormalization, Flatten, Dense, Input, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
# from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.python.client import device_lib

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# from tensorflow.keras.utils import multi_gpu_model
# from tensorflow.keras import backend as K

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from numpy import interp
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from inspect import signature

from pycm import *
import seaborn as sn
from itertools import cycle

' initialize the raw sequence data '

__author__ = 'Tansel, Zhao'


class Initialization(object):

    def __init__(self, x=1):

        self.files_list = []
        # self.get_files(address, self.files_list)
        self.names = []
        self.adata = {}  # v0.0.3 only can read the mtx.
        # To achieve multi_file and multi_times input, adata can be moved out of the __init__
        # and add more read functions or rewrite the current read function.
        # self.skymapper_read_mtx(self.files_list)
        self.master_var_index = []
        self.dim = 0  # Actually, this is a buffer to reduce the calculation.
        self.buffer = 1200  # same method to reduce the calculation
        self.method = 'union'  # same method to reduce the calculation

        self.result = {}  # to prepare the data.
        self.train_size = 6000
        self.test_size = 4000
        self.test_val_split = 0.5

        self.choose_gpu = []
        self.run = 'CD4Subtypes-TestRun9'

        self.data_augmentation = False
        self.subtract_pixel_mean = False

        self.models_list = []

        self.x = x  # A standby viable(should not use this)

    def get_files(self, path):
        if not os.path.isdir(path):
            raise RuntimeError('please enter the path of a directory')
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                self.get_files(file_path)
            else:
                self.files_list.append(file_path)

    def skymapper_read_mtx(self, path):

        self.files_list = []
        self.get_files(path)

        for path_name in self.files_list:
            if os.path.splitext(path_name)[-1] == '.mtx':
                # name = re.split('/', os.path.splitext(path)[0])[]      # can set up a rule to regular the name input
                name = path_name
                self.names.append(name)
                self.adata[name] = sc.read_10x_mtx(os.path.dirname(name),
                                                   var_names='gene_symbols',
                                                   make_unique=True,  # make unique here.
                                                   cache=True)

    # To get a common master features list by union or intersection.
    def get_master_var_list(self, all_data, method='union'):
        all_elems = [set(i.var_names) for i in list(all_data.values())]
        mgl = all_elems[0]
        if method == 'union':
            self.master_var_index = list(mgl.union(*all_elems[1:]))
        else:
            self.master_var_index = list(mgl.intersection(*all_elems[1:]))
        self.master_var_index.sort()

    def name_of(self, adata, row_index):
        l = adata.obs.index[[row_index]]
        return (l._data[0])

    def ideal_image_size_for(self, size_of_vars, buffer):
        image_size = math.sqrt(size_of_vars + 1 + buffer)
        return int((image_size // 4) * 4)  # like it multiple of 4

    def get_dim(self, buffer=1200, method='union'):
        self.buffer = buffer
        self.method = method
        if self.dim == 0:
            self.get_master_var_list(self.adata, method)
            self.dim = self.ideal_image_size_for(len(self.master_var_index), buffer)

    def change_dim(self, buffer=1200, method='union'):

        self.dim = 0
        self.get_dim(buffer, method)

    def to_matrix(self, sc_data):

        if self.dim == 0:
            self.get_dim()  # if don't manually set up the dim, it will use the default value.
        remat = np.array(sc_data.toarray()[0])
        remat.resize(self.dim * self.dim)  # for clarity
        remat = np.reshape(remat, (self.dim, self.dim))
        return remat  # do () needed?

    def plot_sc_matrix(self, mat, sub_index=1, title=None, font=None):
        if font is None:
            font = {'family': 'serif',
                    'color': 'black',
                    'weight': 'normal',
                    'size': 9, }  # for users who don't have experience of matplot
        num_classes = len(self.names)
        plt.subplot(1, num_classes, sub_index)
        plt.imshow(mat, vmin=0, vmax=3)
        if title is not None:
            plt.title(title, fontdict=font)
        plt.axis('off')
        plt.grid(b=None)

    def make_sample_plots(self, row=20, show=False, size=(12, 5), dpi=120, font=None):
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42
        title = ''
        for name in self.names:
            title = title + '               ' + name

        for row_index in range(0, row):
            figure(num=None, figsize=size, dpi=dpi)
            for index, name in enumerate(self.names, start=1):
                sc_data = self.adata[name].X[row_index]
                mat = self.to_matrix(sc_data)
                self.plot_sc_matrix(mat, sub_index=index, title=self.name_of(self.adata[name], row_index), font=font)
            plt.savefig("skymap-" + str(row_index) + ".png", format="png")
            if show is True:
                plt.show()

    def make_sample_video(self, prefix='skymap-', suffix='.png', output_name='skymap.mp4'):
        os.system("ffmpeg -y -r 10 -f image2  -pattern_type glob -i '%s*%s' -crf 25 -s 1440x600 -pix_fmt yuv420p %s" % (
            prefix, suffix, output_name))

    def show_skymap(self, mat):
        pix_labels = self.adata[self.names[0]].var
        fig, ax = plt.subplots()
        #    plt.rcParams["figure.figsize"] = (12,12)
        im = ax.imshow(mat, interpolation='none')
        ax.format_coord = Formatter(im, self.dim, pix_labels)
        plt.axis('off')
        plt.grid(b=None)
        plt.show()

    # show_skymap(mat)

    #### Prepare_Data ####

    def prep_result(self):
        for name in self.names:
            self.result[name] = []
            for index, sc_data in enumerate(self.adata[name].X):
                self.result[name].append([self.name_of(self.adata[name], index), self.to_matrix(sc_data)])

    def transform(self, mat):
        exp = np.dstack(mat).transpose()
        return np.expand_dims(exp, axis=1)

    def one_hot(self, lens, num):
        one_hot = [0] * lens
        one_hot[num - 1] = 1
        return one_hot

    def set_train_size(self, train_size, test_size, test_val_split):
        self.train_size = train_size
        self.test_size = test_size
        self.test_val_split = test_val_split

    def prep_data(self, in_labels, raw_data, desired_size, y_value, shuffle_data=False):
        num_classes = len(self.names)
        data = deepcopy(raw_data)
        labels = deepcopy(in_labels)
        if shuffle_data:
            labels, data = shuffle(in_labels, data)
        data = data[0:desired_size]
        labels = labels[0:desired_size]
        results = np.empty((len(data), num_classes))
        results[0:len(data)] = y_value
        return labels, data, results

    def split_train_test(self, in_labels, in_data, label_index):
        num_classes = len(self.names)
        labels, data, exp_values = self.prep_data(in_labels[:, 0], in_data, self.train_size,
                                                  self.one_hot(num_classes, label_index),
                                                  shuffle_data=True)
        label_tr = labels
        x_tr = data
        y_tr = exp_values
        labels, data, exp_values = self.prep_data(in_labels[:, 0][self.train_size:], in_data[self.train_size:],
                                                  self.test_size,
                                                  self.one_hot(num_classes, label_index), shuffle_data=False)
        label_tst = labels
        x_tst = np.array(data)
        y_tst = exp_values
        return label_tr, x_tr, y_tr, label_tst, x_tst, y_tst

    def prepare(self):
        self.prep_result()
        for index, name in enumerate(self.names):
            data_for_name = np.array(self.result[name])
            matrices = self.transform(data_for_name[:, 1])
            shdmn = matrices.mean(axis=0)
            shdmn = shdmn[0, :, :]
            self.show_skymap(shdmn)
            print(name, "=", data_for_name.shape, ' ', matrices.shape)
            if index == 0:
                self.LX, self.X, self.y, self.LX_test, self.X_test, self.Y_test = self.split_train_test(data_for_name,
                                                                                                        matrices, 1)
            else:
                self.t_LX, self.t_X, self.t_y, self.t_LX_test, self.t_X_test, self.t_Y_test = self.split_train_test(
                    data_for_name, matrices, 1)
                self.LX = np.append(self.LX, self.t_LX, axis=0)
                self.X = np.append(self.X, self.t_X, axis=0)
                self.y = np.append(self.y, self.t_y, axis=0)
                self.LX_test = np.append(self.LX_test, self.t_LX_test, axis=0)
                self.X_test = np.append(self.X_test, self.t_X_test, axis=0)
                self.Y_test = np.append(self.Y_test, self.t_Y_test, axis=0)
            print('lx,x,y, lxt, x_test y_test shapes=', self.LX.shape, self.X.shape, self.y.shape, len(self.LX_test),
                  self.X_test.shape,
                  self.Y_test.shape)

        self.LX_test, self.X_test, self.Y_test = shuffle(self.LX_test, self.X_test, self.Y_test)
        split_point = int(len(self.X_test) * self.test_val_split)
        print("splitting at=", split_point)
        self.LX_validation = self.LX_test[split_point:]
        self.X_validation = self.X_test[split_point:]
        self.Y_validation = self.Y_test[split_point:]
        self.LX_test = self.LX_test[0:split_point]
        self.X_test = self.X_test[0:split_point]
        self.Y_test = self.Y_test[0:split_point]

        print('Tst LX=', len(self.LX_test))
        print('Tst X=', self.X_test.shape)
        print('Tst y=', self.Y_test.shape)
        print('Val LX=', len(self.LX_validation))
        print('Val X=', self.X_validation.shape)
        print('Val y=', self.Y_validation.shape)

        # return self.X, self.X_test, self.X_validation, self.y, self.Y_test, self.Y_validation

    def train(self, version=2, depth=20, epochs=60):

        num_classes = len(self.names)

        strategy = tf.distribute.MirroredStrategy()
        if self.choose_gpu is not []:
            # devices:
            get_gpu(self.choose_gpu)
            print(_get_available_devices())
            print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        gpus = len(self.choose_gpu)

        tensorboard = TensorBoard(log_dir="log-full/{}".format(time.time()))

        # Training parameters
        if gpus == 0:
            batch_size = 32
        else:
            batch_size = 32 * gpus  # multiply by number of GPUs

        # Model name, depth and version
        model_type = 'UResNet%dv%d' % (depth, version)
        print('model_type=', model_type)
        # Input image dimensions.

        input_shape = self.X.shape[1:]
        print('input_shape=', input_shape)

        # Normalize data.
        self.x_train = self.X.astype('float32') / 173
        self.x_test = self.X_test.astype('float32') / 173
        self.x_validation = self.X_validation.astype('float32') / 173

        # If subtract pixel mean is enabled
        if self.subtract_pixel_mean:
            x_train_mean = np.mean(self.x_train, axis=0)
            print(x_train_mean)
            self.x_train -= x_train_mean
            self.x_test -= x_train_mean
            self.x_validation -= x_train_mean

        print('x_train shape:', self.x_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')
        print(self.x_validation.shape[0], 'validation samples')
        print('y_train shape:', self.y.shape)

        # Convert class vectors to binary class matrices.
        self.y_train = self.y  # keras.utils.to_categorical(Y_train, num_classes)
        self.y_test = self.Y_test  # keras.utils.to_categorical(Y_test, num_classes)
        self.y_validation = self.Y_validation

        try:
            # if gpus > 1:
            #    with tf.device("/cpu:0"):
            #        if version == 2:
            #            model = resnet_v2(input_shape=input_shape, depth=depth)
            #        else:
            #            model = resnet_v1(input_shape=input_shape, depth=depth)
            # else:
            #    if version == 2:
            #        model = resnet_v2(input_shape=input_shape, depth=depth)
            #    else:
            #        model = resnet_v1(input_shape=input_shape, depth=depth)
            # model.summary()
            #    plot_model(model, to_file='model_plot.svg', show_shapes=True, show_layer_names=True)
            # if gpus > 1:
            #    model = multi_gpu_model(model, gpus=gpus, cpu_merge=False)

            if gpus >= 1:
                with strategy.scope():
                    if version == 2:
                        model = resnet_v2(input_shape=input_shape, depth=depth, num_classes=num_classes)
                    else:
                        model = resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)

                    model.summary()

                    model.compile(loss='categorical_crossentropy',
                                  optimizer=Adam(lr=lr_schedule(0)),
                                  metrics=['accuracy'])

            else:
                if version == 2:
                    model = resnet_v2(input_shape=input_shape, depth=depth, num_classes=num_classes)
                else:
                    model = resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)

                model.summary()

                model.compile(loss='categorical_crossentropy',
                              optimizer=Adam(lr=lr_schedule(0)),
                              metrics=['accuracy'])

            print(model_type)

            # Prepare model model saving directory.
            self.save_dir = os.path.join(os.getcwd(), 'saved_models')
            model_name = self.run + '_newdata_sc_%s_model.{epoch:03d}.h5' % model_type
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
            filepath = os.path.join(self.save_dir, model_name)

            # Prepare callbacks for model saving and for learning rate adjustment.
            checkpoint = ModelCheckpoint(filepath=filepath,
                                         monitor='val_accuracy',
                                         verbose=1,
                                         save_best_only=True)

            lr_scheduler = LearningRateScheduler(lr_schedule)

            lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                           cooldown=0,
                                           patience=5,
                                           min_lr=0.5e-6)

            callbacks = [tensorboard, checkpoint, lr_reducer, lr_scheduler]

            # Run training, with or without data augmentation.
            if not self.data_augmentation:
                print('Not using data augmentation.')
                self.history = model.fit(self.x_train, self.y_train,
                                         batch_size=batch_size,
                                         epochs=epochs,
                                         #              steps_per_epoch=batch_size,
                                         validation_data=(self.x_test, self.y_test),
                                         #              validation_steps=1,
                                         shuffle=True,
                                         callbacks=callbacks)
            else:
                print('Using real-time data augmentation.')
                # This will do preprocessing and realtime data augmentation:
                datagen = ImageDataGenerator(
                    # set input mean to 0 over the dataset
                    featurewise_center=False,
                    # set each sample mean to 0
                    samplewise_center=False,
                    # divide inputs by std of dataset
                    featurewise_std_normalization=False,
                    # divide each input by its std
                    # samplewise_s1td_normalization=False,
                    # apply ZCA whitening
                    zca_whitening=False,
                    # epsilon for ZCA whitening
                    zca_epsilon=0,
                    # randomly rotate images in the range (deg 0 to 180)
                    rotation_range=0,
                    # randomly shift images horizontally
                    width_shift_range=0.1,
                    # randomly shift images vertically
                    height_shift_range=0.1,
                    # set range for random shear
                    shear_range=0.,
                    # set range for random zoom
                    zoom_range=0.,
                    # set range for random channel shifts
                    channel_shift_range=0.,
                    # set mode for filling points outside the input boundaries
                    fill_mode='nearest',
                    # value used for fill_mode = "constant"
                    cval=0.,
                    # randomly flip images
                    horizontal_flip=False,
                    # randomly flip images
                    vertical_flip=False,
                    # set rescaling factor (applied before any other transformation)
                    rescale=None,
                    # set function that will be applied on each input
                    preprocessing_function=None,
                    # image data format, either "channels_first" or "channels_last"
                    data_format=None,
                    # fraction of images reserved for validation (strictly between 0 and 1)
                    validation_split=0.0)

                # Compute quantities required for featurewise normalization
                # (std, meadata = adata[:, adata.var['highly_variable']]adata = adata[:, adata.var['highly_variable']]an, and principal components if ZCA whitening is applied).
                datagen.fit(self.x_train)

                # Fit the model on the batches generated by datagen.flow().
                self.history = model.fit(datagen.flow(self.x_train, self.y_train, batch_size=batch_size),
                                         validation_data=(self.x_test, self.y_test),
                                         epochs=epochs, verbose=1, workers=4,
                                         steps_per_epoch=batch_size,
                                         callbacks=callbacks)
        except Exception as e:
            print('Error:', str(e))
            let_tansel_know("Training error=" + str(e)[0:1300])
        let_tansel_know("Training finished")

    def get_models(self, path):
        if not os.path.isdir(path):
            raise RuntimeError('something wrong with the save directory')
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                self.get_models(file_path)
            else:
                self.models_list.append(file_path)

    def analyze(self):

        num_classes = len(self.names)

        mpl.rcParams['figure.figsize'] = [8.0, 6.0]
        mpl.rcParams['figure.dpi'] = 80
        mpl.rcParams['savefig.dpi'] = 100

        mpl.rcParams['font.size'] = 14
        mpl.rcParams['legend.fontsize'] = 'large'
        mpl.rcParams['figure.titlesize'] = 'medium'

        # Measure accuracy
        plt.plot(self.history.history['accuracy'])  # acc or val_acc may cause problems
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig("accuracy.svg", format="svg")
        plt.show()

        # Plot training & validation loss values
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig("loss.svg", format="svg")
        plt.show()

        ## load best model ##

        self.get_models(self.save_dir)
        self.models_list = list(set(self.models_list))

        print("Load the best model %s" % self.models_list[0])
        self.model = load_model(self.models_list[0])

        # Test and also validate with an unseen sample dataset

        # Validate trained model.
        scores = self.model.evaluate(self.x_validation, self.y_validation, verbose=1)
        print('Validation loss:', scores[0])
        print('Validation accuracy:', scores[1])
        # print(scores)

        ## Rerun the tests ##

        self.y_predicted = self.model.predict(self.x_validation, verbose=1)


        ##### CONFIDENCE MATRIX #####

        # Prepare a confidence matrix for each threshold by applying various thresholds

        self.y_test_predicted = self.y_predicted
        self.y_test = self.y_validation

        for threshold in [0.9]:                   # should be 0.0, 0.1, 0.3, 0.5, 0.7, as well
            cm_with_heatmap_for_threshold(self.y_test_predicted, self.y_test, num_classes,threshold=threshold)

        # Learn to predict each class against the other

        y_score = self.y_test_predicted
        n_classes = num_classes
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(self.y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        ##############################################################################
        # Plot ROC curves for the subtypes

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Average and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        fig = plt.figure(figsize=(8, 8), dpi=120)

        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC (area = %0.2f)'
                       % roc_auc["micro"],
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC (area = %0.2f)'
                       % roc_auc["macro"],
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

        lw = 2  # line width

        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of ' + self.names[i] + ' (area = %0.2f)' % roc_auc[i])

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating curve for subclasses separation')
        plt.legend(loc="lower right")
        plt.savefig("ROC.svg", format="svg")
        plt.show()

        fig = plt.figure(figsize=(6, 6), dpi=120)
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC (area = %0.2f)'
                       % roc_auc["micro"],
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC (area = %0.2f)'
                       % roc_auc["macro"],
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of ' + self.names[i] + ' (area = %0.2f)' % roc_auc[i])
        # plt.plot([0, 0.2], [0.8, 1], 'k--', lw=lw)
        plt.xlim([0.0, 0.3])
        plt.ylim([0.7, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating curve zoomed')
        plt.legend(loc="lower right")
        plt.savefig("ROC_Zoomed.svg", format="svg")
        plt.show()


        ############

        average_precision = average_precision_score(self.y_test, y_score)
        print('Average precision-recall score: {0:0.2f}'.format(average_precision))
        # precision, recall, _ = precision_recall_curve(self.y_test, y_score)

        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(self.Y_test[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(self.Y_test[:, i], y_score[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(self.Y_test.ravel(), y_score.ravel())
        average_precision["micro"] = average_precision_score(self.Y_test, y_score, average="micro")
        print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

        # precision[0][0] + precision[1][0] + precision[2][0] + precision[3][0]

        # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
        step_kwargs = None
        # plt.step(recall, precision, color='b', alpha=0.2,
        #         where='post')
        # plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        plt.figure()
        plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
        # plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b',
        #                  **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
        plt.savefig("APS_MA.svg", format="svg")
        plt.show()

        ####################

        # setup plot details
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

        plt.figure(figsize=(7, 8))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

        lines.append(l)
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro-average Precision-recall (area = {0:0.2f})'
                      ''.format(average_precision["micro"]))

        for i, color in zip(range(n_classes), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                          ''.format(i, average_precision[i]))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve multi-class')
        plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

        plt.savefig("Precision-Recall_curve.svg", format="svg")
        plt.show()







class Formatter(object):
    def __init__(self, im, dim, pix_labels):
        self.im = im
        self.dim = dim
        self.pix_labels = pix_labels

    def __call__(self, x, y):
        label = self.label_for_pix(y, x)  # +' '+adata[names[3]].var.iloc[index,0]
        return '{}'.format(label)

    def label_for_pix(self, x, y):
        label = ''
        index = int(x) * self.dim + int(y)
        if index < len(self.pix_labels):
            label = self.pix_labels.iloc[index].name
        return label


### training fuction ###

def residual_block_unit(inputs,
                        num_filters=16,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        activation='relu',
                        batch_normalization=True,
                        conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        net (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  data_format='channels_first',
                  kernel_regularizer=l2(1e-4))

    net = inputs
    if conv_first:
        net = conv(net)
        print("Convolution name= ", net.name, " numfilters=", num_filters, " kernel_size=", kernel_size, " strides=",
              strides)
        if batch_normalization:
            net = BatchNormalization()(net)
            print("Batch normalisation")
        if activation is not None:
            net = Activation(activation)(net)
            print("Activation")
    else:
        if batch_normalization:
            net = BatchNormalization()(net)
            print("Batch normalisation")
        if activation is not None:
            net = Activation(activation)(net)
            print("Activation")
        net = conv(net)
        print("Convolution name= ", net.name, " numfilters=", num_filters, " kernel_size=", kernel_size, " strides=",
              strides)
    # conv.name = conv.name + '_' + str(kernel_size) + 'x' + str(kernel_size) + '_' + str(num_filters) + '_' + str(
    #    strides)  # can be exam later
    return net


def resnet_v1(input_shape, depth, num_classes=4):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    x = 1

    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    net = residual_block_unit(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            short_cut = residual_block_unit(inputs=net,
                                            num_filters=num_filters,
                                            strides=strides)
            short_cut = residual_block_unit(inputs=short_cut,
                                            num_filters=num_filters,
                                            activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                net = residual_block_unit(inputs=net,
                                          num_filters=num_filters,
                                          kernel_size=1,
                                          strides=strides,
                                          activation=None,
                                          batch_normalization=False)
            net = tf.keras.layers.add([net, short_cut])
            net = Activation('relu')(net)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    net = AveragePooling2D(pool_size=8)(net)
    short_cut = Flatten()(net)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(short_cut)

    # the original version is y, but may cause some problem. Have to exam in the test.

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=4):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:777
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = residual_block_unit(inputs=inputs,
                            num_filters=num_filters_in,
                            conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = residual_block_unit(inputs=x,
                                    num_filters=num_filters_in,
                                    kernel_size=1,
                                    strides=strides,
                                    activation=activation,
                                    batch_normalization=batch_normalization,
                                    conv_first=False)
            y = residual_block_unit(inputs=y,
                                    num_filters=num_filters_in,
                                    conv_first=False)
            y = residual_block_unit(inputs=y,
                                    num_filters=num_filters_out,
                                    kernel_size=1,
                                    conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = residual_block_unit(inputs=x,
                                        num_filters=num_filters_out,
                                        kernel_size=1,
                                        strides=strides,
                                        activation=None,
                                        batch_normalization=False)
            x = tf.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # why here use y instead of x ?

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 10, 20, 30, 50 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 60:
        lr *= 1e-4
    elif epoch > 55:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def config():
    print('''                                 Model parameter
         ----------------------------------------------------------------------------
                   |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
         Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
                   |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
         ----------------------------------------------------------------------------
         ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
         ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
         ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
         ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
         ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
         ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
         ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
         ---------------------------------------------------------------------------''')


def get_gpu(ggg):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    for gpu in ggg:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def _get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def let_tansel_know(str):
    print(str)


## Get some statistics ##

def spectifity(true_negatives, false_positives):
    return true_negatives / (true_negatives + false_positives)


def sensitivity(true_positives, false_negatives):
    return true_positives / (true_positives + false_negatives)


def confidence_margin(npa):
    sorted_npa = np.sort(npa)[::-1]  # return sort in reverse, i.e. descending
    return sorted_npa[0] - sorted_npa[1]


# print(confidence_margin([0.0, 1.0, 0.5, 0.0]))


def result_for(npa):
    half = len(npa) // 2
    return np.argmax(npa[0:half]) == np.argmax(npa[half:])


# print(result_for(np.array([0.0, 1.0, 0.5, 0.0, 0.8,0.1, 0.2, 0.1])))


def confidence_corr(npa):
    half = len(npa) // 2
    res = result_for(npa)
    cm = confidence_margin(npa[0:half])
    return res, cm


# this totals_for is an alternative way
'''
def totals_for(combined_results, treshold=0.5):
    tt, t_f, th, tl, fh, fl = 0, 0, 0, 0, 0, 0
    for y in y_val_pred_act:
        res, cm = confidence_corr(y)
        if res:
            tt = tt + 1
            if cm > treshold:
                th = th + 1
            else:
                tl = tl + 1
        else:
            t_f = t_f + 1
            if cm > treshold:
                fh = fh + 1
            else:
                fl = fl + 1
    return tt, t_f, th, tl, fh, fl
'''


def vectorise_with_threshold(predictions, threshold):
    vectorised = []
    for pred in predictions:
        if confidence_margin(pred) > threshold:
            vectorised.append(np.argmax(pred) + 1)
        else:
            vectorised.append(0)
    return np.array(vectorised)


def vectorise_selected_with_threshold(predictions, actuals, threshold):
    vectorised = []
    vect_act = []
    for index, pred in enumerate(predictions):
        if confidence_margin(pred) > threshold:
            vectorised.append(np.argmax(pred) + 1)
            vect_act.append(np.argmax(actuals[index]) + 1)
    return np.array(vectorised), np.array(vect_act)


def totals_for(cm_table, num_classes):
    correct, incorrect = 0, 0
    for i in range(1, num_classes + 1):
        for j in range(1, num_classes + 1):
            if i == j:
                correct = correct + cm_table[i][j]
            else:
                incorrect = incorrect + cm_table[i][j]
    total = correct + incorrect
    return total, correct, incorrect, correct / total, incorrect / total


def cm_with_heatmap_for_threshold(predicted, actual, num_classes, threshold=0.5):
    pv = vectorise_with_threshold(predicted, threshold)
    av = vectorise_with_threshold(actual, threshold)
    cm = ConfusionMatrix(actual_vector=av, predict_vector=pv)
    recognised = np.count_nonzero(pv)
    print("Total=", len(pv), " Classified=", recognised, "Discarded=", len(pv) - recognised, "recovered percent=",
          recognised / len(pv) * 100, "discarded percent=", (len(pv) - recognised) / len(pv) * 100)
    cm_table = cm.table
    print(totals_for(cm_table, num_classes))
    df_cm = pd.DataFrame(cm_table, index=range(1, num_classes + 1), columns=range(1, num_classes + 1))
    plt.figure(figsize=(5, 4))
    ax = plt.axes()
    ax.set_title("Threshold=" + str(threshold))
    sn.heatmap(df_cm, annot=True, fmt='g', ax=ax)  # ,annot_kws=annot_kws)# ,mask=mask)
    plt.savefig("cm_with_heatmap_for_threshold%d" % threshold, format="svg")
    plt.show()
