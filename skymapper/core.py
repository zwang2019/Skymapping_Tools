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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import rcParams
from sklearn.utils import shuffle
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, AveragePooling2D, BatchNormalization, Flatten, Dense, Input, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
# from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.python.client import device_lib

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import skymapper.skymapper_trainer as st



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
        labels, data, exp_values = self.prep_data(in_labels[:, 0], in_data, self.train_size, self.one_hot(num_classes, label_index),
                                                  shuffle_data=True)
        label_tr = labels
        x_tr = data
        y_tr = exp_values
        labels, data, exp_values = self.prep_data(in_labels[:, 0][self.train_size:], in_data[self.train_size:], self.test_size,
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
                self.LX, self.X, self.y, self.LX_test, self.X_test, self.Y_test = self.split_train_test(data_for_name, matrices, 1)
            else:
                self.t_LX, self.t_X, self.t_y, self.t_LX_test, self.t_X_test, self.t_Y_test = self.split_train_test(data_for_name, matrices, 1)
                self.LX = np.append(self.LX, self.t_LX, axis=0)
                self.X = np.append(self.X, self.t_X, axis=0)
                self.y = np.append(self.y, self.t_y, axis=0)
                self.LX_test = np.append(self.LX_test, self.t_LX_test, axis=0)
                self.X_test = np.append(self.X_test, self.t_X_test, axis=0)
                self.Y_test = np.append(self.Y_test, self.t_Y_test, axis=0)
            print('lx,x,y, lxt, x_test y_test shapes=', self.LX.shape, self.X.shape, self.y.shape, len(self.LX_test), self.X_test.shape,
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

    # return X, X_test, X_validation, y, Y_test, Y_validation

    def train(self, X_train, X_test, X_validation, Y_train, Y_test, Y_validation, version=2, depth=20, epochs=60):

        strategy = tf.distribute.MirroredStrategy()
        if self.choose_gpu is not []:
            # devices:
            st.get_gpu(self.choose_gpu)
            print(st._get_available_devices())
            print('Number of devices: {}'.format(strategy.num_replicas_in_sync))   # may have problem

        gpus = len(self.choose_gpu)

        tensorboard = TensorBoard(log_dir="log-full/{}".format(time.time()))

        # Training parameters
        if gpus == 0:
            batch_size = 32
        else:
            batch_size = 32 * gpus  # multiply by number of GPUs
        data_augmentation = False

        # Subtracting pixel mean improves accuracy
        subtract_pixel_mean = False

        # Model name, depth and version
        model_type = 'UResNet%dv%d' % (depth, version)
        print('model_type=', model_type)
        # Input image dimensions.

        input_shape = X_train.shape[1:]
        print('input_shape=', input_shape)

        # Normalize data.
        x_train = X_train.astype('float32') / 173
        x_test = X_test.astype('float32') / 173
        x_validation = X_validation.astype('float32') / 173

        # If subtract pixel mean is enabled
        if subtract_pixel_mean:
            x_train_mean = np.mean(x_train, axis=0)
            print(x_train_mean)
            x_train -= x_train_mean
            x_test -= x_train_mean
            x_validation -= x_train_mean

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        print(x_validation.shape[0], 'validation samples')
        print('y_train shape:', Y_train.shape)

        # Convert class vectors to binary class matrices.
        y_train = Y_train  # keras.utils.to_categorical(Y_train, num_classes)
        y_test = Y_test  # keras.utils.to_categorical(Y_test, num_classes)
        y_validation = Y_validation

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
                        model = resnet_v2(input_shape=input_shape, depth=depth)
                    else:
                        model = resnet_v1(input_shape=input_shape, depth=depth)

                    model.summary()

                    model.compile(loss='categorical_crossentropy',
                                  optimizer=Adam(lr=lr_schedule(0)),
                                  metrics=['accuracy'])

            else:
                if version == 2:
                    model = resnet_v2(input_shape=input_shape, depth=depth)
                else:
                    model = resnet_v1(input_shape=input_shape, depth=depth)

                model.summary()

                model.compile(loss='categorical_crossentropy',
                              optimizer=Adam(lr=lr_schedule(0)),
                              metrics=['accuracy'])

            print(model_type)

            # Prepare model model saving directory.
            save_dir = os.path.join(os.getcwd(), 'saved_models')
            model_name = run + '_newdata_sc_%s_model.{epoch:03d}.h5' % model_type
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            filepath = os.path.join(save_dir, model_name)

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
            if not data_augmentation:
                print('Not using data augmentation.')
                history = model.fit(x_train, y_train,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    #              steps_per_epoch=batch_size,
                                    validation_data=(x_test, y_test),
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
                datagen.fit(x_train)

                # Fit the model on the batches generated by datagen.flow().
                history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                                    validation_data=(x_test, y_test),
                                    epochs=epochs, verbose=1, workers=4,
                                    steps_per_epoch=batch_size,
                                    callbacks=callbacks)
        except Exception as e:
            print('Error:', str(e))
            let_tansel_know("Training error=" + str(e)[0:1300])
        let_tansel_know("Training finished")
















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