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

    #### Prepare_Data ####

    def prep_result(self):
        for name in self.names:
            self.result[name] = []
            for index, sc_data in enumerate(self.adata[name].X):
                self.result[name].append([self.name_of(self.adata[name], index), self.to_matrix(sc_data)])

    def transform(self, mat):
        exp = np.dstack(mat).transpose()
        return np.expand_dims(exp, axis=1)

    def one_hot(self, len, num):
        one_hot = [0] * len
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
        labels, data, exp_values = prep_data(in_labels[:, 0][train_size:], in_data[train_size:], test_size,
                                             one_hot(num_classes, index), shuffle_data=False)
        label_tst = labels
        x_tst = np.array(data)
        y_tst = exp_values
        return (label_tr, x_tr, y_tr, label_tst, x_tst, y_tst)
