# -*- coding: utf-8 -*-

import os
import scanpy as sc

' initialize the raw sequence data '

__author__ = 'Tansel, Zhao'


class Initialization(object):

    def __init__(self, address, x=1):

        self.files_list = []
        self.get_files(address, self.files_list)
        self.names = []
        self.adata = {}  # v0.0.3 only can read the mtx.
        # To achieve multi_file and multi_times input, adata can be moved out of the __init__
        # and add more read functions or rewrite the current read function.
        self.skymapper_read_mtx(self.files_list)

        self.x = x

    def get_files(self, path, files_list):
        if not os.path.isdir(path):
            raise RuntimeError('please enter the path of a directory')
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                self.get_files(file_path, files_list)
            else:
                files_list.append(file_path)

    def skymapper_read_mtx(self, files_list):
        for path in files_list:
            if os.path.splitext(path)[-1] == '.mtx':
                # name = re.split('/', os.path.splitext(path)[0])[]      # can set up a rule to regular the name input
                name = path
                self.names.append(name)
                self.adata[name] = sc.read_10x_mtx(os.path.dirname(path),
                                                   var_names='gene_symbols',
                                                   make_unique=True,    # make unique here.
                                                   cache=True)
