# -*- coding: utf-8 -*-

import os
import scanpy as sc

' initialize the raw sequence data '

__author__ = 'Tansel, Zhao'


class Initialization(object):

    def __init__(self, x=1):

        self.files_list = []
        #self.get_files(address, self.files_list)
        self.names = []
        self.adata = {}  # v0.0.3 only can read the mtx.
        # To achieve multi_file and multi_times input, adata can be moved out of the __init__
        # and add more read functions or rewrite the current read function.
        #self.skymapper_read_mtx(self.files_list)
        self.master_var_index = []

        self.x = x

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
                                                   make_unique=True,    # make unique here.
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

    dim = ideal_image_size_for(len(master_var_list), 1200)
    print(dim, 'x', dim)




    def to_matrix(self, sc_data):
        a = np.array(sc_data.toarray()[0])
        a.resize(dim * dim)  # for clarity
        a = np.reshape(a, (dim, dim))
        return (a)

    % matplotlib
    inline
    from matplotlib.pyplot import figure
    title = ''
    for name in names:
        title = title + '               ' + name
    print(title)
    for row_index in range(0, 20):
        figure(num=None, figsize=(12, 5), dpi=120)
        for index, name in enumerate(names, start=1):
            sc_data = adata[name].X[row_index]
            mat = to_matrix(sc_data)
            plot_sc_matrix(mat, sub_index=index, title=name_of(adata[name], row_index))
        plt.savefig("skymap-" + data_dir + '-' + str(row_index) + ".png", format="png")
        plt.show()