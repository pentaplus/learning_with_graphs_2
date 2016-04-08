"""
Shortest path kernel.

This module provides the function compute_kernel_mat for the
computation of the corresponding kernel matrix. It is a translation of
the MATLAB file RWkernel.m by Karsten Borgwardt and Nino Shervashidze,
which can be downloaded from the following website:
http://mlcb.is.tuebingen.mpg.de/Mitarbeiter/Nino/Graphkernels/
"""

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__credits__ = ["Karsten Borgwardt", "Nino Shervashidze"]
__date__ = "2016-02-28"


import inspect
import networkx as nx
import math
import numpy as np
import sys
import time

from os.path import abspath, dirname, join
from scipy.sparse import csr_matrix, lil_matrix


# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import floyd_warshall, pz


def extract_features(graph_meta_data_of_num, param_range = [None]):
#def compute_kernel_mat(graph_meta_data_of_num, param_range = [None]):
#    kernel_mat_comp_start_time = time.time()
#    
#    kernel_mat_comp_time_of_param = {}
#    kernel_mat_of_param = {}    

    extr_start_time = time.time()
    
    feature_mat_of_param = {}
    extr_time_of_param = {}
    
    INF = 2**31 - 1
    
    num_graphs = len(graph_meta_data_of_num)
#    graph_meta_data = graph_meta_data_of_num.values()
    
#    kernel_mat = np.zeros((num_graphs, num_graphs), dtype = np.float64)
    

    #=============================================================================
    # compute kernel matrix over all graphs in the dataset
    #=============================================================================
    
    Ds = []
        
    max_path = 0
    
#    for i in xrange(num_graphs):
    for i, (graph_path, class_lbl) in \
            enumerate(graph_meta_data_of_num.itervalues()):
                
        # !!
        if i % 10 == 0:
            print i
        
        # load graph
        G = pz.load(graph_path)
        # determine its adjacency matrix
        A = nx.adj_matrix(G, weight = None).astype('d').toarray()
#        A = nx.adj_matrix(G, weight = None).toarray()
        
        is_symmetric = not nx.is_directed(G)
        
#        sys.modules['__main__'].A = A
        
#        adj_mats.append(A)
        
        D = floyd_warshall.floyd_warshall(A, is_symmetric)
        Ds.append(D)
        
#        sys.modules['__main__'].Ds = Ds

        aux = D[np.isfinite(D)].max()
        
        if aux > max_path:
            max_path = aux
#             # !!
##            sys.modules['__main__'].kernel_mat = kernel_mat
            
#            print 'i =', i, 'j =', j
#            print 'i =', i, 'j =', j, kernel_mat[i,j]
#    sp = lil_matrix((max_path + 1), num_graphs)
    feature_mat = lil_matrix((num_graphs, max_path + 1), dtype = np.float64)
    for i in xrange(num_graphs):
        D = Ds[i]
#        sys.modules['__main__'].D = D
        
        I = np.triu(np.isfinite(D))
        
        # shortest_path_lengths
        Ind = D[I].astype(np.int64)
#        sys.modules['__main__'].Ind = Ind
        
        # number of occurences of shortest_path_lengths
        aux = np.bincount(Ind)
#        sys.modules['__main__'].aux = aux
        
        feature_mat[i, Ind] = aux[Ind]
        
#    kernel_mat = sp.T * sp

#    kernel_mat_of_param[None] = kernel_mat
#    
#    kernel_mat_comp_end_time = time.time()
#    kernel_mat_comp_time_of_param[None] = kernel_mat_comp_end_time \
#                                          - kernel_mat_comp_start_time

    feature_mat_of_param[None] = feature_mat.tocsr()
    
    
    extr_end_time = time.time()
    extr_time = extr_end_time - extr_start_time
    
    extr_time_of_param[None] = extr_time

    return feature_mat_of_param, extr_time_of_param


#    
#    import networkx as nx
#    from scipy.sparse import csr_matrix
#    import scipy.io as spio
#
#    G = pz.load(graph_meta_data_of_num.values()[0][0])
#    A = nx.adjacency_matrix(G, weight = None)
#    
##    timeit A = nx.adjacency_matrix(G, weight = None) # 2.3 ms
##    timeit B = A.todense()  183 micros
#    
#    A_sprs = csr_matrix(A)
#    A_sprs
#    I = np.nonzero(A)
#    I[0]
#    
#    mat = spio.loadmat('data.mat')
#    
#    A_mat = mat['A']
#    
#    # utils: 24.0, adj_mat calc: 12.7
#
#    # load all as dense matrices: 81 sec
#    # load all as sparse matrices: 75 sec


if __name__ == '__main__':
    from misc import dataset_loader
    from performance_evaluation import cross_validation
    
#    from sklearn.cross_validation import KFold
    from sklearn.svm import SVC
#    from sklearn.cross_validation import cross_val_score
    from sklearn.metrics.pairwise import pairwise_kernels
    
    DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')
    dataset = 'MUTAG'
#    dataset = 'PTC(MR)'
#    dataset = 'FLASH CFG'
    
    graph_meta_data_of_num, class_lbls \
        = dataset_loader.get_graph_meta_data_and_class_lbls(dataset,
                                                            DATASETS_PATH)    
    
    feature_mat_of_param, extr_time_of_param =\
                                 extract_features(graph_meta_data_of_num, [None])
                                 
    feature_mat = feature_mat_of_param[None]                                                                
                                                                   


    clf = SVC(kernel = 'precomputed')

    
    
#    cross_validation.cross_val(clf, kernel_mat, class_lbls, 10, 10,
#                               open('bla.txt', 'w')) 
    

#    import scipy.io as spio
#    mat = spio.loadmat('data.mat')
#    type(mat)
#    mat.keys()
#    K_mat = mat['ans']