"""
Random walk kernel.

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


# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import pcg, pz


#def get_lambda(graph_meta_data_of_num):
#    """
#    Determine the parameter lamda in depende of the dataset size.
#    
#    The parameter lamda is determined according to the rule of thumb
#    to take the largest power of 10, which is smaller than 1/d^2,
#    d being the largest degree in the dataset.
#    """
#    max_deg = 0
#    
#    for graph_path, class_num in graph_meta_data_of_num.itervalues():
#        G = pz.load(graph_path)
#            
#        if max(G.degree().values()) > max_deg:
#            max_deg = max(G.degree().values())
#    
#    return math.floor(math.log10(1./max_deg**2))

def vec(M):
    return M.reshape((M.shape[0] * M.shape[1], 1))
    

def invvec(M, m, n):
    return M.reshape((m, n))
    
    
def smtfilter(x, A_i, A_j, lambda_):
    yy = vec(A_i.dot(invvec(x, A_i.shape[0], A_j.shape[0])).dot(A_j))
    
    yy *= lambda_
    
    vecu = x - yy
    
    return vecu
    

def compute_kernel_mat(graph_meta_data_of_num, param_range = [None]):
    kernel_mat_comp_start_time = time.time()
    
    kernel_mat_comp_time_of_param = {}
    kernel_mat_of_param = {}    
    
    
    num_graphs = len(graph_meta_data_of_num)
#    graph_meta_data = graph_meta_data_of_num.values()
    
    kernel_mat = np.zeros((num_graphs, num_graphs), dtype = np.float64)
    
    # decaying factor lambda_ for down_weighting longer walks
#    lambda_ = get_lambda(graph_meta_data_of_num)
    LAMBDA = -4

    #=============================================================================
    # 1) precompute the (sparse) adjacency matrices of the graphs in the dataset
    #=============================================================================
    adj_mats = []
    
#    for i in xrange(num_graphs):
    for i, (graph_path, class_lbl) in \
            enumerate(graph_meta_data_of_num.itervalues()):
                
        # !!
#        if i % 10 == 0:
#            print i
        
        # load graph
        G = pz.load(graph_path)
        # determine its adjacency matrix
        A = nx.adj_matrix(G, weight = None)
        
        adj_mats.append(A)
        
    
    #=============================================================================
    # 2) compute kernel matrix over all graphs in the dataset
    #=============================================================================
    for i in xrange(num_graphs):
        A_i = adj_mats[i].todense()

        for j in xrange(i, num_graphs):
            A_j = adj_mats[j].todense()
            
            # !!
#            sys.modules['__main__'].A_j = A_j
            
            # apply preconditioned conjugate gradient method
            b = np.ones((A_i.shape[0] * A_j.shape[0], 1))
            
            x, flag, relres, iter_, resvec \
                = pcg.pcg(lambda x: smtfilter(x, A_i, A_j, LAMBDA), b, 1e-6, 20)
                
            
            kernel_mat[i,j] = np.sum(x)
            if i != j:
                kernel_mat[j,i] = kernel_mat[i,j]
            
#             # !!
##            sys.modules['__main__'].kernel_mat = kernel_mat
            
#            print 'i =', i, 'j =', j
            print 'i =', i, 'j =', j, kernel_mat[i,j]

        
        
    

    kernel_mat_of_param[None] = kernel_mat
    
    kernel_mat_comp_end_time = time.time()
    kernel_mat_comp_time_of_param[None] = kernel_mat_comp_end_time \
                                          - kernel_mat_comp_start_time

    return kernel_mat_of_param, kernel_mat_comp_time_of_param



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
    
    h_range = range(6)
    
    kernel_mat_of_param, kernel_mat_comp_time_of_param =\
                  compute_kernel_mat(graph_meta_data_of_num, param_range = [None])
                                 
    kernel_mat = kernel_mat_of_param[None]                                                                
                                                                   


    clf = SVC(kernel = 'precomputed')

    
    
#    cross_validation.cross_val(clf, kernel_mat, class_lbls, 10, 10,
#                               open('bla.txt', 'w')) 
    

    import scipy.io as spio
    mat = spio.loadmat('data.mat')
    type(mat)
    mat.keys()
    K_mat = mat['ans']