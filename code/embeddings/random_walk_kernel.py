"""
Random walk kernel.

This module provides the function compute_kernel_mat for the
computation of the corresponding kernel matrix.
"""

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-03-28"


import inspect
import networkx as nx
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


def vec(M):
    return M.reshape((M.shape[0] * M.shape[1], 1))
    

def inv_vec(M, m, n):
    return M.reshape((m, n))
    
    
def mat_vec_product(x, A_i, A_j, lambda_):
    """
    Calculate the matrix-vector product (I - lambda_ * A_x) * x, where A_x
    is the adjacency matrix of the direct product graph of G_i and G_j.
    """
    y = vec(A_i.dot(inv_vec(x, A_i.shape[0], A_j.shape[0])).dot(A_j))
    
    return x - lambda_ * y
    

def compute_kernel_mat(graph_meta_data_of_num, param_range = [None]):
    kernel_mat_comp_start_time = time.time()
    
    kernel_mat_comp_time_of_param = {}
    kernel_mat_of_param = {}    
    
    
    num_graphs = len(graph_meta_data_of_num)
    
    kernel_mat = np.zeros((num_graphs, num_graphs), dtype = np.float64)
    
    # decaying factor LAMBDA for down_weighting longer walks
    LAMBDA = -4

    #=============================================================================
    # 1) precompute the (sparse) adjacency matrices of the graphs in the dataset
    #=============================================================================
    adj_mats = []
    
    
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
            
            # apply preconditioned conjugate gradient method in order to solve
            # (I - lambda_*A_x) * x = 1_vec, where A_x is the adjacency matrix of
            # the direct product graph of G_i and G_j, I is the identity matrix
            # and 1_vec is vector with all entries set to 1.
            b = np.ones((A_i.shape[0] * A_j.shape[0], 1))
            
            x, flag, rel_res, iter_, res_vec \
                = pcg.pcg(lambda x: mat_vec_product(x, A_i, A_j, LAMBDA), b, 1e-6,
                          20)
            
            kernel_mat[i,j] = np.sum(x)
            if i != j:
                kernel_mat[j, i] = kernel_mat[i, j]
            
            print 'i =', i, 'j =', j, kernel_mat[i, j]


    kernel_mat_of_param[None] = kernel_mat
    
    kernel_mat_comp_end_time = time.time()
    kernel_mat_comp_time_of_param[None] = kernel_mat_comp_end_time \
                                          - kernel_mat_comp_start_time

    return kernel_mat_of_param, kernel_mat_comp_time_of_param

