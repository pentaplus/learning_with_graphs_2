"""
Weisfeiler-Lehman subtree kernel.

This module provides the function extract_features for the
corresponding feature extraction.
"""
from __future__ import division

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-03-14"


import inspect
import networkx as nx
import numpy as np
import sys
import time

from collections import defaultdict
from itertools import izip
from numpy.linalg import eigvalsh
from operator import itemgetter
from os.path import abspath, dirname, join
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigs, eigsh
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackError, \
    ArpackNoConvergence


# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import utils, pz


def get_avg_nodes_count(graph_meta_data_of_num):
    node_counts = []
    for graph_path, class_lbl in graph_meta_data_of_num.itervalues():
        G = pz.load(graph_path)
        node_counts.append(G.number_of_nodes())
        
    return np.mean(node_counts)
    
    
def get_max_nodes_count(graph_meta_data_of_num):
    node_counts = []
    for graph_path, class_lbl in graph_meta_data_of_num.itervalues():
        G = pz.load(graph_path)
        node_counts.append(G.number_of_nodes())
    
    return max(node_counts)    
#    return np.mean(node_counts)
    
    
def get_node_num_degree_pairs(G):
    """
    Return pairs (node_num, degree) sorted by degree in ascending order.
    """
    node_degrees = G.degree().values()
    return sorted(enumerate(node_degrees), key = itemgetter(1))


def update_row_idxs(upd_row_idx_of_orig_row_idx, removed_node_num):
    
    upd_row_idx_of_orig_row_idx[removed_node_num] = None
    
    for node_num in xrange(removed_node_num + 1,
                           len(upd_row_idx_of_orig_row_idx)):
                               
        if upd_row_idx_of_orig_row_idx[node_num]:
            upd_row_idx_of_orig_row_idx[node_num] -= 1
                                 
    return upd_row_idx_of_orig_row_idx
    
    
def del_row_and_col_at_idx(mat, idx):
    # delete column at index idx
    col_count = mat.shape[1]
    remaining_col_idxs = range(idx) + range(idx + 1, col_count)
#    lil_matrix(csr_matrix(A)[:,remaining_col_idxs])    
    mat = mat[:,remaining_col_idxs]
    
    # delete row at index idx
    n = mat.indptr[idx + 1] - mat.indptr[idx]
    if n > 0:
        mat.data[mat.indptr[idx]:-n] = mat.data[mat.indptr[idx + 1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[idx]:-n] = mat.indices[mat.indptr[idx + 1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[idx:-1] = mat.indptr[idx + 1:]
    mat.indptr[idx:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0] - 1, mat._shape[1])
    
    return mat
    

def extract_features(graph_meta_data_of_num, node_del_fracs):
    extr_start_time = time.time()
    
    feature_mat_of_param = {}
    extr_time_of_param = {}
    
    time_to_subtract_of_param = defaultdict(int)
    mat_constr_times = []
    
    num_graphs = len(graph_meta_data_of_num)
    
    avg_nodes_count = get_avg_nodes_count(graph_meta_data_of_num)
#    max_nodes_count = get_max_nodes_count(graph_meta_data_of_num)
#    avg_nodes_count = get_max_nodes_count(graph_meta_data_of_num)

    feature_mat = np.zeros((num_graphs, int(avg_nodes_count)),
                           dtype = np.float64)
    
    submat_col_count_of_node_del_frac = {}
    for node_del_frac in node_del_fracs:
        submat_col_count_of_node_del_frac[node_del_frac] \
            = int(node_del_frac * avg_nodes_count)
            
    node_del_fracs_desc_order = sorted(node_del_fracs, reverse = True)
    
#    first_eig_val_no_conv = False
    
    conv_count = 0
    no_conv_count = 0
        
    #=============================================================================
    # 1) extract features iterating over all graphs in the dataset
    #=============================================================================
    for i, (graph_path, class_lbl) in \
            enumerate(graph_meta_data_of_num.itervalues()):
        
        # !!
#        if i % 10 == 0:
#            print i
        
        # load graph
        G = pz.load(graph_path)

#        import sys
#        sys.modules['__main__'].G = G
          
        
        # determine its adjacency matrix
#        A = utils.get_adjacency_matrix(G)
        A = nx.adj_matrix(G, weight = None).astype('d')
        
        # calculate adjacency matrix of the undirected version of G
        if nx.is_directed(G):
            A = A + A.T
            
#        import sys
#        sys.modules['__main__'].A = A      
        
        nodes_count = len(G.node)
        upd_row_idx_of_orig_row_idx = dict(izip(xrange(nodes_count),
                                                xrange(nodes_count)))
        
        # get pairs (node_num, degree) sorted by degree in ascending order                                        
        node_num_degree_pairs = get_node_num_degree_pairs(G)
        
        

        j = 0
        last_j = -1
        speed = 1
        
        while j < min(nodes_count, int(avg_nodes_count)):
#        while j < nodes_count:
            sys.stdout.write('i = ' + str(i) + ' (|V| = ' + str(nodes_count)\
                             + '), j = ' + str(j) + ': ')        
            
            inner_loop_start_time = time.time()
            
            # store largest eigenvalue of A in feature matrix
#            feature_mat[i,j] = eigvalsh(A)[-1]
            
            try:
                feature_mat[i, j] = eigsh(A, which = 'LA', k = 1,
                                          maxiter = 20*A.shape[0],
                                          return_eigenvectors = False)

                                         
                                         
#                feature_mat[i,j] = eigs(A, which = 'LR', k = 1,
#                                        maxiter = 20*A.shape[0],
#                                        return_eigenvectors = False)
                
                # algorithm converged
                print(str(feature_mat[i,j]))
                
#                if first_eig_val_no_conv:
#                    feature_mat[i, :j] = feature_mat[i, j]
#                    first_eig_val_no_conv = False
                
                if j == 0:
                    last_j = 0
                    
                conv_count += 1
            except (ArpackError, ArpackNoConvergence):
#                if j == 0:
#                    first_eig_val_no_conv = True
#                else:
                if j > 0:
                    feature_mat[i, j] = feature_mat[i, j - 1]
                print(str(feature_mat[i, j - 1]) + ' [NO CONVERGENCE]')
                                 
                no_conv_count += 1
            
            if last_j < 0:
                # no iteration with convergence so far
                if j > 0:
                    speed *= 2
            else:
                feature_mat[i, last_j + 1: j] = feature_mat[i, j]
                if abs(feature_mat[i, j] - feature_mat[i, last_j]) > 1e-5:
                    last_j = j
                    speed = 1
                else:
                    if j > 0:
                        speed *= 2
            
            # determine the node number, which corresponds to the node with
            # smallest degree, and remove the corresponding row and column of
            # the (original) adjacency matrix of G
            # !! better mathematical term
            for k in xrange(j, min(j + speed, nodes_count, int(avg_nodes_count))):
                if A.shape[0] <= 2:
                    break                
                
                node_num_smallest_deg = node_num_degree_pairs[k][0]
                
                del_idx = upd_row_idx_of_orig_row_idx[node_num_smallest_deg]        
                
                A = del_row_and_col_at_idx(A, del_idx)
                

                
                upd_row_idx_of_orig_row_idx = update_row_idxs(
                    upd_row_idx_of_orig_row_idx,
                    node_num_smallest_deg)
                    
                inner_loop_end_time = time.time()
                inner_loop_time = inner_loop_end_time - inner_loop_start_time
                    
                for node_del_frac in node_del_fracs_desc_order:
                    if k >= submat_col_count_of_node_del_frac[node_del_frac]:
                        time_to_subtract_of_param[node_del_frac] \
                            += inner_loop_time
                    else:
                        break
            
            if A.shape[0] <= 2:
                break
            
            if (j < min(nodes_count, int(avg_nodes_count)) - 1) \
                    and (j + speed) >= min(nodes_count, int(avg_nodes_count)):
                
                feature_mat[i, j + 1:] = feature_mat[i, j]
                
                
            j += speed
        
        # !!
#        import sys
#        sys.modules['__main__'].G = G
#        sys.modules['__main__'].A = A
#        sys.modules['__main__'].F = feature_mat
        
#        x = 0
#        eigvalsh(A)
#
#        for j in xrange(feature_mat.shape[1]):
#            largest_eigen_val = eigvalsh(A)[-1]
        
        
        # feature_mat is of type csr_matrix and has the following form:
        # [feature vector of the first graph,
        #  feature vector of the second graph,
        #                .
        #                .
        #  feature vector of the last graph]
#        feature_mat = csr_matrix((np.array(feature_counts), np.array(features),
#                                  np.array(feature_ptr)),
#                                  shape = (len(graph_meta_data_of_num),
#                                  len(compr_func)), dtype = np.float64)
        
    extr_end_time = time.time()
    extr_time = extr_end_time - extr_start_time
    
    mat_constr_start_time = time.time()
    
    for node_del_frac in node_del_fracs:
        mat_constr_start_time = time.time()            
        
        submat_col_count = submat_col_count_of_node_del_frac[node_del_frac]
        
        feature_mat_of_param[node_del_frac] \
            = feature_mat[:,0:submat_col_count]
    
        mat_constr_end_time = time.time()
        mat_constr_time = mat_constr_end_time - mat_constr_start_time 

        extr_time_of_param[node_del_frac] = extr_time + mat_constr_time \
            - time_to_subtract_of_param[node_del_frac] - sum(mat_constr_times)
  
        mat_constr_times.append(mat_constr_time)
            
#    x = 0
    
    print('\nConvergence ratio: %.3f\n'
          % (conv_count / (conv_count + no_conv_count)))
   
    return feature_mat_of_param, extr_time_of_param



# !!
if __name__ == '__main__':
    from misc import dataset_loader as loader
    
    DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')
    dataset = 'MUTAG'
#    dataset = 'ENZYMES'
#    dataset = 'DD'
#    dataset = 'FLASH CFG'
    
    graph_meta_data_of_num, class_lbls \
        = loader.get_graph_meta_data_and_class_lbls(dataset, DATASETS_PATH)    
    
    node_del_fracs = np.linspace(1/6, 1, 6)
    
    feature_mat_of_param, extr_time_of_param \
        = extract_features(graph_meta_data_of_num, node_del_fracs)
#                                 
#    feature_mat = feature_mat_of_param[None]                                                                
                                                                   

    
    
#A = np.delete(A, (2), axis = 0)
#A = np.delete(A, (2), axis = 1)

#try:
#    5/0
#    print('jo')
#except ZeroDivisionError:
#    print('aha')

#try:
#    5/0
#    x = [1]
#    print 'bla'
#    y = x[1]
##except (IndexError):
#except (IndexError, ZeroDivisionError):
#    print('okay')