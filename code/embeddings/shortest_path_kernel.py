"""
Shortest path kernel.

This module provides the function extract_features for the
corresponding feature extraction. It is a translation of
the MATLAB file SPkernel.m by Nino Shervashidze,
which can be downloaded from the following website:
http://mlcb.is.tuebingen.mpg.de/Mitarbeiter/Nino/Graphkernels/
"""

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__credits__ = ["Nino Shervashidze"]
__date__ = "2016-04-08"


import inspect
import networkx as nx
import numpy as np
import sys
import time

from os.path import abspath, dirname, join
from scipy.sparse import lil_matrix


# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import pz


def floyd_warshall(A, sym):
    # determine number of nodes
    n = A.shape[0]
    
    # initialize distance matrix
    D = np.multiply(A, A)    
    
    # set distance between non-adjacent nodes to infinity
    D[A + np.diag(np.tile(np.inf, n)) == 0] = np.inf
    
    # set distance from each node to itself to zero
    np.fill_diagonal(D, 0)
    
    
    # iteratively update distances
    if sym:
        for k in xrange(n):
            if k % 10 == 0:
                print 'k:', k            
            
            D_aux = np.tile(D[:, k: k + 1], (1, n))
            Sum_dist = D_aux + D_aux.T
            
#            sys.modules['__main__'].D_aux = D_aux
#            sys.modules['__main__'].D = D
#            sys.modules['__main__'].Sum_dist = Sum_dist 
            
            D[Sum_dist < D] = Sum_dist[Sum_dist < D]
    else:
        for k in xrange(n):
            if k % 10 == 0:
                print 'k:', k 
            D_aux_1 = np.tile(D[:, k: k + 1], (1, n))
            D_aux_2 = np.tile(D[k, :], (n, 1))
            Sum_dist = D_aux_1 + D_aux_2
            D[Sum_dist < D] = Sum_dist[Sum_dist < D]
            
    return D
    

def extract_features(graph_meta_data_of_num, param_range = [None]):

    extr_start_time = time.time()
    
    feature_mat_of_param = {}
    extr_time_of_param = {}
    
    num_graphs = len(graph_meta_data_of_num)
    
    
    Ds = []
        
    max_shortest_path_len = 0
    
    #=============================================================================
    # 1) determine (shortest) distance matrices
    #=============================================================================
    for i, (graph_path, class_lbl) in \
            enumerate(graph_meta_data_of_num.itervalues()):
                
        # !!
#        if i % 10 == 0:
#            print i
                
        # load graph
        G = pz.load(graph_path)
        # determine its adjacency matrix
        A = nx.adj_matrix(G, weight = None).astype('d').toarray()
#        A = nx.adj_matrix(G, weight = None).toarray()
        
        print 'i:', i, 'nodes count:', G.number_of_nodes()
        
        is_symmetric = not nx.is_directed(G)
        
#        sys.modules['__main__'].A = A
        
#        adj_mats.append(A)
        
        # determine distance matrix
        D = floyd_warshall(A, is_symmetric)
        Ds.append(D)
        
#        sys.modules['__main__'].Ds = Ds

        # determine maximum distance between nodes, between which a path exists
        max_shortest_path_len_in_G = D[np.isfinite(D)].max()
        
        if max_shortest_path_len_in_G > max_shortest_path_len:
            max_shortest_path_len = max_shortest_path_len_in_G
#             # !!
##            sys.modules['__main__'].kernel_mat = kernel_mat
            
#            print 'i =', i, 'j =', j
#            print 'i =', i, 'j =', j, kernel_mat[i,j]
#    sp = lil_matrix((max_path + 1), num_graphs)
                       
            
    # initialize feature matrix
    feature_mat = lil_matrix((num_graphs, max_shortest_path_len + 1),
                             dtype = np.float64)
                             
    #=============================================================================
    # 2) determine number of occurences of each shortest path length
    #=============================================================================
    for i in xrange(num_graphs):
        D = Ds[i]
#        sys.modules['__main__'].D = D
        
        D_triu_finite_indicators = np.triu(np.isfinite(D))
        
        # shortest_path_lengths
        shortest_path_lengths = D[D_triu_finite_indicators].astype(np.int64)
        

        shortest_path_lengths_counts = np.bincount(shortest_path_lengths)
        
        feature_mat[i, shortest_path_lengths] \
            = shortest_path_lengths_counts[shortest_path_lengths]
        
#    kernel_mat = sp.T * sp


    feature_mat_of_param[None] = feature_mat.tocsr()
    
    
    extr_end_time = time.time()
    extr_time = extr_end_time - extr_start_time
    
    extr_time_of_param[None] = extr_time

    return feature_mat_of_param, extr_time_of_param



if __name__ == '__main__':
    from misc import dataset_loader
    from performance_evaluation import cross_validation
    
#    from sklearn.cross_validation import KFold
    from sklearn.svm import SVC
#    from sklearn.cross_validation import cross_val_score
    from sklearn.metrics.pairwise import pairwise_kernels
    
    DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')
#    dataset = 'MUTAG'
#    dataset = 'PTC(MR)'
#    dataset = 'FLASH CFG'
    dataset = 'DD'
    
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