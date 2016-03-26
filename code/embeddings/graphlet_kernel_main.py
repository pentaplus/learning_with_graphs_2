"""
Graphlet kernel counting graphlets of sizes 3 and 4.

This module provides the function extract_features for the corresponding
feature extraction. It is a translation of the MATLAB files
countall3graphlets.m and countall4graphlets.m by Nino Shervashidze,
which can be downloaded from the following website:
http://mlcb.is.tuebingen.mpg.de/Mitarbeiter/Nino/Graphkernels/
"""
from __future__ import division

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__credits__ = ["Nino Shervashidze"]
__date__ = "2016-02-28"


import inspect
import sys
import time

import numpy as np
from os.path import abspath, dirname, join
from scipy.misc import comb


# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import pz, utils


def calc_cards(v1_nbrs, v2_nbrs, v3_nbrs):
    return np.array([len(v1_nbrs - (v2_nbrs | v3_nbrs)),
                     len(v2_nbrs - (v1_nbrs | v3_nbrs)),
                     len(v3_nbrs - (v1_nbrs | v2_nbrs)),
                     len(v1_nbrs & v2_nbrs - v3_nbrs),
                     len(v1_nbrs & v3_nbrs - v2_nbrs),
                     len(v2_nbrs & v3_nbrs - v1_nbrs),
                     len(v1_nbrs & v2_nbrs & v3_nbrs)])
    

def extract_features(graph_meta_data_of_num, graphlet_size = 4):
    extr_start_time = time.time()
    
    feature_mat_of_param = {}
    extr_time_of_param = {}    
    
    graphlets_count = 0    
    if graphlet_size == 3:
        graphlets_count = 4
    elif graphlet_size == 4:
        graphlets_count = 11
        
    # initialize feature matrix
    graphs_count = len(graph_meta_data_of_num)
    feature_mat = np.zeros((graphs_count, graphlets_count), dtype = np.float64)
    
    
    #=============================================================================
    # extract features iterating over all graphs in the dataset
    #=============================================================================
    for i, (graph_path, class_lbl) in \
            enumerate(graph_meta_data_of_num.itervalues()):
                
        # load graph        
        G = pz.load(graph_path)
        
        nodes_count = len(G.node)
    
        if graphlet_size == 3:
            # count 3-graphlets
            # counts[i] finally holds the number of the graphlet g_(i + 1),
            # i = 0,...,3 (see Figure !!) 
            counts = np.zeros(4, np.float64)
            
            weights = np.array([6, 4, 2], np.float64) 
        
            for v1 in G.nodes_iter():
                has_elem, nbr_iter = utils.has_elem(G.neighbors_iter(v1))
                if not has_elem:
                    # node v1 has no neighbors
                    continue
                
                v1_nbrs = set(G.neighbors(v1))
                
                for v2 in v1_nbrs:
                    v2_nbrs = set(G.neighbors(v2))
                    counts[0] += len(v1_nbrs & v2_nbrs)
                    counts[1] += len(v1_nbrs - (v2_nbrs | {v2}))
                    counts[1] += len(v2_nbrs - (v1_nbrs | {v1}))
                    counts[2] += nodes_count - len(v1_nbrs | v2_nbrs)
            
            counts[:3] /= weights
            counts[3] = comb(nodes_count, 3) - sum(counts)
            
            feature_mat[i] = counts
        
        elif graphlet_size == 4:
            # count 4-graphlets
            # c[i] finally holds the number of the graphlet g_(i + 1),
            # i = 0,...,10 (see Figure !!)
            counts = np.zeros(11, np.float64)
            
            weights = np.array([1/12, 1/10, 1/8, 1/6, 1/8, 1/6, 1/6, 1/4, 1/4,
                                1/2, 0], np.float64)
            
            # each undirected edge is only counted once
            edges_count = G.number_of_edges()
        
            for v1 in G.nodes_iter():
                has_elem, nbrs_iter = utils.has_elem(G.neighbors_iter(v1))
                if not has_elem:
                    # node v1 has no neighbors
                    continue
                
                v1_nbrs = set(G.neighbors(v1))
                
                for v2 in v1_nbrs:
                    K = 0                    
                    tmp_counts = np.zeros(11, np.float64)
                    
                    v2_nbrs = set(G.neighbors(v2))
                    
                    v1_nbrs_inter_v2_nbrs = v1_nbrs & v2_nbrs
                    v1_nbrs_minus_v2_nbrs = v1_nbrs - v2_nbrs
                    v2_nbrs_minus_v1_nbrs = v2_nbrs - v1_nbrs
                    
                    
                    for v3 in v1_nbrs_inter_v2_nbrs:
                        v3_nbrs = set(G.neighbors(v3))
                        
                        cards = calc_cards(v1_nbrs, v2_nbrs, v3_nbrs)
                        
                        tmp_counts[0] += 1/2*cards[6]
                        tmp_counts[1] += 1/2*(cards[3] - 1)
                        tmp_counts[1] += 1/2*(cards[4] - 1)
                        tmp_counts[1] += 1/2*(cards[5] - 1)
                        tmp_counts[2] += 1/2*cards[0]
                        tmp_counts[2] += 1/2*cards[1]
                        tmp_counts[2] += cards[2]
                        tmp_counts[6] += nodes_count - sum(cards)
                        
                        K += 1/2*cards[6] + 1/2*(cards[4] - 1) \
                             + 1/2*(cards[5] - 1) + cards[2]

                    for v3 in v1_nbrs_minus_v2_nbrs - {v2}:
                        v3_nbrs = set(G.neighbors(v3))
                        
                        cards = calc_cards(v1_nbrs, v2_nbrs, v3_nbrs)

                        tmp_counts[1] += 1/2*cards[6]
                        tmp_counts[2] += 1/2*cards[3]
                        tmp_counts[2] += 1/2*cards[4]
                        tmp_counts[4] += 1/2*(cards[5] - 1)
                        tmp_counts[3] += 1/2*(cards[0] - 2)
                        tmp_counts[5] += 1/2*cards[1]
                        tmp_counts[5] += cards[2]
                        tmp_counts[7] += nodes_count - sum(cards)

                        K += 1/2*cards[6] + 1/2*cards[4] \
                             + 1/2*(cards[5] - 1) + cards[2]
                    
                    for v3 in v2_nbrs_minus_v1_nbrs - {v1}:
                        v3_nbrs = set(G.neighbors(v3))
                        
                        cards = calc_cards(v1_nbrs, v2_nbrs, v3_nbrs)
                        
                        tmp_counts[1] += 1/2*cards[6]
                        tmp_counts[2] += 1/2*cards[3]
                        tmp_counts[4] += 1/2*(cards[4] - 1)
                        tmp_counts[2] += 1/2*cards[5]
                        tmp_counts[5] += 1/2*cards[0]
                        tmp_counts[3] += 1/2*(cards[1] - 2)
                        tmp_counts[5] += cards[2]
                        tmp_counts[7] += nodes_count - sum(cards)
                        
                        K += 1/2*cards[6] + 1/2*(cards[4] - 1) \
                             + 1/2*cards[5] + cards[2]
                             
                    tmp_counts[8] += edges_count + 1 - len(v1_nbrs) \
                                     - len(v2_nbrs) - K
                    tmp_counts[9] += (nodes_count \
                                      - len(v1_nbrs_inter_v2_nbrs) \
                                      - len(v1_nbrs_minus_v2_nbrs) \
                                      - len(v2_nbrs_minus_v1_nbrs)) \
                                     * (nodes_count \
                                        - len(v1_nbrs_inter_v2_nbrs)
                                        - len(v1_nbrs_minus_v2_nbrs)
                                        - len(v2_nbrs_minus_v1_nbrs) - 1)/2 \
                                     - (edges_count + 1 - len(v1_nbrs) \
                                     - len(v2_nbrs) - K)
                    
                    counts += tmp_counts * weights
            
            counts[10] = comb(nodes_count, 4) - sum(counts[:10])           
            
            feature_mat[i] = counts
    
    feature_mat_of_param[None] = feature_mat
    
    extr_end_time = time.time()
    extr_time_of_param[None] = extr_end_time - extr_start_time

    return feature_mat_of_param, extr_time_of_param


if __name__ == '__main__':
    
    from misc import dataset_loader as loader
    
    DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')
    dataset = 'MUTAG'
#    dataset = 'DD'
#    dataset = 'ENZYMES'
#    dataset = 'NCI1'
#    dataset = 'NCI109'
    graph_meta_data_of_num, class_lbls \
        = loader.get_graph_meta_data_and_class_lbls(dataset, DATASETS_PATH)
    
    
    start = time.time()
    feature_mat_of_param, extr_time_of_param = \
            extract_features(graph_meta_data_of_num, 4)
    end = time.time()
    print end - start
    

