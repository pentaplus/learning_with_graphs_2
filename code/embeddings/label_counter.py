"""
Statistical kernel.

This module provides the function extract_features for the corresponding
feature extraction.
"""

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-02-28"


import inspect
import numpy as np
import sys
import time

from collections import defaultdict
from os.path import abspath, dirname, join
from scipy.sparse import csr_matrix


# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import pz


def extract_features(graph_meta_data_of_num, param_range = [None]):
    extr_start_time = time.time()
    
    # the keys are graph numbers and the values are lists of features   
    features_dict = defaultdict(list)
    
    # the keys are graph numbers and the values are lists which contain the number
    # of occurences of the features corresponding to the feature at the same index
    # in the feature list in features_dict, that is
    # feature_counts_dict[graph_number][i] == number of occurences of feature
    # features_dict[graph_number][i]
    feature_counts_dict = defaultdict(list)
    
    # the keys are graph numbers and the values are dictionaries which map
    # features to their position in features_dict[graph_number] and
    # feature_counts_dict[graph_number], respectively
    idx_of_lbl_dict = defaultdict(dict)
    
    # the keys are graph numbers and the values are dictionaries which map
    # nodes to their updated label
    upd_lbls_dict = defaultdict(dict)
    
    # keys are the node labels which are stored in the dataset and the values are
    # new compressed labels
    compr_func = {}
    
    # next_compr_lbl is used for assigning new compressed labels to the nodes
    # These build the features (= columns in feature_mat) used for the explicit
    # graph embedding
    next_compr_lbl = 0


    # iterate over all graphs in the dataset -------------------------------------
    # r == 0
    for graph_num, (graph_path, class_lbl) in graph_meta_data_of_num.iteritems():
        G = pz.load(graph_path)
        
        for v in G:
            uncompr_lbl = G.node[v]['label']
            if not uncompr_lbl in compr_func:
                # assign a compressed label new_compr_lbl to uncompr_lbl
                new_compr_lbl = next_compr_lbl
                compr_func[uncompr_lbl] = new_compr_lbl
                next_compr_lbl += 1
            else:
                # determine compressed label new_compr_lbl assigned to
                # uncompr_lbl
                new_compr_lbl = compr_func[uncompr_lbl]

            if new_compr_lbl not in idx_of_lbl_dict[graph_num]:
                # len(feature_counts_dict[graph_num])
                # == len(features_dict[graph_num])
                idx = len(feature_counts_dict[graph_num])

                idx_of_lbl_dict[graph_num][new_compr_lbl] = idx

                # features_dict[graph_num][idx]
                # == feature upd_lbls_dict[graph_num][v] (== new_compr_lbl)
                features_dict[graph_num].append(new_compr_lbl)

                # increase number of occurrences of the feature
                # upd_lbls_dict[graph_num][v] (== new_compr_lbl)
                feature_counts_dict[graph_num].append(1)
            else:
                # features_dict[graph_num][idx]
                # == feature upd_lbls_dict[graph_num][v] (== new_compr_lbl)
                idx = idx_of_lbl_dict[graph_num][new_compr_lbl]

                # increase number of occurrences of the feature
                # upd_lbls_dict[graph_num][v] (== new_compr_lbl)
                feature_counts_dict[graph_num][idx] += 1

            # upd_lbls_dict[graph_num][v] == compr_func[lbl]
            # == new_compr_lbl
            upd_lbls_dict[graph_num][v] = new_compr_lbl


    # list containing the features of all graphs
    features = []

    # list containing the corresponding features counts of all graphs
    feature_counts = []

    # list indicating to which graph (= row in feature_mat) the features in the
    # list features belong. The difference feature_ptr[i+1] - feature_ptr[i]
    # equals the number of specified entries for row i. Consequently, the number
    # of rows of feature_mat equals len(feature_ptr) - 1.
    feature_ptr = [0]


    for graph_num in graph_meta_data_of_num.iterkeys():
        features += features_dict[graph_num]
        feature_counts += feature_counts_dict[graph_num]
        feature_ptr.append(feature_ptr[-1] + len(features_dict[graph_num]))


    # feature_mat is of type csr_matrix and has the following form:
    # [feature vector of the first graph,
    #  feature vector of the second graph,
    #                .
    #                .
    #  feature vector of the last graph]
    feature_mat = csr_matrix((np.array(feature_counts), np.array(features),
                           np.array(feature_ptr)),
                          shape = (len(graph_meta_data_of_num), len(compr_func)),
                          dtype = np.float64)
                          

    extr_end_time = time.time()
    extr_time = extr_end_time - extr_start_time

    # !! DEBUG
#    Z = feature_mat.todense()

    return {None: feature_mat}, {None: extr_time}


# !!
if __name__ == '__main__':
    from misc import dataset_loader
    
    DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')
    dataset = 'MUTAG'
#    dataset = 'DD'
#    dataset = 'ENZYMES'
#    dataset = 'NCI1'
#    dataset = 'NCI109'
    
    graph_meta_data_of_num, class_lbls \
        = dataset_loader.get_graph_meta_data_and_class_lbls(dataset,
                                                            DATASETS_PATH)
    
    
    start = time.time()
    feature_mat_of_param, extr_time_of_param =\
                                  extract_features(graph_meta_data_of_num, [None])
    end = time.time()
    print end - start