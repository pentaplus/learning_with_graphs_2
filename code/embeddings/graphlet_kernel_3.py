"""
Graphlet kernel counting graphlets of size 3.

This module is a wrapper around the module graphlet_kernel_main.py. It
provides the function extract_features for the corresponding feature
extraction.
"""

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-02-28"


import inspect
import sys

from os.path import abspath, dirname, join

# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from embeddings import graphlet_kernel_main


def extract_features(graph_meta_data_of_num, param_range = [None]):
    return graphlet_kernel_main.extract_features(graph_meta_data_of_num, 3)
    

# !!
if __name__ == '__main__':
    import time
    from misc import dataset_loader
        
    DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')
    dataset = 'MUTAG'
    
    graph_meta_data_of_num, class_lbls \
        = dataset_loader.get_graph_meta_data_and_class_lbls(dataset,
                                                            DATASETS_PATH)
    
    start = time.time()
    feature_mat_of_param, extr_time_of_param =\
                                 extract_features(graph_meta_data_of_num, [None])
    
    end = time.time()
    
    print '%r' % feature_mat_of_param
    
    
        