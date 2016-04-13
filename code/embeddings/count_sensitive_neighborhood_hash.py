"""
Count-sensitive neighborhood hash.

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

from embeddings import neighborhood_hash_main


def extract_features(graph_meta_data_of_num, h_range):
    return neighborhood_hash_main.extract_features(graph_meta_data_of_num,
                                                   h_range,
                                                   count_sensitive = True,
                                                   all_iter = False)
    
