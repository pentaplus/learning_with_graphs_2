"""
Android FCG compressor.

This module provides functionality for compressing Android function
call graphs. The compression is performed by mapping the node
identifiers and bit labels from strings to numbers and from
numpy arrays to numbers, respectively.
"""

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-02-28"


import inspect
import networkx as nx
import os
import sys

from os.path import abspath, dirname, join



# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import pz, utils

def bin_array_to_num(array):
    array_str = ''
    for i in xrange(array.shape[0]):
        array_str += str(array[i])
        
    return int(array_str, 2)


DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')

SOURCE_CLASSES_PATH = '/media/benjamin/Backups/ANDROID FCG/pz'

utils.check_for_pz_folder()
                           
os.makedirs('pz')

class_folders = utils.list_sub_dirs(SOURCE_CLASSES_PATH)

counter = 0

with open(join(SOURCE_CLASSES_PATH, 'hash_num_map.txt'), 'w') as f:
    for class_folder in class_folders:
        source_class_path = join(SOURCE_CLASSES_PATH, class_folder)
        target_class_path = join('pz', class_folder)
        os.makedirs(target_class_path)
        
        graph_file_names = utils.list_files(source_class_path)
        
        for graph_file_name in graph_file_names:
            id_to_num_mapper = utils.Id_to_num_mapper()
            G_uncompr = pz.load(join(source_class_path, graph_file_name))
            
            if G_uncompr.number_of_nodes() == 0:
                print 'Warning! Graph ' + graph_file_name + ' has no nodes!'
                os.remove(join(source_class_path, graph_file_name))
            elif G_uncompr.number_of_edges() == 0:
                print 'Warning! Graph ' + graph_file_name + ' has no edges!'
                os.remove(join(source_class_path, graph_file_name))
                
            counter += 1
            

            
            if counter % 100 == 0:
                print 'Graphs processed: %d' % counter  

