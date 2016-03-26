"""
Renaming of graph files (hash value -> number).
"""

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-02-28"


import inspect
import os
import re
import sys

from os.path import abspath, dirname, join


# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import utils


DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')

ANDROID_FCG_14795_PATH = join(DATASETS_PATH,
                              'ANDROID FCG 14795 (2 classes, 14795 directed graphs, '
                              'unlabeled edges)',
                              'pz')
                                
FLASH_CFG_PATH = join(DATASETS_PATH,
                      'FLASH CFG (2 classes, 1715 directed graphs, unlabeled '
                      'edges)',
                      'pz')
                                

DATASET_PATH = ANDROID_FCG_14795_PATH
#DATASET_PATH = FLASH_CFG_PATH

class_folders = utils.list_sub_dirs(DATASET_PATH)

graph_num = 0
with open(join(DATASET_PATH, 'hash_num_map.txt'), 'w') as f:
    for class_folder in class_folders:
        graph_files_path = join(DATASET_PATH, class_folder)
        graph_file_names = utils.list_files(graph_files_path)
        
        for graph_file_name in graph_file_names:
            os.rename(join(graph_files_path, graph_file_name),
                      join(graph_files_path, str(graph_num) + '.pz'))
                      
            hash_part = re.search('.*?(?=\.)', graph_file_name).group(0)
            f.write(hash_part + ': ' + str(graph_num) + '\n')          
            
            graph_num += 1
            
            if graph_num % 10 == 0:
                print('graph_num: %d' % graph_num)


    