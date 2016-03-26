"""
Dataset loader.

This module provides functionality for loading the meta data of the
specified dataset. This data comprises the absolute paths of the graph
files and their class labels.
"""

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-02-28"


import inspect
import numpy as np
import pz
import re
import sys

from collections import defaultdict, OrderedDict
from os import listdir
from os.path import abspath, dirname, join

# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import utils


def get_folder_of_dataset_dict(datasets_path):
    folder_of_dataset = {}
    folders = utils.list_sub_dirs(datasets_path)
    for folder in folders:
        m = re.match('.*(?= \()', folder)
        if not m:
            continue
        
        dataset_name = m.group(0)
        folder_of_dataset[dataset_name] = folder
        
    return folder_of_dataset
    
    
def get_folder_of_class_dict(classes_path):
    folder_of_class = {}
    dataset_classes = listdir(classes_path)
    for dataset_class in dataset_classes:
        reg_exp = '(?<=class )-?\d+(?= \()' if '(' in dataset_class else \
                  '(?<=class )-?\d+'
        m = re.search(reg_exp, dataset_class)
        if not m:
            continue
        
        class_lbl = int(m.group(0))
        folder_of_class[class_lbl] = dataset_class
    
    return folder_of_class
        
    
def get_class_lbls(graph_meta_data_of_num):
    class_lbls = []
    for graph_path, class_lbl in graph_meta_data_of_num.itervalues():
        class_lbls.append(class_lbl)
        
    return np.array(class_lbls)
    
    
def get_graph_meta_data_and_class_lbls(dataset, datasets_path):
    folder_of_dataset = get_folder_of_dataset_dict(datasets_path)
    
    datasets = folder_of_dataset.keys()
    if not dataset in datasets:
        print '%s is not a valid dataset name.' % dataset
        sys.exit(1)
    
    classes_path = join(datasets_path, folder_of_dataset[dataset], 'pz')
    
    folder_of_class = get_folder_of_class_dict(classes_path)
    
    graph_meta_data_of_num = {}
    
    for class_lbl, folder in folder_of_class.iteritems():
        path_to_graphs_of_cur_class = join(classes_path, folder)    
        
        for graph_file in utils.list_files(path_to_graphs_of_cur_class):
            m = re.match('\d+(?=.pz)', graph_file)
            if not m:
                continue
            
            graph_num = int(m.group(0))
            # load graph with number graph_num
            graph_path = join(path_to_graphs_of_cur_class, graph_file)
            graph_meta_data_of_num[graph_num] = (graph_path, class_lbl)
            
    graph_meta_data_of_num =\
            OrderedDict(sorted(graph_meta_data_of_num.iteritems()))
                           
    class_lbls = get_class_lbls(graph_meta_data_of_num)
    
    return graph_meta_data_of_num, class_lbls
    

def get_graphs_of_class_dict(graph_meta_data_of_num):
    graphs_of_class = defaultdict(list)

    for graph_num, (graph_path, class_lbl) in graph_meta_data_of_num.iteritems():
        graphs_of_class[class_lbl].append((graph_num, graph_path))
        
    return graphs_of_class
    
    
    
if __name__ == '__main__':
    DATASET_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')
#    dataset = 'MUTAG'
#    dataset = 'PTC(MR)'
    dataset = 'ANDROID FCG 14795'
    
    graph_meta_data_of_num, class_lbls \
        = get_graph_meta_data_and_class_lbls(dataset, DATASET_PATH)
                                                             
    G = pz.load(graph_meta_data_of_num[0][0])
    
    
#G = pz.load(graph_meta_data_of_num[13669][0])
#G.number_of_nodes()
#
#G = pz.load(graph_meta_data_of_num[13670][0])
#G.number_of_nodes()
#
#pz.load(graph_meta_data_of_num[14182][0]).number_of_nodes()
#
#G = pz.load('121563fa0518ed7bf843f08958ea1cab302a304dd990ff792bba893ebf716fdf.fcgnx.pckl')


