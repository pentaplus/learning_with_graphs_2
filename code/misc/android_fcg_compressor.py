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

SOURCE_CLASSES_PATH = join(DATASETS_PATH, 'ANDROID FCG 14795 (2 classes, '
                           '14795 directed graphs, unlabeled edges)', 'pz')

utils.check_for_pz_folder()
                           
os.makedirs('pz')

class_folders = utils.list_sub_dirs(SOURCE_CLASSES_PATH)

compressed_graphs_count = 0

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
            if G_uncompr.number_of_edges() == 0:
                print 'Warning! Graph ' + graph_file_name + ' has no edges!'
            
            G_compr = nx.DiGraph()
            
            id_to_num_mapper = utils.Id_to_num_mapper()
            
            # process nodes
            for node_id_tuple, lbl_dict in G_uncompr.node.iteritems():
                node_id = '\n'.join(node_id_tuple)
                node_num = id_to_num_mapper.map_id_to_num(node_id)
                
                lbl_array = lbl_dict['label']
                lbl_num = bin_array_to_num(lbl_array)
                
                G_compr.add_node(node_num, label = lbl_num)
                
            # process edges
            for node_id_tuple, edge_label_dict_of_node_neigh_id_tuple in \
                    G_uncompr.edge.iteritems():
                        
                node_id = '\n'.join(node_id_tuple)
                node_num = id_to_num_mapper.map_id_to_num(node_id)
                
#                G_compr.edge[node_num] = {}
                
                for node_neigh_id_tuple, edge_label_dict in \
                        edge_label_dict_of_node_neigh_id_tuple.iteritems():
                            
                    node_neigh_id = '\n'.join(node_neigh_id_tuple)
                    node_neigh_num = id_to_num_mapper.map_id_to_num(node_neigh_id)
                    
#                    G_compr.edge[node_num][node_neigh_num] = edge_label_dict
                    G_compr.add_edge(node_num, node_neigh_num, **edge_label_dict)
                    
                    
            pz.save(G_compr, join(target_class_path, graph_file_name))
            
            compressed_graphs_count += 1
            
            if compressed_graphs_count % 10 == 0:
                print 'Graphs compressed: %d' % compressed_graphs_count   

