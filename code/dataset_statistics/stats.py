"""
Statistics of datasets.
"""

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-03-18"


import inspect
import numpy as np
import sys
import time

from os.path import abspath, dirname, join

# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import dataset_loader, pz


t0 = time.time()

DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')

# datasets
MUTAG = 'MUTAG'
PTC_MR = 'PTC(MR)'
ENZYMES = 'ENZYMES'
DD = 'DD'
NCI1 = 'NCI1'
NCI109 = 'NCI109'
ANDROID_FCG_14795 = 'ANDROID FCG 14795'
FLASH_CFG = 'FLASH CFG'


#DATASET = MUTAG
#DATASET = PTC(MR)
#DATASET = ENZYMES
#DATASET = DD
#DATASET = NCI1
#DATASET = NCI109
DATASET = FLASH_CFG
#DATASET = ANDROID_FCG_14795


graph_meta_data_of_num, class_lbls \
    = dataset_loader.get_graph_meta_data_and_class_lbls(DATASET, DATASETS_PATH)
    
graphs_of_class = dataset_loader.get_graphs_of_class_dict(graph_meta_data_of_num)

classes = graphs_of_class.keys()

# calculate statistics
node_counts = []
edge_counts = []
degrees = []
min_deg = float('inf')
max_deg = 0
number_of_isolated_nodes = 0

for graph_path, class_lbl in graph_meta_data_of_num.itervalues():
    G = pz.load(graph_path)
    node_counts.append(G.number_of_nodes())
    edge_counts.append(G.number_of_edges())
    degrees.append(np.mean(G.degree().values()))
    
    if min(G.degree().values()) < min_deg:
        min_deg = min(G.degree().values())
        
    if max(G.degree().values()) > max_deg:
        max_deg = max(G.degree().values())
        
    for degree in G.degree().values():
        if degree == 0:
           number_of_isolated_nodes += 1 

avg_v = np.mean(node_counts)
avg_e = np.mean(edge_counts)
max_v = max(node_counts)
max_e = max(edge_counts)
min_v = min(node_counts)
avg_deg = np.mean(degrees)

print 'dataset:', DATASET
print '# graphs:', len(graph_meta_data_of_num)
print '# classes:', len(classes)

for class_lbl in graphs_of_class.iterkeys():
    print 'class %d: %d' % (class_lbl, len(graphs_of_class[class_lbl]))

print 'avg_v: %.1f' % avg_v
print 'max_v:', max_v
print 'min_v:', min_v
if DATASET not in [FLASH_CFG, ANDROID_FCG_14795]:
    print 'avg_e: %.1f, 2*avg_e: %.1f' % (avg_e, 2*avg_e)
    print 'max_e: %d, 2*max_e: %d' % (max_e, 2*max_e)
else:
    print 'avg_e: %.1f' % avg_e
    print 'max_e: %d' % max_e
print 'avg_deg: %.2f' % avg_deg
print 'max_deg:', max_deg
#print 'min_deg:', min_deg
print 'isolated:', number_of_isolated_nodes, '\n'

t1 = time.time()
total = t1 - t0

print 'The execution took %.2f seconds.' % total

