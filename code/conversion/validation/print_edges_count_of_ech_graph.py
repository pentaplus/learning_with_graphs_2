"""
Printing the number of edges for each graph within a dataset.
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
sys.path.append(join(SCRIPT_FOLDER_PATH, '..', '..'))

from misc import dataset_loader, pz


DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', '..', 'datasets')
# dataset = 'MUTAG'
# dataset = 'DD'
dataset = 'ENZYMES'
# dataset = 'NCI1'
# dataset = 'NCI109'

graph_meta_data_of_num, class_lbls \
    = dataset_loader.get_graph_meta_data_and_class_lbls(dataset, DATASETS_PATH)

f = open('python_edges_count_of_each_graph.csv', 'w')    
for graph_num, (graph_path, class_lbl) in graph_meta_data_of_num.iteritems():
    G = pz.load(graph_path)
    f.write(str(graph_num) + '; ' + str(2*G.number_of_edges()) + '\n')
f.close()