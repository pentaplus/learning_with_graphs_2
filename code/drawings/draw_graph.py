"""
Draw some graphs.
"""

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-04-13"


import inspect
import matplotlib.pyplot as plt
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

from misc import pz


ORANGE = '#FF6600'
DARK_BLUE = '#3F3D99'


# GRAPH_NAME = "android_fcg_7ab"    # This graph has 32635 nodes.
# GRAPH_NAME = "dd_class1_1"        # This graph has 327 nodes and 899 edges.
# GRAPH_NAME = "enzymes_class1_201" # This graph has 29 nodes and 53 edges.
GRAPH_NAME = "mutag_class1_1"       # This graph has 23 nodes and 27 edges.
# GRAPH_NAME = "nc1_class0_1"       # This graph has 21 nodes and 21 edges.
# GRAPH_NAME = "nci109_class0_1"    # This graph has 21 nodes and 21 edges.


G = pz.load(GRAPH_NAME + ".pz")
print('number of nodes: ' + str(G.number_of_nodes()))
print('number of edges: ' + str(G.number_of_edges()))

ax = plt.axes(frameon = True)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)



# nc1_class0_1 and nci109_class0_1 (21 nodes and 21 edges) ====================

# k controls the distance between the nodes and varies between 0 and 1
# iterations is the number of times simulated annealing is run
# default k = 0.1 and iterations = 50
#pos = nx.spring_layout(G, k = 0.1, iterations = 10000)
#
#nx.draw_networkx_nodes(G, pos, node_size = 50, node_color = 'r')
#nx.draw_networkx_edges(G, pos, alpha = 0.4)

# =============================================================================



# mutag_class1_1 (23 nodes and 27 edges) ======================================


labels = {}
for i in xrange(G.number_of_nodes()):
    labels[i] = str(i) + ':' + G.nodes(data = True)[i][1]['label']

pos = nx.spring_layout(G, k = 0.2, iterations = 10000)

nx.draw_networkx_nodes(G, pos, node_size = 125, node_color = ORANGE)
nx.draw_networkx_edges(G, pos, alpha = 0.4)

nx.draw_networkx_labels(G, pos, labels, font_size = 9, font_color = DARK_BLUE,
                        font_weight = 'bold')
                        
plt.savefig(os.path.join("drawings", GRAPH_NAME + "_NEW.svg"), dpi = 10000)

# =============================================================================



# enzymes_class1_201 (29 nodes and 53 edges) ==================================


#pos = nx.spring_layout(G, k = 0.065, iterations = 10000)
#
#nx.draw_networkx_nodes(G, pos, node_size = 20, node_color = 'r')
#nx.draw_networkx_edges(G, pos, alpha = 0.4)

# =============================================================================



# dd_class1_1 (327 nodes and 899 edges) =======================================


#pos = nx.spring_layout(G, k = 0.015, iterations = 500)
#
#nx.draw_networkx_nodes(G, pos, node_size = 5, node_color = 'r')
#nx.draw_networkx_edges(G, pos, alpha = 0.4)

# =============================================================================


# plt.savefig(os.path.join("drawings", GRAPH_NAME + ".svg"), dpi = 10000)

