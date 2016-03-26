import networkx as nx
import os
import pz
from os.path import splitext

MUTAG = 'MUTAG'
ENZYMES = 'ENZYMES'
NCI1 = 'NCI1'
NCI109 =  'NCI109'
DD = 'DD'

# choose dataset -----------------------------------------------------------------
#DATASET = MUTAG
DATASET = ENZYMES
#DATASET = NCI1
#DATASET = NCI109
#DATASET = DD
# --------------------------------------------------------------------------------

FIND_NODE_LABELS_LINE = 0
PARSE_NODE_LABELS = 1
FIND_ADJ_LIST_LINE = 2
PARSE_ADJ_LIST = 3
PARSE_EDGE_LBLS_FOR_MUTAG = 4
PARSE_EDGE_WEIGHTS_FOR_NCI = 5

   

def read_node_labels(node_labels_line):
    # splitting on whitespaces
    return node_labels_line.split()



files = os.listdir('.')

graph_numbers = []
for f in files:
    if not f.endswith('.graph'):
        continue
    
    file_name = splitext(f)[0]
    if file_name.isdigit():
        graph_numbers.append(int(file_name))


for graph_num in graph_numbers:
    print('Converting graph no ' + str(graph_num))
    
    adjacency_list = []
    edges = []
    weight_lists = []
            
    mode = FIND_NODE_LABELS_LINE

    with open(str(graph_num) + '.graph', 'r') as f:
        # ------------------------------------------------------------------------
        # 1) parse graph file
        # ------------------------------------------------------------------------
        for line in f:
            line = line.rstrip()
            
            if mode == FIND_NODE_LABELS_LINE:
                if line == 'node labels':
                    mode = PARSE_NODE_LABELS
                    continue
            
            elif mode == PARSE_NODE_LABELS:
                node_labels = read_node_labels(line)
                mode = FIND_ADJ_LIST_LINE
                continue
                
            elif mode == FIND_ADJ_LIST_LINE:
                if line == 'adjacency list':
                    mode = PARSE_ADJ_LIST
                    continue
                
            elif mode == PARSE_ADJ_LIST:
                if line == 'edge labels' and DATASET == MUTAG:
                    mode = PARSE_EDGE_LBLS_FOR_MUTAG
                    continue
                
                if line == 'edge labels' and DATASET in [NCI1, NCI109]:
                    mode = PARSE_EDGE_WEIGHTS_FOR_NCI
                    continue
                
                nbrs_str = line.split()
                nbrs_int = [int(nbr_str) for nbr_str in nbrs_str]
                adjacency_list.append(nbrs_int)
                continue
                
            elif mode == PARSE_EDGE_LBLS_FOR_MUTAG:
                edge_str = line.split()
                edge_int = map(int, edge_str)
                if len(edge_int) == 3:
                    edges.append(tuple(edge_int))
                continue
                    
            elif mode == PARSE_EDGE_WEIGHTS_FOR_NCI:
                weight_list_str = line.split()
                weight_list_int = map(int, weight_list_str)
                weight_lists.append(weight_list_int)
    
    # ----------------------------------------------------------------------------
    # 2) create a networkx graph corresponding to the parsed graph
    # ----------------------------------------------------------------------------
    
    # create an empty graph (according to the description of the datasets covered
    # in this script all corresponding graphs are undirected)        
    G = nx.Graph()
    
    # add nodes to the graph
    nodes_count = len(node_labels)
    
    for i in xrange(nodes_count):
        G.add_node(i, label = node_labels[i])
    
    # add edges to the graph
    if DATASET in [ENZYMES, DD]:
        nodes_count = len(adjacency_list)
        for i in xrange(nodes_count):
            nbrs = adjacency_list[i]
            for j in nbrs:
                G.add_edge(i, j - 1)
            
    elif DATASET in [NCI1, NCI109]:
        if len(adjacency_list) != len(weight_lists):
            print('len(adjacency_list) != len(weight_lists)')
            exit()
        
        for i in xrange(nodes_count):
            nbrs = adjacency_list[i]
            weights = []
            try:
                weights = weight_lists[i]
            except IndexError:
                exit()
                
            if len(nbrs) != len(weights):
                print('len(nbrs) != len(weights)')
                exit()
            
            nbrs_count = len(nbrs)
            for j in xrange(nbrs_count):
                G.add_edge(i, nbrs[j] - 1, weight = weights[j]) 
    
    elif DATASET == MUTAG:   
        edges_count = len(edges)
        for i in xrange(edges_count):
            G.add_edge(edges[i][0] - 1, edges[i][1] - 1,
            weight = edges[i][2])
    
        
    pz.save(G, str(graph_num) + ".pz")