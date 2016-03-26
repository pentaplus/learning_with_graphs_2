import inspect
import networkx as nx
import os
import pz
import sys

from os.path import abspath, dirname, join

# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import utils
    

def read_line(fid):
    return fid.readline().rstrip()


def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False    

def read_class_label(cur_class_label_line, counter):        
    cur_class_label_line_list = cur_class_label_line.split()
    
    if len(cur_class_label_line_list) != 4:
        utils.fatal_error('class label line is badly formatted.', fid)
    if cur_class_label_line_list[0] != 'c':
        utils.fatal_error('class label line does not start with \'c\'.', fid)
    if not is_int(cur_class_label_line_list[2]):        
        utils.fatal_error('graph no is not an integer.', fid)
    if not is_int(cur_class_label_line_list[3]):        
        utils.fatal_error('class label is not an integer.', fid)

    cur_graph_no = int(cur_class_label_line_list[2])        
    cur_class_label = int(cur_class_label_line_list[3])        
        
    if cur_graph_no != counter:
        utils.fatal_error('graph no is incorrect.', fid)    
    if cur_class_label not in [-1, 1]:
        utils.fatal_error('invalid class label.', fid)
        
    return cur_class_label
    
    
def check_graph_no(fid, counter):
    cur_line = read_line(fid)
    if cur_line == '':
        utils.fatal_error('graph number expected', fid)
        
    graph_no_line_list = cur_line.split()
    
    if len(graph_no_line_list) != 3:
        utils.fatal_error('graph no line is badly formatted.', fid)
    if graph_no_line_list[0] != 't':
        utils.fatal_error('graph no line does not start with \'t\'.', fid)
    if graph_no_line_list[1] != '#':
        utils.fatal_error('the second character of a graph no line is not \'#\'.',
                    fid)
    if not is_int(graph_no_line_list[2]):        
        utils.fatal_error('graph no is not an integer.', fid)
        
    cur_graph_no = int(graph_no_line_list[2])
    
    if cur_graph_no != counter:
        utils.fatal_error('graph no is incorrect.', fid)
        
    return


def read_node_list(fid):
    cur_node_counter = 0
    cur_node_labels = []
    
    while True:
        cur_line = read_line(fid)
        if cur_line == '':
            utils.fatal_error('node or edge expected', fid)
        
        if cur_line.startswith('e'):
            if len(cur_node_labels) == 0:
                utils.fatal_error('graph without nodes.', fid)
            
            first_edge_line = cur_line
            return (first_edge_line, cur_node_labels)
            
        cur_node_line_list = cur_line.split()
        
        if len(cur_node_line_list) != 3:
            utils.fatal_error('node line is badly formatted.', fid)
        if cur_node_line_list[0] != 'v':
            utils.fatal_error('node line does not start with \'v\'.', fid)
        if not is_int(cur_node_line_list[1]):
            utils.fatal_error('node no is not an integer', fid)
        if not is_int(cur_node_line_list[2]):
            utils.fatal_error('node label is not an integer', fid)            
        
        cur_node_no = int(cur_node_line_list[1])
        cur_node_label = int(cur_node_line_list[2])
        
        if cur_node_no != cur_node_counter:
            utils.fatal_error('node no is incorrect.', fid)

        cur_node_labels.append(cur_node_label)
            
        cur_node_counter += 1


def read_edge_list(first_edge_line, cur_nodes_count, fid):
    cur_edges = []
    while True:
        if len(cur_edges) == 0:
            cur_line = first_edge_line
        else:
            cur_line = read_line(fid)
            if cur_line == '':
                cur_class_label_line = None
                return (cur_class_label_line, cur_edges) 
        
        if cur_line.startswith('c'):
            if len(cur_edges) == 0:
                utils.fatal_error('graph without edges.', fid)
            
            cur_class_label_line = cur_line
            return (cur_class_label_line, cur_edges)   
            
        cur_edge_line_list = cur_line.split()
        
        if len(cur_edge_line_list) != 4:
            utils.fatal_error('edge line is badly formatted.', fid)
        if cur_edge_line_list[0] != 'e':
            utils.fatal_error('edge line does not start with \'e\'.', fid)
            
        if not is_int(cur_edge_line_list[1]):
            utils.fatal_error('first node of an edge is not an integer', fid)
        if not is_int(cur_edge_line_list[2]):
            utils.fatal_error('second node of an edge is not an integer', fid)
        if not is_int(cur_edge_line_list[3]):
            utils.fatal_error('edge label is not an integer', fid)
        
        cur_first_edge_node = int(cur_edge_line_list[1])
        cur_second_edge_node = int(cur_edge_line_list[2])
        cur_edge_label = int(cur_edge_line_list[3])
        
        if not (0 <= cur_first_edge_node < cur_nodes_count):
            utils.fatal_error('first node of an edge is invalid.', fid)
        if not (0 <= cur_second_edge_node < cur_nodes_count):
            utils.fatal_error('second node of an edge is invalid.', fid)
        if not is_int(cur_edge_label):
            utils.fatal_error('edge label is not an integer', fid)
        
        cur_edge = (cur_first_edge_node, cur_second_edge_node, cur_edge_label)
        cur_edges.append(cur_edge)
    

# main ---------------------------------------------------------------------------
utils.check_for_pz_folder()
    
path_class_1 = os.path.join('pz', 'class 1')
path_class_minus_1 = os.path.join('pz', 'class -1')
os.makedirs(path_class_1)
os.makedirs(path_class_minus_1)

try:
    fid = open('plain.txt', 'r')
except IOError:
    utils.fatal_error('file \'plain.txt\' could not be opened.')
    
counter = 0

cur_class_label_line = read_line(fid)

while True:
    # ----------------------------------------------------------------------------
    # 1) parse graph
    # ----------------------------------------------------------------------------    
    cur_class_label = read_class_label(cur_class_label_line, counter)
        
    check_graph_no(fid, counter)
    
    (first_edge_line, cur_node_labels) = read_node_list(fid)
    cur_nodes_count = len(cur_node_labels)
     
    (cur_class_label_line, cur_edges) = read_edge_list(first_edge_line,
                                                       cur_nodes_count, fid)
    
    
    # ----------------------------------------------------------------------------
    # 2) create a networkx graph corresponding to the parsed graph
    # ----------------------------------------------------------------------------
    cur_path = path_class_1 if cur_class_label == 1 else path_class_minus_1
    file_path = os.path.join(cur_path, str(counter) + '.pz')
    
    G = nx.Graph()
    
    # add nodes to graph G
    nodes_count = len(cur_node_labels)
    
    for i in xrange(nodes_count):
        G.add_node(i, label = cur_node_labels[i])
        
    # add eddges to graph G
    for edge in cur_edges:
        cur_first_edge_node = edge[0]
        cur_second_edge_node = edge[1]
        cur_edge_label = edge[2]
        G.add_edge(cur_first_edge_node, cur_second_edge_node,
                   weight = cur_edge_label)    
        

    pz.save(G, file_path)

    if cur_class_label_line == None:
        break
    
    counter += 1


fid.close()        


