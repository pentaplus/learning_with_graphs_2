"""
Conversion of Flash control flow graphs to networkx graphs.
"""

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-02-28"


import inspect
import networkx as nx
import os
import re
import sys

from os.path import abspath, basename, dirname, join, splitext


# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import dataset_loader, pz, utils


class Id_to_num_mapper():
    def __init__(self):
        self.id_to_num_map = {}
        self.next_num = 0
    
    def map_id_to_num(self, id_to_map):
        if id_to_map in self.id_to_num_map.iterkeys():
            id_num = self.id_to_num_map[id_to_map]
        else:
            id_num = self.next_num
            self.id_to_num_map[id_to_map] = self.next_num
            self.next_num += 1
        
        return id_num
    

def parse_block(line, G, id_to_num_mapper, line_num):
    block_with_succ = False
    block_with_succ_reg_exp = 'block\s+@-?\d{5,}\s+(\S+)\s+\([ms]:\S+\)\s+(\S+)'
    block_without_succ_reg_exp = 'block\s+@-?\d{5,}\s+(\S+)\s+\([ms]:\S+\)'
    
    m = re.match(block_with_succ_reg_exp, line)
    if m:
        block_with_succ = True
    else:
        m = re.match(block_without_succ_reg_exp, line)
        if not m:
            print str(line_num) + '!'
            sys.exit(1)
        
    block_id = m.group(1)
    block_num = id_to_num_mapper.map_id_to_num(block_id)
    
    if block_with_succ:
        succ_node_id = m.group(2)
        succ_node_num = id_to_num_mapper.map_id_to_num(succ_node_id)
        G.add_edge(block_num, succ_node_num)
        
    G.add_node(block_num, label = 'b')

    
def parse_cond(line, G, id_to_num_mapper, line_num):
    cond_with_meta_data_reg_exp =\
                                 'cond\s+@-?\d{5,}\s+(\S+)\s+\([ms]:\S+\)\s+(\S+)'
    cond_without_meta_data_reg_exp = 'cond\s+@-?\d{5,}\s+(\S+)\s+(\S+)'
    
    m = re.match(cond_with_meta_data_reg_exp, line)
    if not m:
        m = re.match(cond_without_meta_data_reg_exp, line)
    if not m:
        print str(line_num) + '!'
        sys.exit(1)
        
    cond_id = m.group(1)
    cond_num = id_to_num_mapper.map_id_to_num(cond_id)
    
    succ_node_ids = m.group(2).split(',')
    for succ_node_id in succ_node_ids:
        succ_node_num = id_to_num_mapper.map_id_to_num(succ_node_id)
        G.add_edge(cond_num, succ_node_num)
        
    G.add_node(cond_num, label = 'c')


def parse_func(line, G, id_to_num_mapper, line_num):
    m = re.match('fct\s+@-?\d{5,}\s+(\S+)\s+(\S+)', line)
    if not m:
        print str(line_num) + '!'
        sys.exit(1)
    
    func_id = m.group(1)
    func_num = id_to_num_mapper.map_id_to_num(func_id)
    
    succ_node_id = m.group(2)
    succ_node_num = id_to_num_mapper.map_id_to_num(succ_node_id)
    G.add_edge(func_num, succ_node_num)
    
    G.add_node(func_num, label = 'f')


def parse_ref(line, G, id_to_num_mapper, line_num):
    m = re.match('ref\s+@-?\d{5,}\s+(\S+)\s+(\S+)', line)
    if not m:
        print str(line_num) + '!'
        sys.exit(1)
    
    ref_id = m.group(1)
    ref_num = id_to_num_mapper.map_id_to_num(ref_id)
    
    succ_node_ids = m.group(2).split(',')
    for succ_node_id in succ_node_ids:
        succ_node_num = id_to_num_mapper.map_id_to_num(succ_node_id)
        G.add_edge(ref_num, succ_node_num)
        
    G.add_node(ref_num, label = 'r')
    
                           
SOURCE_CLASSES_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets',
                           ('FLASH CFG (2 classes, 1715 directed graphs, '
                           'unlabeled edges)'), 'plain')
                    
folder_of_class = dataset_loader.get_folder_of_class_dict(SOURCE_CLASSES_PATH)


utils.check_for_pz_folder()


for folder in folder_of_class.itervalues():
    source_class_path = join(SOURCE_CLASSES_PATH, folder)
    target_class_path = join('pz', folder)
    os.makedirs(target_class_path)
    
    for file_num, file_name in enumerate(utils.list_files(source_class_path)):
        base_file_name, file_extension = splitext(file_name)
        
        if not file_extension == '.cfg':
            continue
        
        with open(join(source_class_path, file_name)) as f:
            print '\n' + str(file_num) + ': ' + basename(f.name) + ':'
            # --------------------------------------------------------------------
            # parse graph file and create a corresponding directed networkx graph
            # --------------------------------------------------------------------
            G = nx.DiGraph()
            
            id_to_num_mapper = Id_to_num_mapper()
        
            for line_num, line in enumerate(f):
                if (line_num + 1) % 1000 == 0:
                    print line_num + 1
                
                line = line.rstrip()
                
#                if basename(f.name).startswith('000c6b') and line_num + 1 == 87:
#                    x = 0
                
                if line.startswith('block'):
                    parse_block(line, G, id_to_num_mapper, line_num + 1)
                elif line.startswith('cond'):
                    parse_cond(line, G, id_to_num_mapper, line_num + 1)
                elif line.startswith('fct'):
                    parse_func(line, G, id_to_num_mapper, line_num + 1)
                elif line.startswith('ref'):
                    parse_ref(line, G, id_to_num_mapper, line_num + 1)
                    
                
            pz.save(G, join(target_class_path, base_file_name + '.pz'))



#TEST = True 
##TEST = False
#if TEST:
#    sys.path.append(join(SCRIPT_FOLDER_PATH, '..', '..'))
#    from misc import dataset_loader, pz, utils
#    
#    file_names = os.listdir('.')
#    file_name = file_names[0]
#    G = pz.load(file_name)
#    
#    problematic_nodes = []
#    counter = 0
#    for node_num, label_dict in G.node.iteritems():
#        label = label_dict['label']
#        if label not in ['b', 'c', 'f', 'r']:
#            problematic_nodes.append(node_num)
#        else:
#            counter += 1
#    
#    if len(G.node) == counter:
#        print 'It holds: len(G.node) == counter'
#        
