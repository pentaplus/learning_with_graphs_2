import networkx as nx
import os
import shutil
import sys
import time

from itertools import tee
from os import listdir
from os.path import isdir, isfile, join


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
        

def calc_hash_of_array(array):
    array.flags.writeable = False
    return hash(array.data)
    
    
def check_for_pz_folder():
    if isdir('pz'):
        shutil.rmtree('pz') # !!
        return        
        
        user_input = raw_input('The directory \'pz\' already exists. '
                               'Do you want to delete it (y/n)? ').strip()
        while True:
            if user_input == 'y':
                shutil.rmtree('pz')
                time.sleep(1)
                break
            if user_input == 'n':
                sys.exit(1)
            
            user_input = raw_input('Invalid input! The directory \'pz\' already '
                                   'exists. Do you want to delete it '
                                   '(y/n)? ').strip()
        

def fatal_error(msg, fid = None):
    print('Fatal error: ' + msg)
    
    if fid != None:
        fid.close()
    
    sys.exit(1)
    

# !!   
def get_adjacency_matrix(G):
    return (nx.to_numpy_matrix(G) != 0).astype(int)


def has_elem(it): 
    it, any_check = tee(it)
    try:
        any_check.next()
        return True, it
    except StopIteration:
        return False, iter


def list_sub_dirs(path):
    return [d for d in listdir(path) if isdir(join(path, d))]

        
def list_files(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


def makedir(path):
    try: 
        os.makedirs(path)
    except OSError:
        if not isdir(path):
            raise

  
def write(string, result_file):
    sys.stdout.write(string)
    result_file.write(string)
        
    

        


#    for k in d.iterkeys():
#        d[k] = {}
#    return