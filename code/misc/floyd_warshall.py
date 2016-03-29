"""
Floyd-Warshall algorithm.

This module provides the function compute_kernel_mat for the
computation of the corresponding kernel matrix. It is a translation of
the MATLAB file RWkernel.m by Karsten Borgwardt and Nino Shervashidze,
which can be downloaded from the following website:
http://mlcb.is.tuebingen.mpg.de/Mitarbeiter/Nino/Graphkernels/
"""

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__credits__ = ["Karsten Borgwardt", "Nino Shervashidze"]
__date__ = "2016-02-28"


import inspect
import networkx as nx
import math
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


def floyd_warshall(A):

    
#    n = size(A,1); % number of nodes
    n = A.shape[0]
    
#    D=zeros(n,n);
    D = np.zeros(n, n)
    
    
#    if nargin<3 % if the graph is not weighted, then
#      w=A;
#    end
    w = A
    
#    D=w.*A; # if A(i,j)=1,  D(i,j)=w(i,j);
    D = np.multiply(w, A)    
    
#    D(A+diag(repmat(Inf,n,1))==0)=Inf; % If A(i,j)~=0 and i~=j D(i,j)=Inf;
    # !!
    # repmat(Inf, n, 1)
    D[A + np.diag(np.tile(np.inf, n)) == 0] = np.inf
        
    
    D=full(D.*(ones(n)-eye(n))); % set the diagonal to zero
    
#    %t=cputime;
#    if sym % then it is a bit faster
#      for k=1:n
#        Daux=repmat(full(D(:,k)),1,n);
#        Sumdist=Daux+Daux';
#        D(Sumdist<D)=Sumdist(Sumdist<D);
#      end
#    else  
      for k=1:n
        Daux1=repmat(full(D(:,k)),1,n);
        Daux2=repmat(full(D(k,:)),n,1);
        Sumdist=Daux1+Daux2;
        D(Sumdist<D)=Sumdist(Sumdist<D);

