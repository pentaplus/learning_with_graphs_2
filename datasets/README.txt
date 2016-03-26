This folder contains 5 datasets of undirected labeled graphs in plain text as well as
in pz format for graph classification: MUTAG, NCI1, NCI109, ENZYMES, and D&D.


=== Description ===

MUTAG (Debnath et al., 1991) is a dataset of 188 mutagenic aromatic and heteroaromatic
nitro compounds labeled according to whether or not they have a mutagenic effect on the
Gram-negative bacterium Salmonella typhimurium. 

NCI1 and NCI109 represent two balanced subsets of datasets of chemical compounds screened 
for activity against non-small cell lung cancer and ovarian cancer cell lines respectively
(Wale and Karypis (2006) and http://pubchem.ncbi.nlm.nih.gov). 

ENZYMES is a dataset of protein tertiary structures obtained from (Borgwardt et al., 
2005) consisting of 600 enzymes from the BRENDA enzyme database (Schomburg et al., 2004). 
In this case the task is to correctly assign each enzyme to one of the 6 EC top-level 
classes. 

D&D is a dataset of 1178 protein structures (Dobson and Doig, 2003). Each protein is 
represented by a graph, in which the nodes are amino acids and two nodes are connected 
by an edge if they are less than 6 Angstroms apart. The prediction task is to classify 
the protein structures into enzymes and non-enzymes.