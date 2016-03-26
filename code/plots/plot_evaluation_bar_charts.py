# 1. RW
# 2. EGK

"""
Plot bar charts representing the results of the embedding methods.

The plots show the classification accuracy and the runtime of the
respective methods.
"""

from __future__ import division

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-03-19"


import inspect
import itertools
import matplotlib as mpl
import numpy as np
import sys

from os.path import abspath, dirname, join

# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))


#=================================================================================
# constants
#=================================================================================
TARGET_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', '..', 'tex', 'figures')

# embeddings
WEISFEILER_LEHMAN = 'weisfeiler_lehman'
NEIGHBORHOOD_HASH = 'neighborhood_hash'
COUNT_SENSITIVE_NEIGHBORHOOD_HASH = 'count_sensitive_neighborhood_hash'
COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER \
    = 'count_sensitive_neighborhood_hash_all_iter'
GRAPHLET_KERNEL_3 = 'graphlet_kernel_3'
GRAPHLET_KERNEL_4 = 'graphlet_kernel_4'
LABEL_COUNTER = 'label_counter'
RANDOM_WALK_KERNEL = 'random_walk_kernel'
EIGEN_KERNEL = 'eigen_kernel'

# datasets
MUTAG = 'MUTAG'
PTC_MR = 'PTC(MR)'
ENZYMES = 'ENZYMES'
DD = 'DD'
NCI1 = 'NCI1'
NCI109 = 'NCI109'
FLASH_CFG = 'FLASH CFG'
ANDROID_FCG = 'ANDROID FCG'

SMALL = 'small'
LARGE = 'large'

SCORES = 'scores'
RUNTIMES = 'runtimes'


EMBEDDING_ABBRVS = {
    WEISFEILER_LEHMAN: 'WL',
    NEIGHBORHOOD_HASH: 'NH',
    COUNT_SENSITIVE_NEIGHBORHOOD_HASH: 'CSNH',
    COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER: 'CSNH ALL',
    GRAPHLET_KERNEL_3: '3-GK',
    GRAPHLET_KERNEL_4: '4-GK',
    RANDOM_WALK_KERNEL: 'RW',
    EIGEN_KERNEL: 'EGK'}
    
    
FONT_SIZE = 10
LEGEND_FONT_SIZE = 6


    
DATASET_TYPES = [SMALL, LARGE]

MODES = [SCORES, RUNTIMES]


mpl.use("pgf")
pgf_with_rc_fonts = {
    "font.family": "Minion Pro",
#    "font.serif": [],                   # use latex default serif font
#    "font.sans-serif": ["DejaVu Sans"], # use a specific sans-serif font
    "pgf.texsystem": "xelatex",
    "text.fontsize": FONT_SIZE
}
mpl.rcParams.update(pgf_with_rc_fonts)


# must be imported after the specification of the RC parameters
import matplotlib.pyplot as plt


# The data matrices DATA_SD and DATA_LD have the following columns:
# embedding name, dataset, score, standard deviation, runtime (in seconds)
DATA_SD = np.array(
    [[WEISFEILER_LEHMAN, MUTAG, 91.3, 0.8, 110.3/10 + 0.6],
     [WEISFEILER_LEHMAN, PTC_MR, 64.6, 1.1, 425.8/10 + 1.2],
     [WEISFEILER_LEHMAN, ENZYMES, 60.7, 1.2, 998.5/10 + 2.6],
     [NEIGHBORHOOD_HASH, MUTAG, 88.9, 0.8, 108.7/10 + 0.5],
     [NEIGHBORHOOD_HASH, PTC_MR, 66.4, 1.0, 435.7/10 + 1.3],
     [NEIGHBORHOOD_HASH, ENZYMES, 46.8, 1.0, 1325.8/10 + 2.1],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH, MUTAG, 91.1, 1.0, 109.7/10 + 0.6],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH, PTC_MR, 65.0, 1.0, 288.5/10 + 1.3],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH, ENZYMES, 56.2, 1.0, 598.5/10 + 2.8],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, MUTAG, 91.4, 0.8,
      112.0/10 + 0.6],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, PTC_MR, 65.6, 1.3,
      397.9/10 + 1.3],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, ENZYMES, 61.4, 1.3,
      1036.8/10 + 3.1],
     [GRAPHLET_KERNEL_3, MUTAG, 85.8, 2.1, 929.7/10 + 0.2],
     [GRAPHLET_KERNEL_3, PTC_MR, 55.3, 1.0, 3988.8/10 + 0.3],
     [GRAPHLET_KERNEL_3, ENZYMES, 19.6, 1.9, 31158.3/10 + 1.1],
     [GRAPHLET_KERNEL_4, MUTAG, 86.4, 1.0, 1370.8/10 + 0.7],
     [GRAPHLET_KERNEL_4, PTC_MR, 54.4, 1.7, 4668.8/10 + 1.5],
     [GRAPHLET_KERNEL_4, ENZYMES, 17.6, 1.4, 41190.1/10 + 8.6],
     [RANDOM_WALK_KERNEL, MUTAG,  83.4, 2.7, 15.4/10 + 11.2],
     [RANDOM_WALK_KERNEL, PTC_MR, 53.5, 1.4, 55.7/10 + 57.9],
     [RANDOM_WALK_KERNEL, ENZYMES, 14.8, 0.8, 33.1/10 + 231.4],
     [EIGEN_KERNEL, MUTAG, 88.6, 0.5, 106.6/10 + 14.8],
     [EIGEN_KERNEL, PTC_MR, 61.9, 1.0, 130.4/10 + 62.5],
     [EIGEN_KERNEL, ENZYMES, 25.7, 1.1, 803.2/10 + 99.6]])
     

DATA_LD = np.array(
    [[WEISFEILER_LEHMAN, DD, 79.1, 0.5, 2170.2/10 + 31.9],
     [WEISFEILER_LEHMAN, NCI1, 86.0, 0.2, 2603.8/10 + 17.6],
     [WEISFEILER_LEHMAN, NCI109, 86.3, 0.1, 2636.1/10 + 17.4],
     [WEISFEILER_LEHMAN, FLASH_CFG, 85.9, 0.4, 676.3/10 + 129.2],
     [WEISFEILER_LEHMAN, ANDROID_FCG, 90.0, 0.0, 3600.0],
     [NEIGHBORHOOD_HASH, DD, 76.7, 1.6, 255.9/10 + 26.0],
     [NEIGHBORHOOD_HASH, NCI1, 79.1, 0.3, 655.6/10 + 16.7],
     [NEIGHBORHOOD_HASH, NCI109, 79.1, 0.3, 684.1/10 + 15.3],
     [NEIGHBORHOOD_HASH, FLASH_CFG, 83.9, 2.4, 405.2/10 + 117.8],
     [NEIGHBORHOOD_HASH, ANDROID_FCG, 90.0, 0.0, 3600.0],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH, DD, 77.8, 0.8, 259.8/10 + 40.9],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH, NCI1, 83.9, 0.2, 742.0/10 + 19.9],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH, NCI109, 83.4, 0.4, 761.7/10 + 20.1],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH, FLASH_CFG, 85.4, 0.5, 425.3/10 + 142.6],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH, ANDROID_FCG, 90.0, 0.0, 3600.0],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, DD, 78.8, 0.5,
      2178.7/10 + 42.6],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, NCI1, 85.1, 0.1,
      2297.1/10 + 19.8],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, NCI109, 85.0, 0.1,
      2272.7/10 + 20.4],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, FLASH_CFG, 86.2, 0.4,
      666.8/10 + 143.0],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, ANDROID_FCG, 90.0, 0.0, 3600.0],
     [GRAPHLET_KERNEL_3, DD, 55.1, 4.2, 29.1/10 + 21.1],
     [GRAPHLET_KERNEL_3, NCI1, 54.7, 2.0, 89.7/10 + 4.5],
     [GRAPHLET_KERNEL_3, NCI109, 53.3, 1.7, 94.5/10 + 4.6],
     [GRAPHLET_KERNEL_3, FLASH_CFG, 65.1, 4.4, 47.0/10 + 38.7],
     [GRAPHLET_KERNEL_3, ANDROID_FCG, 90.0, 0.0, 3600.0],
     [GRAPHLET_KERNEL_4, DD, 46.6, 3.3, 38.3/10 + 311.0],
     [GRAPHLET_KERNEL_4, NCI1, 51.1, 1.8, 132.0/10 + 22.7],
     [GRAPHLET_KERNEL_4, NCI109, 54.3, 1.1, 132.2/10 + 22.9],
     [GRAPHLET_KERNEL_4, FLASH_CFG, 66.1, 4.1, 72.3/10 + 125.1],
     [GRAPHLET_KERNEL_4, ANDROID_FCG, 90.0, 0.0, 3600.0],
     [RANDOM_WALK_KERNEL, DD, 73.8, 0.2, 28.5/10 + 37552.2],
     [RANDOM_WALK_KERNEL, NCI1, 55.8, 0.9, 110.1/10 + 9604.3],
     [RANDOM_WALK_KERNEL, NCI109, 55.2, 1.1, 115.7/10 + 9578.5],
     [RANDOM_WALK_KERNEL, FLASH_CFG, 0.0, 0.0, 2*24*60*60],
     [RANDOM_WALK_KERNEL, ANDROID_FCG, 90.0, 0.0, 3600.0],
     [EIGEN_KERNEL, DD, 75.9, 0.5, 1554.6/10 + 3117.9],
     [EIGEN_KERNEL, NCI1, 64.4, 0.2, 929.7/10 + 898.1],
     [EIGEN_KERNEL, NCI109, 64.5, 0.3, 964.3/10 + 879.0],
     [EIGEN_KERNEL, FLASH_CFG, 79.7, 2.6, 2596.1/10 + 3387.9],
     [EIGEN_KERNEL, ANDROID_FCG, 90.0, 0.0, 3600.0]])
     
      
# order according to the sequence of the embeddings in the data matrices
COLORS = ['#00008F', '#0020FF', '#00AFFF', '#40FFBF', '#CFFF30', '#FF9F00',
          '#FF1000', '#800000']
          
# #87FF77 can be used between #40FFBF and #CFFF30
         

for dataset_type, mode in itertools.product(DATASET_TYPES, MODES):         
    
    data = DATA_SD if dataset_type == SMALL else DATA_LD
    
    figsize = (5.58, 3) if mode == SCORES else (5.8, 3)
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)
    
    
    # embeddings ordered according to their sequence in the data matrices
    idxs = np.unique(data[:,0], return_index = True)[1]
    embeddings = [data[:,0][idx] for idx in sorted(idxs)]
    
    # datasets ordered according to their sequence in the data matrices
    idxs = np.unique(data[:,1], return_index = True)[1]
    datasets = [data[:,1][idx] for idx in sorted(idxs)]
    #scores = data[:,2]
    #std_devs = data[:,3]
    #runtimes = data[:,4]
    
    space = 2/(len(embeddings) + 2)
    width = (1 - space) / len(embeddings)
    
    for i in xrange(len(embeddings)):
        positions = []

        scores = []
        std_devs = []

        runtimes = []
        
#        for j, dataset in enumerate(datasets):
        for j in xrange(len(datasets)):
            position = j + 1 - (1 - space)/2 + i * width
            positions.append(position)                
            
            if mode == SCORES:   
                score, std_dev = tuple(
                    data[i * len(datasets) + j][2:4].astype(float))
                
                scores.append(score)
                std_devs.append(std_dev)
            else:
                # mode == RUNTIMES
                runtime = float(data[i * len(datasets) + j][4])
                runtimes.append(runtime)

        
        if mode == SCORES:                      
            ax.bar(positions, scores, width = width, color = COLORS[i],
                   yerr = std_devs, ecolor = 'black',
                   label = EMBEDDING_ABBRVS[embeddings[i]])
        else:
            # mode == RUNTIMES
            ax.bar(positions, runtimes, width = width, color = COLORS[i],
                   label = EMBEDDING_ABBRVS[embeddings[i]])
               
    
    ax.set_xlim(0.5 - space/2, len(datasets) + 0.5 + space/2)
    
    x = range(1, len(datasets) + 1)
    #plt.xticks(x, datasets, fontsize = FONT_SIZE)
    plt.xticks(x, datasets)
    
    # Drawing the canvas causes the labels to be positioned, which is necessary
    # in order to get their values
    fig.canvas.draw()
    xtick_labels = [item.get_text() for item in ax.get_xticklabels()]
    
    if mode == SCORES:
        ax.set_ylim([0, 120])
        
        y = np.linspace(0, 100, 11).astype(int)
        #plt.yticks(y, datasets, fontsize = FONT_SIZE)
        
        plt.tick_params(axis = 'x', which = 'both', bottom = 'off', top = 'off')
            
        plt.yticks(y)
    else:
        # mode == RUNTIMES
        plt.yscale('log')
        
        ax.set_ylim([1, 1.5*24*60*60])
        
        plt.tick_params(axis = 'x', which = 'both', bottom = 'off', top = 'off')
        
        plt.tick_params(axis = 'y', which = 'minor', left = 'off', right = 'off')
        
        y = range(1, 10, 1) + range(10, 60, 10) + range(60, 10*60, 60) \
            + [10*60, 20*60, 30*60] + range(60*60, 12*60*60, 60*60) \
            + [12*60*60, 24*60*60]
            
        y_labels = ['1 sec'] + ['']*8 + ['10 sec'] + ['']*4 + ['1 min'] \
                   + ['']*8 + ['10 min', '', '30 min', '1 h'] + ['']*10 \
                   + ['12 h', '1 day']

        plt.yticks(y, y_labels)
    
    
    # plot legend
    handles, labels = ax.get_legend_handles_labels()

    if mode == SCORES:
        if dataset_type == SMALL:
#            loc = None
            loc = 9
            ncol = 5
#            bbox_to_anchor = (0.6, 1.0)
        else:
            # dataset_type == LARGE
            loc = 9
            ncol = 5
    else:
#        mode == RUNTIMES
        if dataset_type == SMALL:
            loc = 9
            ncol = 5
        else:
            # dataset_type == LARGE
            loc = 9
            ncol = 5
    
    if loc:
        ax.legend(handles, labels, loc = loc, ncol = ncol,
                  prop = {'size': LEGEND_FONT_SIZE})
#    else:
#        # location specified by bbox_to_anchor
#        ax.legend(handles, labels, bbox_to_anchor = bbox_to_anchor, ncol = ncol,
#                  prop = {'size': LEGEND_FONT_SIZE})
    
    plt.tight_layout(0.5)
    
    output_file_name = 'figure_' + dataset_type + '_' + mode
    plt.savefig(output_file_name + '.pdf')
    plt.savefig(join(TARGET_PATH, output_file_name + '.pgf'))




#data = DATA_SD
#x = data[np.logical_and(data[:,0] == 'neighborhood_hash', data[:,1] == 'MUTAG')]


