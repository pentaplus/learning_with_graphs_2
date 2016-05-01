"""
Plot bar charts representing the results of the embedding methods.

The plots show the classification accuracy and the runtime as well as
the ratio classification accuracy per runtime for the respective
methods.
"""

from __future__ import division

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-04-30"


import inspect
import itertools
import matplotlib as mpl
import numpy as np
import sys

from itertools import chain
from matplotlib.ticker import MultipleLocator
from os.path import abspath, dirname, join

# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))


#==============================================================================
# constants
#==============================================================================
TARGET_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', '..', 'tex', 'figures')

# embeddings
WL = 'weisfeiler_lehman'
NH = 'neighborhood_hash'
CSNH = 'count_sensitive_neighborhood_hash'
CSNH_ALL = 'count_sensitive_neighborhood_hash_all_iter'
GK_3 = 'graphlet_kernel_3'
GK_4 = 'graphlet_kernel_4'
SP = 'shortest_path_kernel'
RW = 'random_walk_kernel'
EGK = 'eigen_kernel'

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
EFFICIENCY_RATIOS = 'efficiency_ratios'


EMBEDDING_ABBRVS = {
    WL: 'WL',
    NH: 'NH',
    CSNH: 'CSNH',
    CSNH_ALL: 'CSNH ALL',
    GK_3: '3-GK',
    GK_4: '4-GK',
    RW: 'RW',
    SP: 'SP',
    EGK: 'EGK'}
    
    
FONT_SIZE = 10
LEGEND_FONT_SIZE = 6

    
DATASET_TYPES = [SMALL, LARGE]

# !!
#MODES = [SCORES, RUNTIMES]
MODES = [EFFICIENCY_RATIOS]


mpl.use("pgf")
pgf_with_rc_fonts = {
    "font.family": "Minion Pro",
    "pgf.texsystem": "xelatex",
    "text.fontsize": FONT_SIZE
}
mpl.rcParams.update(pgf_with_rc_fonts)


# Module pyplot must be imported after the specification of the RC parameters.
import matplotlib.pyplot as plt


# The data matrices DATA_SD and DATA_LD have the following columns:
# embedding name, dataset, score, standard deviation, runtime (in seconds)
DATA_SD = np.array(
    [[WL, MUTAG, 91.3, 0.8, 110.3/10 + 0.6],
     [WL, PTC_MR, 64.6, 1.1, 425.8/10 + 1.2],
     [WL, ENZYMES, 60.7, 1.2, 998.5/10 + 2.6],
     [NH, MUTAG, 88.9, 0.8, 108.7/10 + 0.5],
     [NH, PTC_MR, 66.4, 1.0, 435.7/10 + 1.3],
     [NH, ENZYMES, 46.8, 1.0, 1325.8/10 + 2.1],
     [CSNH, MUTAG, 91.1, 1.0, 109.7/10 + 0.6],
     [CSNH, PTC_MR, 65.0, 1.0, 288.5/10 + 1.3],
     [CSNH, ENZYMES, 56.2, 1.0, 598.5/10 + 2.8],
     [CSNH_ALL, MUTAG, 91.4, 0.8, 112.0/10 + 0.6],
     [CSNH_ALL, PTC_MR, 65.6, 1.3, 397.9/10 + 1.3],
     [CSNH_ALL, ENZYMES, 61.4, 1.3, 1036.8/10 + 3.1],
     [GK_3, MUTAG, 85.8, 2.1, 929.7/10 + 0.2],
     [GK_3, PTC_MR, 55.3, 1.0, 3988.8/10 + 0.3],
     [GK_3, ENZYMES, 19.6, 1.9, 31158.3/10 + 1.1],
     [GK_4, MUTAG, 86.4, 1.0, 1370.8/10 + 0.7],
     [GK_4, PTC_MR, 54.4, 1.7, 4668.8/10 + 1.5],
     [GK_4, ENZYMES, 17.6, 1.4, 41190.1/10 + 8.6],
     [SP, MUTAG, 86.1, 1.3, 34.3/10 + 0.3],
     [SP, PTC_MR, 56.3, 1.6, 1101.0/10 + 0.8],
     [SP, ENZYMES, 25.3, 2.5, 14979.0/10 + 2.4],
     [RW, MUTAG,  83.4, 2.7, 15.4/10 + 11.2],
     [RW, PTC_MR, 53.5, 1.4, 55.7/10 + 57.9],
     [RW, ENZYMES, 14.8, 0.8, 33.1/10 + 231.4],
     [EGK, MUTAG, 88.6, 0.5, 106.6/10 + 14.8],
     [EGK, PTC_MR, 61.9, 1.0, 130.4/10 + 62.5],
     [EGK, ENZYMES, 25.7, 1.1, 803.2/10 + 99.6]])
     
# The scores and runtimes for ANDROID_FCG are composed from two runs,
# where each run encompasses 5 repetitions (the results for the second
# run can be found in the folder "results_final\Android FCG logs")
DATA_LD = np.array(
    [[WL, DD, 79.1, 0.5, 2170.2/10 + 31.9],
     [WL, NCI1, 86.0, 0.2, 2603.8/10 + 17.6],
     [WL, NCI109, 86.3, 0.1, 2636.1/10 + 17.4],
     [WL, FLASH_CFG, 85.9, 0.4, 676.3/10 + 129.2],
     [WL, ANDROID_FCG, 93.9, 0.1, (105315.5 + 104838.5)/10 + 12295.0],
     [NH, DD, 76.8, 0.9, 253.7/10 + 22.2],
     [NH, NCI1, 79.1, 0.3, 655.6/10 + 16.7],
     [NH, NCI109, 79.1, 0.3, 684.1/10 + 15.3],
     [NH, FLASH_CFG, 83.9, 2.4, 405.2/10 + 117.8],
     [NH, ANDROID_FCG, 94.6, 0.1, (27575.8 + 28380.9)/10 + 7507.8],
     [CSNH, DD, 76.3, 1.1, 263.5/10 + 37.9],
     [CSNH, NCI1, 83.9, 0.2, 742.0/10 + 19.9],
     [CSNH, NCI109, 83.4, 0.4, 761.7/10 + 20.1],
     [CSNH, FLASH_CFG, 85.4, 0.5, 425.3/10 + 142.6],
     [CSNH, ANDROID_FCG, 94.6, 0.1, (31028.3 + 29955.7)/10 + 7779.1], 
     [CSNH_ALL, DD, 78.5, 1.0, 2265.1/10 + 42.9],
     [CSNH_ALL, NCI1, 85.1, 0.1, 2297.1/10 + 19.8],
     [CSNH_ALL, NCI109, 85.0, 0.1, 2272.7/10 + 20.4],
     [CSNH_ALL, FLASH_CFG, 86.2, 0.4, 666.8/10 + 143.0],
     [CSNH_ALL, ANDROID_FCG, 93.6, 0.6, (106281.2 + 103615.5)/10 + 12801.0],
     [GK_3, DD, 55.1, 4.2, 29.1/10 + 21.1],
     [GK_3, NCI1, 54.7, 2.0, 89.7/10 + 4.5],
     [GK_3, NCI109, 53.3, 1.7, 94.5/10 + 4.6],
     [GK_3, FLASH_CFG, 65.1, 4.4, 47.0/10 + 38.7],
     [GK_3, ANDROID_FCG, 62.1, 6.5, (182.5 + 188.9)/10 + 608.8],
     [GK_4, DD, 46.6, 3.3, 38.3/10 + 311.0],
     [GK_4, NCI1, 51.1, 1.8, 132.0/10 + 22.7],
     [GK_4, NCI109, 54.3, 1.1, 132.2/10 + 22.9],
     [GK_4, FLASH_CFG, 66.1, 4.1, 72.3/10 + 125.1],
     [GK_4, ANDROID_FCG, 64.3, 6.9, (284.2 + 288.3)/10 + 7386.7],
     [SP, DD, 67.4, 2.0, 52.3/10 + 5269.1],
     [SP, NCI1, 65.1, 0.9, 167.6/10 + 12.4],
     [SP, NCI109, 64.5, 0.9, 165.4/10 + 12.2],
     [SP, FLASH_CFG, 0.0, 0.0, 2*24*60*60],
     [SP, ANDROID_FCG, 0.0, 0.0, 2*24*60*60],
     [RW, DD, 73.8, 0.2, 28.5/10 + 37552.2],
     [RW, NCI1, 55.8, 0.9, 110.1/10 + 9604.3],
     [RW, NCI109, 55.2, 1.1, 115.7/10 + 9578.5],
     [RW, FLASH_CFG, 0.0, 0.0, 2*24*60*60],
     [RW, ANDROID_FCG, 0.0, 0.0, 2*24*60*60],
     [EGK, DD, 75.9, 0.5, 1554.6/10 + 3117.9],
     [EGK, NCI1, 64.4, 0.2, 929.7/10 + 898.1],
     [EGK, NCI109, 64.5, 0.3, 964.3/10 + 879.0],
     [EGK, FLASH_CFG, 79.7, 2.6, 2596.1/10 + 3387.9],
     [EGK, ANDROID_FCG, 88.5, 2.7, (35149.8 + 34944.0)/10 + 59154.0]])
     
      
# order according to the sequence of the embeddings in the data matrices
COLORS = ['#00008F', '#0020FF', '#00AFFF', '#40FFBF', '#87FF77', '#CFFF30',
          '#FF9F00', '#FF1000', '#800000']
          

for dataset_type, mode in itertools.product(DATASET_TYPES, MODES):         
    data = DATA_SD if dataset_type == SMALL else DATA_LD
    
    figsize = (5.58, 3) if mode in [SCORES, EFFICIENCY_RATIOS] else (5.8, 3)
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)
    
    
    # embeddings ordered according to their sequence in the data matrices
    idxs = np.unique(data[:, 0], return_index = True)[1]
    embeddings = [data[:, 0][idx] for idx in sorted(idxs)]
    
    # datasets ordered according to their sequence in the data matrices
    idxs = np.unique(data[:, 1], return_index = True)[1]
    datasets = [data[:, 1][idx] for idx in sorted(idxs)]
    
    space = 2/(len(embeddings) + 2)
    width = (1 - space) / len(embeddings)
    
    for i in xrange(len(embeddings)):
        positions = []

        scores = []
        std_devs = []

        runtimes = []
        
        eff_ratios = []
        
        for j in xrange(len(datasets)):
            position = j + 1 - (1 - space)/2 + i * width
            positions.append(position)                
            
            if mode == SCORES:   
                score, std_dev = tuple(
                    data[i * len(datasets) + j][2:4].astype(float))
                
                scores.append(score)
                std_devs.append(std_dev)
            elif mode == RUNTIMES:
                runtime = float(data[i * len(datasets) + j][4])
                runtimes.append(runtime)
            else:
                # mode == EFFICIENCY_RATIOS
                score, std_dev = tuple(
                    data[i * len(datasets) + j][2:4].astype(float))
                    
                runtime = float(data[i * len(datasets) + j][4])
                
                eff_ratio = score / runtime
                eff_ratios.append(eff_ratio)

        
        if mode == SCORES:                      
            ax.bar(positions, scores, width = width, color = COLORS[i],
                   yerr = std_devs, ecolor = 'black',
                   label = EMBEDDING_ABBRVS[embeddings[i]])
        elif mode == RUNTIMES:
            ax.bar(positions, runtimes, width = width, color = COLORS[i],
                   label = EMBEDDING_ABBRVS[embeddings[i]])
        else:
            # mode == EFFICIENCY_RATIOS
            ax.bar(positions, eff_ratios, width = width, color = COLORS[i],
                   label = EMBEDDING_ABBRVS[embeddings[i]])            
               
    
    ax.set_xlim(0.5 - space/2, len(datasets) + 0.5 + space/2)
    
    x = range(1, len(datasets) + 1)
    plt.xticks(x, datasets)
    
    # Drawing the canvas causes the labels to be positioned, which is necessary
    # in order to get their values
    fig.canvas.draw()
    xtick_labels = [item.get_text() for item in ax.get_xticklabels()]
    
    
    if mode == SCORES:
        major_locator = MultipleLocator(5)
        ax.yaxis.set_major_locator(major_locator)
    plt.grid(axis = 'y', color = '0.5', alpha = 0.5)
    
    if mode == SCORES:
        ax.set_ylim([0, 120])
        
        y = np.linspace(0, 100, 21).astype(int)
        
        plt.tick_params(axis = 'x', which = 'both', bottom = 'off',
                        top = 'off')
        
        y_labels = chain.from_iterable(zip(np.linspace(0, 100, 11).astype(int),
                                           11*['']))        
        
        plt.yticks(y, y_labels)
    elif mode == RUNTIMES:
        plt.yscale('log')
        
        ax.set_ylim([1, 1.5*24*60*60])
        
        plt.tick_params(axis = 'x', which = 'both', bottom = 'off',
                        top = 'off')
        
        plt.tick_params(axis = 'y', which = 'minor', left = 'off',
                        right = 'off')
        
        y = range(1, 10, 1) + range(10, 60, 10) + range(60, 10*60, 60) \
            + [10*60, 20*60, 30*60] + range(60*60, 12*60*60, 60*60) \
            + [12*60*60, 24*60*60]
            
        y_labels = ['1 sec'] + 8*[''] + ['10 sec'] + 4*[''] + ['1 min'] \
                   + 8*[''] + ['10 min', '', '30 min', '1 h'] + 10*[''] \
                   + ['12 h', '1 day']

        plt.yticks(y, y_labels)
    else:
        # mode == EFFICIENCY_RATIOS
        plt.yscale('log')
        
        if dataset_type == SMALL:
            ax.set_ylim([0, 150])
        else:
            # dataset_type == LARGE
            ax.set_ylim([0, 15])
        
        plt.tick_params(axis = 'x', which = 'both', bottom = 'off',
                        top = 'off')
        
        plt.grid(axis = 'y', which = 'minor', color = '0.5', alpha = 0.5)
        
    
    # plot legend
    handles, labels = ax.get_legend_handles_labels()
    
    ax.legend(handles, labels, loc = 9, ncol = 5,
              prop = {'size': LEGEND_FONT_SIZE})
    
    plt.tight_layout(0.5)
    
    output_file_name = 'figure_' + dataset_type + '_' + mode
    plt.savefig(output_file_name + '.pdf')
    plt.savefig(join(TARGET_PATH, output_file_name + '.pgf'))

