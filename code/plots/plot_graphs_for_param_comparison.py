"""
Plot graphs for comparing the performance for various parameters.

The plots show the classification accuracy, the runtime of the
respective methods and the length of the feature vectors.
"""

from __future__ import division

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-04-03"


import inspect
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
WL = 'weisfeiler_lehman'
NH = 'neighborhood_hash'
CSNH = 'count_sens_neigh_hash'
CSNH_ALL = 'count_sens_neigh_hash_all_iter'
GK_3 = 'graphlet_kernel_3'
GK_4 = 'graphlet_kernel_4'
RW = 'random_walk_kernel'
EK = 'eigen_kernel'

DISPLAY_NAME_OF_EMBEDDING = {
    WL: 'Weisfeiler-Lehman subtree kernel',
    CSNH: 'Count-sensitive neighborhood hash kernel',
    CSNH_ALL: 'Count-sensitive neighborhood hash kernel (all it.)',
    EK: 'Eigen graph kernel'
}

# datasets
MUTAG = 'MUTAG'
PTC_MR = 'PTC(MR)'
ENZYMES = 'ENZYMES'
DD = 'DD'
NCI1 = 'NCI1'
NCI109 = 'NCI109'
FLASH_CFG = 'FLASH_CFG'
ANDROID_FCG = 'ANDROID_FCG'

SMALL = 'small'
LARGE = 'large'

SCORES = 'scores'
RUNTIMES = 'runtimes'
FEATURES_COUNT = 'features_count'

H_0 = 0
H_1 = 1
H_2 = 2
H_3 = 3
H_4 = 4
H_5 = 5

R_1_6 = 1/6
R_2_6 = 2/6
R_3_6 = 3/6
R_4_6 = 4/6
R_5_6 = 5/6
R_6_6 = 6/6


EMBEDDING_ABBRVS = {
    WL: 'WL',
    NH: 'NH',
    CSNH: 'CSNH',
    CSNH_ALL: 'CSNH ALL',
    GK_3: '3-GK',
    GK_4: '4-GK',
    RW: 'RW',
    EK: 'EGK'}
    
    
FONT_SIZE = 8
LEGEND_FONT_SIZE = 6
MARKER_SIZE = 4


MODES = [SCORES, RUNTIMES, FEATURES_COUNT]


mpl.use("pgf")
pgf_with_rc_fonts = {
    "font.family": "Minion Pro",
    "pgf.texsystem": "xelatex",
    "text.fontsize": FONT_SIZE
}
mpl.rcParams.update(pgf_with_rc_fonts)


# The module pyplot must be imported after the specification of the RC parameters
import matplotlib.pyplot as plt


def is_completely_integer(x):
    return (np.mod(x, 1) == 0).all()

# The data matrix DATA has the following columns:
# embedding name, dataset, parameter, score, standard deviation,
# runtime (in seconds, for ANDROID FCG in hours), length of feature vectors
#
# The scores and runtimes for ANDROID_FCG are composed from two runs,
# where each run encompasses 5 repetitions (the results for the second
# run can be found in the folder "results_final\Android FCG logs")
DATA = np.array(
    [[WL, MUTAG, H_0, 85.5, 0.9, 15.7/10 + 0.1, 7],
     [WL, MUTAG, H_1, 87.3, 1.3, 15.6/10 + 0.2, 40],
     [WL, MUTAG, H_2, 85.9, 0.9, 15.7/10 + 0.3, 214],
     [WL, MUTAG, H_3, 84.7, 1.5, 15.7/10 + 0.4, 786],
     [WL, MUTAG, H_4, 84.1, 1.4, 16.0/10 + 0.5, 1983],
     [WL, MUTAG, H_5, 84.0, 1.5, 18.8/10 + 0.6, 3749],
     [WL, PTC_MR, H_0, 56.5, 2.2, 59.0/10 + 0.2, 19],
     [WL, PTC_MR, H_1, 57.0, 1.8, 150.8/10 + 0.4, 179],
     [WL, PTC_MR, H_2, 58.8, 1.8, 33.4/10 + 0.6, 1217],
     [WL, PTC_MR, H_3, 55.8, 2.2, 33.6/10 + 0.8, 3841],
     [WL, PTC_MR, H_4, 56.4, 2.6, 40.0/10 + 1.0, 7841],
     [WL, PTC_MR, H_5, 57.5, 1.2, 42.1/10 + 1.2, 12613],
     [WL, ENZYMES, H_0, 39.3, 2.4, 174.2/10 + 0.3, 3],
     [WL, ENZYMES, H_1, 60.6, 1.0, 190.9/10 + 0.7, 232],
     [WL, ENZYMES, H_2, 44.8, 3.4, 79.1/10 + 1.1, 10646],
     [WL, ENZYMES, H_3, 46.0, 2.8, 100.8/10 + 2.8, 25852],
     [WL, ENZYMES, H_4, 46.4, 2.3, 133.1/10 + 2.1, 41879],
     [WL, ENZYMES, H_5, 47.3, 3.4, 151.4/10 + 2.6, 58327],
     [WL, DD, H_0, 76.8, 1.3, 52.6/10 + 2.6, 82],
     [WL, DD, H_1, 78.4, 0.4, 392.9/10 + 7.9, 253231],
     [WL, DD, H_2, 77.4, 0.3, 343.3/10 + 13.3, 586248],
     [WL, DD, H_3, 76.2, 0.3, 356.1/10 + 18.8, 920338],
     [WL, DD, H_4, 76.0, 0.4, 373.0/10 + 25.4, 1254664],
     [WL, DD, H_5, 75.4, 0.3, 385.3/10 + 31.3, 1589062],
     [WL, NCI1, H_0, 64.4, 0.7, 95.4/10 + 1.8, 37],
     [WL, NCI1, H_1, 74.0, 0.3, 142.2/10 + 4.8, 303],
     [WL, NCI1, H_2, 81.7, 0.1, 207.7/10 + 8.1, 4335],
     [WL, NCI1, H_3, 84.9, 0.2, 345.4/10 + 11.3, 27257],
     [WL, NCI1, H_4, 85.6, 0.2, 519.2/10 + 14.7, 71739],
     [WL, NCI1, H_5, 85.7, 0.2, 700.0/10 + 17.5, 130661],
     [WL, NCI109, H_0, 63.5, 0.7, 96.2/10 + 1.8, 38],
     [WL, NCI109, H_1, 73.0, 0.4, 149.1/10 + 5.1, 294],
     [WL, NCI109, H_2, 81.6, 0.3, 203.2/10 + 8.1, 4365],
     [WL, NCI109, H_3, 84.9, 0.2, 345.2/10 + 10.9, 27749],
     [WL, NCI109, H_4, 85.9, 0.3, 516.8/10 + 14.2, 72767],
     [WL, NCI109, H_5, 86.0, 0.3, 690.3/10 + 17.3, 132194],
     [WL, FLASH_CFG, H_0, 74.2, 3.4, 44.2/10 + 12.1, 4],
     [WL, FLASH_CFG, H_1, 77.1, 3.2, 54.2/10 + 36.1, 71],
     [WL, FLASH_CFG, H_2, 82.7, 0.8, 63.7/10 + 58.4, 583],
     [WL, FLASH_CFG, H_3, 85.4, 0.9, 78.0/10 + 81.4, 2158],
     [WL, FLASH_CFG, H_4, 85.3, 0.8, 121.3/10 + 104.7, 7174],
     [WL, FLASH_CFG, H_5, 85.5, 0.8, 191.5/10 + 129.2, 21019],
     [WL, ANDROID_FCG, H_0, 93.3, 0.5, ((1529.5 + 1655.5)/10 + 278.9)/3600, 4530],
     [WL, ANDROID_FCG, H_1, 93.7, 0.3, ((5735.4 + 6149.4)/10 + 1073.5)/3600, 807181],
     [WL, ANDROID_FCG, H_2, 93.7, 0.2, ((11029.5 + 11857.0)/10 + 2508.2)/3600, 1953948],
     [WL, ANDROID_FCG, H_3, 93.1, 1.1, ((16799.5 + 18079.0)/10 + 4806.8)/3600, 3175014],
     [WL, ANDROID_FCG, H_4, 93.5, 0.4, ((23409.4 + 24538.1)/10 + 7998.7)/3600, 4418993],
     [WL, ANDROID_FCG, H_5, 93.3, 0.4, ((32145.7 + 32766.4)/10 + 12247.7)/3600, 5673882],
     [CSNH, MUTAG, H_0, 85.3, 0.8, 16.2/10 + 0.1, 7],
     [CSNH, MUTAG, H_1, 87.3, 1.6, 15.8/10 + 0.2, 29],
     [CSNH, MUTAG, H_2, 86.1, 0.9, 15.8/10 + 0.3, 117],
     [CSNH, MUTAG, H_3, 86.7, 1.1, 15.9/10 + 0.4, 424],
     [CSNH, MUTAG, H_4, 83.9, 1.0, 16.0/10 + 0.5, 912],
     [CSNH, MUTAG, H_5, 84.1, 1.4, 16.0/10 + 0.6, 1447],
     [CSNH, PTC_MR, H_0, 56.4, 1.4, 59.6/10 + 0.2, 19],
     [CSNH, PTC_MR, H_1, 60.1, 1.8, 87.8/10 + 0.4, 149],
     [CSNH, PTC_MR, H_2, 57.7, 2.8, 28.7/10 + 0.6, 758],
     [CSNH, PTC_MR, H_3, 55.9, 1.9, 22.5/10 + 0.8, 2066],
     [CSNH, PTC_MR, H_4, 55.1, 0.8, 18.5/10 + 1.0, 3271],
     [CSNH, PTC_MR, H_5, 54.6, 1.4, 21.2/10 + 1.2, 4336],
     [CSNH, ENZYMES, H_0, 40.6, 1.5, 167.0/10 + 0.3, 3],
     [CSNH, ENZYMES, H_1, 55.9, 2.0, 110.8/10 + 0.8, 196],
     [CSNH, ENZYMES, H_2, 46.0, 1.2, 69.7/10 + 1.2, 5406],
     [CSNH, ENZYMES, H_3, 39.5, 2.7, 57.7/10 + 1.8, 11452],
     [CSNH, ENZYMES, H_4, 20.0, 2.5, 52.1/10 + 2.2, 13543],
     [CSNH, ENZYMES, H_5, 32.2, 2.7, 51.4/10 + 2.8, 14239],
     [CSNH, DD, H_0, 76.8, 1.1, 51.3/10 + 2.6, 82],
     [CSNH, DD, H_1, 71.8, 0.4, 44.6/10 + 9.5, 164726],
     [CSNH, DD, H_2, 51.3, 0.4, 31.3/10 + 16.3, 301015],
     [CSNH, DD, H_3, 43.0, 0.2, 31.5/10 + 23.2, 322998],
     [CSNH, DD, H_4, 42.6, 0.3, 31.4/10 + 30.1, 324659],
     [CSNH, DD, H_5, 42.0, 0.1, 31.6/10 + 37.0, 324858],
     [CSNH, NCI1, H_0, 64.8, 0.4, 95.8/10 + 2.7, 37],
     [CSNH, NCI1, H_1, 73.6, 0.4, 115.6/10 + 6.3, 230],
     [CSNH, NCI1, H_2, 79.1, 0.2, 94.7/10 + 9.5, 1560],
     [CSNH, NCI1, H_3, 81.9, 0.1, 90.2/10 + 13.1, 8343],
     [CSNH, NCI1, H_4, 83.5, 0.3, 113.4/10 + 16.8, 17567],
     [CSNH, NCI1, H_5, 83.1, 0.2, 126.7/10 + 19.7, 28656],
     [CSNH, NCI109, H_0, 63.7, 0.3, 100.4/10 + 2.6, 38],
     [CSNH, NCI109, H_1, 72.7, 0.2, 96.1/10 + 5.6, 220],
     [CSNH, NCI109, H_2, 77.7, 0.2, 94.4/10 + 9.2, 1566],
     [CSNH, NCI109, H_3, 81.1, 0.2, 89.9/10 + 12.8, 8407],
     [CSNH, NCI109, H_4, 82.7, 0.2, 108.5/10 + 16.5, 18031],
     [CSNH, NCI109, H_5, 83.0, 0.3, 140.7/10 + 19.9, 28322],
     [CSNH, FLASH_CFG, H_0, 76.0, 3.2, 49.2/10 + 12.3, 4],
     [CSNH, FLASH_CFG, H_1, 76.6, 2.2, 54.4/10 + 38.4, 67],
     [CSNH, FLASH_CFG, H_2, 83.1, 0.6, 56.5/10 + 64.2, 510],
     [CSNH, FLASH_CFG, H_3, 84.4, 0.5, 57.6/10 + 90.6, 1441],
     [CSNH, FLASH_CFG, H_4, 84.4, 0.5, 67.0/10 + 116.7, 4360],
     [CSNH, FLASH_CFG, H_5, 85.0, 0.3, 79.5/10 + 142.5, 9628],
     [CSNH, ANDROID_FCG, H_0, 92.5, 1.0, ((1523.2 + 1503.8)/10 + 1640.3)/3600, 4530],
     [CSNH, ANDROID_FCG, H_1, 94.4, 0.1, ((4410.1 + 4296.2)/10 + 2435.5)/3600, 742522],
     [CSNH, ANDROID_FCG, H_2, 94.6, 0.1, ((4958.5 + 4817.5)/10 + 3513.3)/3600, 996877],
     [CSNH, ANDROID_FCG, H_3, 94.5, 0.0, ((5153.8 + 5029.5)/10 + 4816.1)/3600, 1161715],
     [CSNH, ANDROID_FCG, H_4, 94.4, 0.1, ((4862.7 + 4745.9)/10 + 6269.6)/3600, 1104893],
     [CSNH, ANDROID_FCG, H_5, 94.5, 0.1, ((5067.1 + 4918.3)/10 + 7755.4)/3600, 1182199],
#     [CSNH_ALL, MUTAG, H_0, 85.4, 0.6, 16.1/10 + 0.1, 7],
#     [CSNH_ALL, MUTAG, H_1, 87.8, 1.9, 16.1/10 + 0.2, 36],
#     [CSNH_ALL, MUTAG, H_2, 85.8, 1.0, 16.0/10 + 0.3, 153],
#     [CSNH_ALL, MUTAG, H_3, 84.8, 0.7, 16.1/10 + 0.4, 568],
#     [CSNH_ALL, MUTAG, H_4, 83.8, 1.5, 16.2/10 + 0.5, 1456],
#     [CSNH_ALL, MUTAG, H_5, 85.3, 0.9, 16.1/10 + 0.6, 2854],
#     [CSNH_ALL, PTC_MR, H_0, 55.5, 1.3, 60.4/10 + 0.1, 19],
#     [CSNH_ALL, PTC_MR, H_1, 57.1, 2.4, 149.9/10 + 0.4, 168],
#     [CSNH_ALL, PTC_MR, H_2, 58.4, 1.7, 31.6/10 + 0.6, 928],
#     [CSNH_ALL, PTC_MR, H_3, 57.1, 1.4, 31.8/10 + 0.8, 2985],
#     [CSNH_ALL, PTC_MR, H_4, 57.2, 2.5, 32.2/10 + 1.0, 6122],
#     [CSNH_ALL, PTC_MR, H_5, 58.4, 1.7, 41.3/10 + 1.3, 10035],
#     [CSNH_ALL, ENZYMES, H_0, 40.2, 1.9, 165.3/10 + 0.3, 3],
#     [CSNH_ALL, ENZYMES, H_1, 61.3, 0.8, 192.8/10 + 0.9, 199],
#     [CSNH_ALL, ENZYMES, H_2, 44.7, 2.2, 87.6/10 + 1.5, 5427],
#     [CSNH_ALL, ENZYMES, H_3, 44.9, 2.9, 107.9/10 + 2.0, 15838],
#     [CSNH_ALL, ENZYMES, H_4, 48.8, 3.3, 129.3/10 + 2.5, 26006],
#     [CSNH_ALL, ENZYMES, H_5, 48.6, 1.5, 148.9/10 + 3.1, 34453],
#     [CSNH_ALL, DD, H_0, 77.0, 0.7, 46.9/10 + 2.1, 82],
#     [CSNH_ALL, DD, H_1, 77.7, 0.2, 403.0/10 + 10.9, 164450],
#     [CSNH_ALL, DD, H_2, 77.1, 0.3, 371.2/10 + 18.2, 462671],
#     [CSNH_ALL, DD, H_3, 76.5, 0.3, 391.2/10 + 25.8, 776554],
#     [CSNH_ALL, DD, H_4, 75.9, 0.4, 401.9/10 + 33.5, 1086010],
#     [CSNH_ALL, DD, H_5, 75.5, 0.2, 431.1/10 + 41.3, 1389887],
#     [CSNH_ALL, NCI1, H_0, 64.3, 0.5, 94.8/10 + 2.0, 37],
#     [CSNH_ALL, NCI1, H_1, 73.4, 0.4, 153.9/10 + 5.6, 266],
#     [CSNH_ALL, NCI1, H_2, 80.3, 0.2, 218.2/10 + 8.7, 1800],
#     [CSNH_ALL, NCI1, H_3, 83.4, 0.2, 291.0/10 + 12.4, 9909],
#     [CSNH_ALL, NCI1, H_4, 84.4, 0.1, 436.0/10 + 15.7, 24675],
#     [CSNH_ALL, NCI1, H_5, 85.0, 0.2, 602.5/10 + 19.6, 42671],
#     [CSNH_ALL, NCI109, H_0, 63.7, 0.3, 99.3/10 + 2.3, 38],
#     [CSNH_ALL, NCI109, H_1, 72.5, 0.2, 153.8/10 + 6.0, 258],
#     [CSNH_ALL, NCI109, H_2, 78.6, 0.3, 212.7/10 + 9.4, 1820],
#     [CSNH_ALL, NCI109, H_3, 82.8, 0.2, 294.8/10 + 13.0, 9791],
#     [CSNH_ALL, NCI109, H_4, 84.3, 0.3, 438.5/10 + 16.5, 24468],
#     [CSNH_ALL, NCI109, H_5, 84.7, 0.1, 573.4/10 + 20.2, 42006],
#     [CSNH_ALL, FLASH_CFG, H_0, 75.0, 2.9, 49.8/10 + 12.3, 4],
#     [CSNH_ALL, FLASH_CFG, H_1, 73.8, 3.9, 57.8/10 + 37.5, 71],
#     [CSNH_ALL, FLASH_CFG, H_2, 82.2, 0.8, 67.6/10 + 63.4, 580],
#     [CSNH_ALL, FLASH_CFG, H_3, 85.6, 0.6, 82.3/10 + 89.6, 1988],
#     [CSNH_ALL, FLASH_CFG, H_4, 85.5, 0.5, 118.7/10 + 114.7, 6198],
#     [CSNH_ALL, FLASH_CFG, H_5, 86.0, 0.2, 174.2/10 + 142.9, 14679],
#     [CSNH_ALL, ANDROID_FCG, H_0, 93.1, 0.7, ((1585.0 + 1510.3)/10 + 1699.1)/3600, 4530],
#     [CSNH_ALL, ANDROID_FCG, H_1, 93.8, 0.3, ((5810.0 + 5537.1)/10 + 2534.5)/3600, 746670],
#     [CSNH_ALL, ANDROID_FCG, H_2, 93.5, 0.6, ((11434.9 + 10918.3)/10 + 3946.7)/3600, 1698312],
#     [CSNH_ALL, ANDROID_FCG, H_3, 93.5, 0.4, ((17808.0 + 17211.8)/10 + 6107.2)/3600, 2740659],
#     [CSNH_ALL, ANDROID_FCG, H_4, 93.3, 0.6, ((23960.5 + 23222.7)/10 + 8987.9)/3600, 3663995],
#     [CSNH_ALL, ANDROID_FCG, H_5, 93.4, 0.3, ((31428.6 + 31720.8)/10 + 12731.0)/3600, 4587380],
     [EK, MUTAG, R_1_6, 85.4, 0.5, 15.1/10 + 3.3, 2],
     [EK, MUTAG, R_2_6, 87.2, 0.1, 15.2/10 + 7.3, 5],
     [EK, MUTAG, R_3_6, 87.2, 0.2, 15.1/10 + 10.6, 8],
     [EK, MUTAG, R_4_6, 86.3, 0.6, 15.3/10 + 13.0, 11],
     [EK, MUTAG, R_5_6, 84.4, 0.7, 15.4/10 + 14.5, 14],
     [EK, MUTAG, R_6_6, 84.2, 0.5, 15.2/10 + 15.3, 17],
     [EK, PTC_MR, R_1_6, 54.5, 0.8, 16.1/10 + 16.6, 4],
     [EK, PTC_MR, R_2_6, 57.6, 1.3, 16.4/10 + 29.9, 8],
     [EK, PTC_MR, R_3_6, 58.6, 0.5, 16.2/10 + 40.3, 12],
     [EK, PTC_MR, R_4_6, 57.3, 0.9, 19.4/10 + 49.1, 17],
     [EK, PTC_MR, R_5_6, 57.6, 1.0, 23.7/10 + 54.2, 21],
     [EK, PTC_MR, R_6_6, 58.0, 0.9, 23.5/10 + 57.9, 25],
     [EK, ENZYMES, R_1_6, 19.9, 1.0, 29.8/10 + 32.8, 5],
     [EK, ENZYMES, R_2_6, 18.6, 1.3, 41.5/10 + 53.8, 10],
     [EK, ENZYMES, R_3_6, 20.0, 0.9, 72.1/10 + 72.7, 16],
     [EK, ENZYMES, R_4_6, 21.5, 0.9, 124.0/10 + 83.5, 21],
     [EK, ENZYMES, R_5_6, 20.7, 1.0, 173.2/10 + 92.3, 27],
     [EK, ENZYMES, R_6_6, 22.3, 1.4, 209.4/10 + 97.2, 32],
     [EK, DD, R_1_6, 68.8, 1.3, (111.8/10 + 1012.6)/60, 47],
     [EK, DD, R_2_6, 72.1, 0.3, (167.3/10 + 1814.0)/60, 94],
     [EK, DD, R_3_6, 75.1, 0.2, (208.5/10 + 2383.6)/60, 142],
     [EK, DD, R_4_6, 75.2, 0.3, (245.1/10 + 2777.8)/60, 189],
     [EK, DD, R_5_6, 74.4, 0.3, (287.7/10 + 3050.7)/60, 236],
     [EK, DD, R_6_6, 74.0, 0.3, (310.6/10 + 3225.1)/60, 284],
     [EK, NCI1, R_1_6, 57.7, 0.2, (60.6/10 + 260.8)/60, 4],
     [EK, NCI1, R_2_6, 60.0, 0.2, (126.5/10 + 522.3)/60, 9],
     [EK, NCI1, R_3_6, 61.7, 0.1, (146.0/10 + 698.8)/60, 14],
     [EK, NCI1, R_4_6, 62.7, 0.4, (137.2/10 + 808.6)/60, 19],
     [EK, NCI1, R_5_6, 63.8, 0.2, (146.7/10 + 874.5)/60, 24],
     [EK, NCI1, R_6_6, 63.9, 0.2, (171.7/10 + 914.9)/60, 29],
     [EK, NCI109, R_1_6, 58.4, 0.1, (60.6/10 + 259.5)/60, 4],
     [EK, NCI109, R_2_6, 59.6, 0.1, (132.4/10 + 517.9)/60, 9],
     [EK, NCI109, R_3_6, 62.6, 0.2, (128.2/10 + 689.0)/60, 14],
     [EK, NCI109, R_4_6, 63.2, 0.4, (161.3/10 + 795.6)/60, 19],
     [EK, NCI109, R_5_6, 63.6, 0.5, (163.1/10 + 859.0)/60, 24],
     [EK, NCI109, R_6_6, 63.9, 0.1, (168.7/10 + 899.0)/60, 29],
     [EK, FLASH_CFG, R_1_6, 79.1, 3.0, (221.5/10 + 1685.2)/60, 174],
     [EK, FLASH_CFG, R_2_6, 79.8, 2.3, (320.4/10 + 2268.5)/60, 348],
     [EK, FLASH_CFG, R_3_6, 80.5, 3.0, (379.2/10 + 2646.6)/60, 522],
     [EK, FLASH_CFG, R_4_6, 82.0, 1.6, (431.2/10 + 2966.3)/60, 696],
     [EK, FLASH_CFG, R_5_6, 81.7, 1.7, (452.7/10 + 3215.6)/60, 870],
     [EK, FLASH_CFG, R_6_6, 81.3, 1.7, (466.4/10 + 3542.6)/60, 1044],
     [EK, ANDROID_FCG, R_1_6, 90.4, 2.3, ((2332.2 + 2368.8)/10 + 14951.6)/3600, 307],
     [EK, ANDROID_FCG, R_2_6, 87.4, 3.5, ((3806.3 + 3837.4)/10 + 28330.0)/3600, 615],
     [EK, ANDROID_FCG, R_3_6, 88.6, 1.8, ((4939.4 + 4959.0)/10 + 39367.0)/3600, 923],
     [EK, ANDROID_FCG, R_4_6, 87.4, 2.3, ((5694.9 + 5828.0)/10 + 48818.0)/3600, 1231],
     [EK, ANDROID_FCG, R_5_6, 89.4, 1.7, ((6468.5 + 6454.0)/10 + 56604.0)/3600, 1539],
     [EK, ANDROID_FCG, R_6_6, 88.2, 2.6, ((7168.6 + 7133.8)/10 + 62868.8)/3600, 1847]])
     
    
          
BLUE = '#3F3D99'
GREY = '#080808'
         

# embeddings ordered according to their sequence in the data matrix
idxs = np.unique(DATA[:,0], return_index = True)[1]
EMBEDDINGS = [DATA[:,0][idx] for idx in sorted(idxs)]

# datasets ordered according to their sequence in the data matrix
idxs = np.unique(DATA[:,1], return_index = True)[1]
DATASETS = [DATA[:,1][idx] for idx in sorted(idxs)]


figsize = (5.9, 6.32)


#for dataset in [MUTAG]:
for dataset in DATASETS:
    fig, axes_mat = plt.subplots(nrows = 3, ncols = 3, figsize = figsize)
     

    ax_of_mode_of_embedding = dict()
    for i, embedding in enumerate(EMBEDDINGS):
        ax_of_mode_of_embedding[embedding] = dict(zip(MODES, axes_mat[i]))
    
    for embedding in EMBEDDINGS:
        data_of_dataset_and_embedding = DATA[np.logical_and(
            DATA[:, 0] == embedding,
            DATA[:, 1] == dataset)]
        
        # sort data_of_dataset_and_embedding by parameter 
        data_of_dataset_and_embedding \
            = data_of_dataset_and_embedding[
                data_of_dataset_and_embedding[:, 2].argsort()]
            
        params = data_of_dataset_and_embedding[:, 2].astype(float)
        if is_completely_integer(params):
            params = params.astype(int)
            
        
        for mode in MODES:
            ax = ax_of_mode_of_embedding[embedding][mode]
            plt.sca(ax)
            
            plt.grid(color = '0.5', alpha = 0.5)
            
            plt.tick_params(axis = 'both', which = 'major', length = 3)
            plt.tick_params(axis = 'both', which = 'minor', length = 2)
        
            if mode == SCORES:
                y = data_of_dataset_and_embedding[:, 3].astype(float)
                y_err = data_of_dataset_and_embedding[:, 4].astype(float)
            elif mode == RUNTIMES:
                y = data_of_dataset_and_embedding[:, 5].astype(float)
            elif mode == FEATURES_COUNT:
                y = data_of_dataset_and_embedding[:, 6].astype(float)    
            
            # make plot
            if mode == SCORES:
                plt.plot(params, y, linestyle = 'dashed', marker = 'o',
                         color = BLUE, markersize = MARKER_SIZE)
                plt.errorbar(params, y, yerr = y_err, linestyle = 'None',
                             marker = 'None', color = BLUE)
            else:
                plt.plot(params, y, linestyle = 'dashed', marker = 'o',
                         color = BLUE, markersize = MARKER_SIZE)
            
            if embedding != EK:
                    ax.set_xticks(params)
            else:
                x_labels = ['1/6', '2/6', '3/6', '4/6', '5/6', '6/6']
                plt.xticks(params, x_labels)
        
            # set x range and y range
            min_param, max_param = min(params), max(params)
            hor_space = (max_param - min_param)/10
            ax.set_xlim(min_param - hor_space, max_param + hor_space)
            
            min_y, max_y = min(ax.get_yticks()), max(ax.get_yticks())
            if mode != FEATURES_COUNT:
                ver_space = (max_y - min_y)/10
                ax.set_ylim(min_y - ver_space, max_y + ver_space)
            else:
                if min_y < 10:
                    min_y = 1
                ax.set_ylim(min_y, 3*max_y)
        
            # set label of x-axis
            if embedding != EK:
                ax.set_xlabel('h')
            else:
                # embeeding == EK
                ax.set_xlabel('fraction of used features')
            
            
            # set label of y-axis    
            if mode == SCORES:
                ax.set_ylabel('classification accuracy')
            elif mode == RUNTIMES:
                if embedding == EK and dataset in [DD, NCI1, NCI109, FLASH_CFG]:
                    ax.set_ylabel('runtime in minutes')
                elif dataset == ANDROID_FCG:
                    ax.set_ylabel('runtime in hours')
                else:
                    ax.set_ylabel('runtime in seconds')
                    
            elif mode == FEATURES_COUNT:
                ax.set_ylabel('length of feature vectors')
                
            if mode == FEATURES_COUNT:
                plt.yscale('log')
    
            if embedding == EK and mode == FEATURES_COUNT:
                plt.grid(which = 'minor', color = '0.5', alpha = 0.5)
                
    plt.tight_layout(w_pad = 2, h_pad = 4.7, rect = (0, 0.05, 1, 1))

    # add captions
    y_first_ax_bottom = axes_mat[0, 0].get_position().get_points()[0, 1]
    y_second_ax_top = axes_mat[1, 0].get_position().get_points()[1, 1]
    y_second_ax_bottom = axes_mat[1, 0].get_position().get_points()[0, 1]
    y_third_ax_bottom = axes_mat[2, 0].get_position().get_points()[0, 1]
    
    y_offset = abs(y_first_ax_bottom - y_second_ax_top)/2
    y_offset *= 1.2            
    
    y = y_first_ax_bottom - y_offset
    ax.text(0.5, y, '\\normalsize \\textbf{(a)} \\textit{Weisfeiler-Lehman '
            'subtree kernel}', horizontalalignment = 'center',
            verticalalignment = 'center', transform = fig.transFigure)
            
    y = y_second_ax_bottom - y_offset
    ax.text(0.5, y, '\\normalsize \\textbf{(b)} \\textit{Count-sensitive '
            'neighborhood hash kernel}', horizontalalignment = 'center',
            verticalalignment = 'center', transform = fig.transFigure)

    y = y_third_ax_bottom - y_offset
    ax.text(0.5, y, '\\normalsize \\textbf{(c)} \\textit{Eigen graph kernel}',
            horizontalalignment = 'center', verticalalignment = 'center',
            transform = fig.transFigure)                
            
            
    output_file_name = 'figure_' + dataset
    plt.savefig(output_file_name + '.pdf')
    plt.savefig(join(TARGET_PATH, output_file_name + '.pgf'))
    
