"""
Evaluation of embedding methods.

This module provides functions for evaluating the performance of four
explecit two implicit graph embedding methods. The explicit ones are the
Weisfeiler-Lehman subtree kernel, the neighborhood hash kernel (in three
variants) and the !!. The implicit embeddings comprise the random walk
kernel and the !!. The classification accuracies and runtimes are
evaluated on the following 8 datasets: MUTAG, PTC(MR), ENZYMES, DD,
NCI1, NCI109, FLASH CFG, and ANDROID FCG.
"""
from __future__ import division


__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-03-18"


# planed procedure:
#
# at Ben-PC:
# !!!
# 01. finish the implementation of the Eigen kernel
# 02. optimize coding style
# 10. document RWkernel, PCG, graphlet_kernel and get_lamda
# 

# at Benny-Notebook:
#
# 00. test Eigen kernel on FLASH CFG
# 01. test all methods on ANDROID FCG 14795
# 02. test Eigen kernel on FLASH CFG and ANDROID FCG 14795 using "to_undirected"
# 02. test methods on ENZYMES with ovo
# 90. test on large datasets with twice param grid size
# 

# at Sylvia-Notebook:
#
# 00. compress ANDROID FCG 14795 using add_egde
# 01. WL on ANDROID FCG 14795 (feature extraction took ca. 5 h, 0.94)
# 02. EGK on ANDROID FCG 14795 (feature extraction took ca. x h, x)
# 03. RW on ANDROID FCG 14795 (feature extraction took ca. x h, x)
# 04. GK-3 on ANDROID FCG 14795 (feature extraction took ca. x h, x)


import numpy as np
import importlib
import inspect
import sys
import time

from os.path import abspath, dirname, join
from sklearn import svm
from sklearn.grid_search import GridSearchCV


# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))


from misc import dataset_loader, utils
from performance_evaluation import cross_validation


#=================================================================================
# constants
#=================================================================================
DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', 'datasets')

# embeddings
WEISFEILER_LEHMAN = 'weisfeiler_lehman'
NEIGHBORHOOD_HASH = 'neighborhood_hash'
COUNT_SENSITIVE_NEIGHBORHOOD_HASH = 'count_sensitive_neighborhood_hash'
COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER =\
                                      'count_sensitive_neighborhood_hash_all_iter'
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
ANDROID_FCG_14795 = 'ANDROID FCG 14795'


#=================================================================================
# parameter definitions
#=================================================================================
#EMBEDDING_NAMES = [WEISFEILER_LEHMAN, COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER,
#                   EIGEN_KERNEL, RANDOM_WALK_KERNEL, GRAPHLET_KERNEL_3]
#EMBEDDING_NAMES = [EIGEN_KERNEL, RANDOM_WALK_KERNEL, GRAPHLET_KERNEL_3]
#EMBEDDING_NAMES = [WEISFEILER_LEHMAN]
#EMBEDDING_NAMES = [WEISFEILER_LEHMAN, GRAPHLET_KERNEL_3, GRAPHLET_KERNEL_4]
EMBEDDING_NAMES = [WEISFEILER_LEHMAN, COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER]
#EMBEDDING_NAMES = [WEISFEILER_LEHMAN, COUNT_SENSITIVE_NEIGHBORHOOD_HASH,
#                   COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER]
#EMBEDDING_NAMES = [WEISFEILER_LEHMAN, NEIGHBORHOOD_HASH]
#EMBEDDING_NAMES = [COUNT_SENSITIVE_NEIGHBORHOOD_HASH]
#EMBEDDING_NAMES = [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER]
#EMBEDDING_NAMES = [NEIGHBORHOOD_HASH, COUNT_SENSITIVE_NEIGHBORHOOD_HASH,
#                   COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER]
#EMBEDDING_NAMES = [GRAPHLET_KERNEL_3]
#EMBEDDING_NAMES = [GRAPHLET_KERNEL_4]
#EMBEDDING_NAMES = [GRAPHLET_KERNEL_3, GRAPHLET_KERNEL_4]
#EMBEDDING_NAMES = [RANDOM_WALK_KERNEL]
#EMBEDDING_NAMES = [EIGEN_KERNEL]


# keys are indices of the list EMBEDDING_NAMES, values are the respective
# parameters
EMBEDDING_PARAM_RANGES = {
    WEISFEILER_LEHMAN: range(6),
    NEIGHBORHOOD_HASH: range(6),
    COUNT_SENSITIVE_NEIGHBORHOOD_HASH: range(6),
    COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER: range(6),
    GRAPHLET_KERNEL_3: [None],
    GRAPHLET_KERNEL_4: [None],
    RANDOM_WALK_KERNEL: [None],
    EIGEN_KERNEL: np.linspace(1/6, 1, 6)}


# sorted by number of graphs in ascending order
#DATASETS = [MUTAG, PTC_MR, ENZYMES, DD, NCI1, NCI109, FLASH_CFG]
#DATASETS = [NCI109, FLASH_CFG]
#DATASETS = [MUTAG, PTC_MR, ENZYMES, NCI1, NCI109]
#DATASETS = [DD, NCI1, NCI109, FLASH_CFG]
#DATASETS = [MUTAG, PTC_MR, ENZYMES]
#DATASETS = [DD, NCI1, NCI109]
#DATASETS = [MUTAG]
#DATASETS = [PTC_MR]
#DATASETS = [ENZYMES]
#DATASETS = [DD]
#DATASETS = [NCI1]
#DATASETS = [NCI109]
#DATASETS = [FLASH_CFG]
DATASETS = [ANDROID_FCG_14795]

OPT_PARAM = True
#OPT_PARAM = False

COMPARE_PARAMS = True
#COMPARE_PARAMS = False

SEARCH_OPT_SVM_PARAM_IN_PAR = True
#SEARCH_OPT_SVM_PARAM_IN_PAR = False

#EXPER_NUM_ITER = 10
EXPER_NUM_ITER = 5
#EXPER_NUM_ITER = 3
#EXPER_NUM_ITER = 1

# maximum number of iterations for small datasets (having less than 1000 samples)
CLF_MAX_ITER_SD = 1e7 # final value (take care of perfectionism!!!)

# maximum number of iterations for large datasets (having more than 1000 samples)
CLF_MAX_ITER_LD = 1e3 # final value (take care of perfectionism!!!)

# number of folds used in cross validation for performance evaluation
NUM_OUTER_FOLDS = 10

# number of folds used in cross validation on training data for small datasets
# (i.e., less than 1000 samples)
NUM_INNER_FOLDS_SD = 3

# number of folds used in cross validation on training data for large datasets
# (i.e., more than 1000 samples)
NUM_INNER_FOLDS_LD = 2

NUM_CROSS_VAL_JOBS = 4


def extract_features(graph_meta_data_of_num, embedding, param_range, result_file):
    print '-------------------------------------------------------------\n'
    result_file.write('------------------------------------------\n\n')

    feat_extr_start_time = time.time()

    feature_mat_of_param, extr_time_of_param =\
                   embedding.extract_features(graph_meta_data_of_num, param_range)
                   
#    import sys
#    sys.modules['__main__'].F = feature_mat_of_param[None]
    
#    kernel_mat_of_param = {None: F}
#    kernel_mat_comp_time_of_param = {None: 0}

    feat_extr_end_time = time.time()
    feat_extr_time = feat_extr_end_time - feat_extr_start_time
    utils.write('Graph loading and feature exraction took %.1f seconds.\n' %\
                                                      feat_extr_time, result_file)
    print ''

    return feature_mat_of_param, extr_time_of_param
    
    
def compute_kernel_matrix(graph_meta_data_of_num, embedding, param_range,
                          result_file):
    print '-------------------------------------------------------------\n'
    result_file.write('------------------------------------------\n\n')

    kernel_mat_comp_start_time = time.time()

    kernel_mat_of_param, kernel_mat_comp_time_of_param \
        = embedding.compute_kernel_mat(graph_meta_data_of_num, param_range)

    kernel_mat_comp_end_time = time.time()
    kernel_mat_comp_time = kernel_mat_comp_end_time - kernel_mat_comp_start_time
    utils.write('The computation of the kernel matrix took %.1f seconds.\n' %\
                                                kernel_mat_comp_time, result_file)
    print ''

    return kernel_mat_of_param, kernel_mat_comp_time_of_param
        

def get_params(graph_meta_data_of_num, embedding_name):
    num_samples = len(graph_meta_data_of_num)

    if num_samples >= 1000:
        dataset_is_large = True
        clf_max_iter = CLF_MAX_ITER_LD
        num_inner_folds = NUM_INNER_FOLDS_LD
    else:
        dataset_is_large = False
        clf_max_iter = CLF_MAX_ITER_SD
        num_inner_folds = NUM_INNER_FOLDS_SD
        
    implicit_embeddings = [RANDOM_WALK_KERNEL]
    
    if embedding_name in implicit_embeddings:
        embedding_is_implicit = True
        # use library LIBSVM
        use_liblinear = False
        svm_param_grid = {'C': tuple(np.logspace(-2, 3, NUM_CROSS_VAL_JOBS))}
        kernel = 'precomputed'
    else:
        # embedding is explicit
        embedding_is_implicit = False
        if dataset_is_large:
            # use library LIBLINEAR
            use_liblinear = True
            svm_param_grid = {'C': tuple(np.logspace(-2, 3, NUM_CROSS_VAL_JOBS))}
            kernel = 'linear'
        else:
            # use library LIBSVM
            use_liblinear = False
            svm_param_grid = {'kernel': ('linear', 'rbf'), 'C': (0.1, 10)}
            kernel = 'linear/rbf'

    return dataset_is_large, embedding_is_implicit, use_liblinear, kernel, \
        svm_param_grid, clf_max_iter, num_inner_folds
    

def get_svm_param_grid_str(svm_param_grid):
    kernel_is_given = 'kernel' in svm_param_grid.iterkeys()
    C_is_given = 'C' in svm_param_grid.iterkeys()
    svm_param_grid_str = '{'
    if kernel_is_given:
        if isinstance(svm_param_grid['kernel'], (list, tuple)):
            svm_param_grid_str += 'kernel: (' \
                                  + ', '.join(svm_param_grid['kernel']) + ')'
        else:
            svm_param_grid_str += 'kernel: ' + svm_param_grid['kernel']
    if kernel_is_given and C_is_given:
        svm_param_grid_str += ', '
    if C_is_given:
        if isinstance(svm_param_grid['C'], (list, tuple)):
            C_values_str = ['%.1e' % C_value for C_value in svm_param_grid['C']]
            svm_param_grid_str += 'C: (' + ', '.join(C_values_str) + ')'
        else:
            svm_param_grid_str += 'C: %.1e' % svm_param_grid['C']
    svm_param_grid_str += '}'
    
    return svm_param_grid_str
    

def write_param_info(use_liblinear, embedding_is_implicit, svm_param_grid,
                     clf_max_iter, num_inner_folds, result_file):
                         
    if use_liblinear:
        utils.write('LIBRARY: LIBLINEAR\n', result_file)
    else:
        utils.write('LIBRARY: LIBSVM\n', result_file)
    if embedding_is_implicit:
        utils.write('EMBEDDING TYPE: IMPLICIT\n', result_file)
    else:
        utils.write('EMBEDDING TYPE: EXPLICIT\n', result_file) 
    utils.write('EXPER_NUM_ITER: %d\n' % EXPER_NUM_ITER, result_file)
    utils.write('SVM_PARAM_GRID: %s\n' % get_svm_param_grid_str(svm_param_grid),
                result_file)
    utils.write('NUM_CROSS_VAL_JOBS: %d\n' % NUM_CROSS_VAL_JOBS, result_file)
    utils.write('NUM_OUTER_FOLDS: %d\n' % NUM_OUTER_FOLDS, result_file)
    utils.write('NUM_INNER_FOLDS: %d\n' % num_inner_folds, result_file)
    if clf_max_iter == -1:
        utils.write('CLF_MAX_ITER: UNLIMITED\n', result_file)
    else:
        utils.write('CLF_MAX_ITER: %.e\n' % clf_max_iter, result_file)
    utils.write('SEARCH_OPT_SVM_PARAM_IN_PAR: %s\n' \
        % SEARCH_OPT_SVM_PARAM_IN_PAR.__str__().upper(), result_file)
    sys.stdout.write('\n')
    

def init_grid_clf(embedding_is_implicit, dataset_is_large, svm_param_grid,
                  clf_max_iter, num_inner_folds):
    """
    Initialize classifier.
    
    For multiclass classification the One-Versus-Rest scheme is applied,
    i.e., in case of N different classes N classifiers are trained in
    total. !! further details
    """
    if dataset_is_large:
        if embedding_is_implicit:
            # library LIBSVM is used
            clf = svm.SVC(kernel = 'precomputed', max_iter = clf_max_iter,
                          decision_function_shape = 'ovr')
        else:
            # library LIBLINEAR is used            
            clf = svm.LinearSVC(max_iter = clf_max_iter)
    else:
        # library LIBSVM is used
        if embedding_is_implicit:
            clf = svm.SVC(kernel = 'precomputed', max_iter = clf_max_iter,
                          decision_function_shape = 'ovr')
#                          decision_function_shape = 'ovo')
        else:
            clf = svm.SVC(max_iter = clf_max_iter,
                          decision_function_shape = 'ovr')
#                          decision_function_shape = 'ovo')
    
    if SEARCH_OPT_SVM_PARAM_IN_PAR:
        grid_clf = GridSearchCV(clf, svm_param_grid, cv = num_inner_folds,
                                n_jobs = NUM_CROSS_VAL_JOBS,
                                pre_dispatch = '2*n_jobs')
    else:
        grid_clf = GridSearchCV(clf, svm_param_grid, cv = num_inner_folds)
    
    return grid_clf        
        
    
def write_eval_info(dataset, embedding_name, kernel, mode = None):
    mode_str = ' (' + mode + ')' if mode else ''
    
    print ('%s with %s kernel%s on %s\n') %\
               (embedding_name.upper(), kernel.upper(), mode_str.upper(), dataset)
           

def write_feature_mat_dim_and_extr_time(param, feature_mat_of_param,
                                        extr_time_of_param, result_file):
    print '-------------------------------------------------------------'
    result_file.write('------------------------------------------\n')
    utils.write('Parameter: %r\n\n' % param, result_file)
    utils.write('Feature extraction took %.1f seconds.\n' %\
                extr_time_of_param[param], result_file)
    utils.write('Feature matrix dimension: %s\n' %\
                                (feature_mat_of_param[param].shape,), result_file)
    sys.stdout.write('\n')
    
    
def write_kernel_mat_dim_and_kernel_comp_time(param, kernel_mat_of_param,
                                              kernel_mat_comp_time_of_param,
                                              result_file):
    print '-------------------------------------------------------------'
    result_file.write('------------------------------------------\n')
    utils.write('Parameter: %r\n\n' % param, result_file)
    utils.write('The computation of the kernel matrix took %.1f seconds.\n' %\
                kernel_mat_comp_time_of_param[param], result_file)
    utils.write('Kernel matrix dimension: %s\n' %\
                                 (kernel_mat_of_param[param].shape,), result_file)
    sys.stdout.write('\n')
    
    

script_exec_start_time = time.time()

for dataset in DATASETS:
    #=============================================================================
    # 1) retrieve graph meta data and class lables
    #=============================================================================
    graph_meta_data_of_num, class_lbls \
        = dataset_loader.get_graph_meta_data_and_class_lbls(dataset,
                                                            DATASETS_PATH)
    
    for embedding_name in EMBEDDING_NAMES:
        # set parameters depending on whether or not the number of samples within 
        # the dataset is larger than 1000 and depending on wether the embedding is
        # implict or explicit
        embedding = importlib.import_module('embeddings.' + embedding_name)
        
        # initialize parameters
        dataset_is_large, embedding_is_implicit, use_liblinear, kernel, \
            svm_param_grid, clf_max_iter, num_inner_folds = get_params(
                graph_meta_data_of_num,
                embedding_name)
        
        param_range = EMBEDDING_PARAM_RANGES[embedding_name]
        
        result_path = join(SCRIPT_FOLDER_PATH, '..', 'results', embedding_name)
        utils.makedir(result_path)
        result_file = open(join(result_path, dataset + '.txt'), 'w')
        
        write_param_info(use_liblinear, embedding_is_implicit, svm_param_grid,
                         clf_max_iter, num_inner_folds, result_file)
        

        #=========================================================================
        # 2) extract features if embedding is an explicit embedding, else compute
        #    the kernel matrix
        #=========================================================================
        if not embedding_is_implicit:
            # !!
#            pass
            feature_mat_of_param, extr_time_of_param \
                = extract_features(graph_meta_data_of_num, embedding, param_range,
                                   result_file)
        else:
            kernel_mat_of_param, kernel_mat_comp_time_of_param =\
                          compute_kernel_matrix(graph_meta_data_of_num, embedding,
                                                param_range, result_file)
                                                
        # initialize SVM classifier
        grid_clf = init_grid_clf(embedding_is_implicit, dataset_is_large,
                                 svm_param_grid, clf_max_iter, num_inner_folds)

# !!                                 
#        feature_mat_of_param = F
#        extr_time_of_param = {}
#        extr_time_of_param[0.2] = 0
#        extr_time_of_param[0.4] = 0
#        extr_time_of_param[0.6] = 0
#        extr_time_of_param[0.8] = 0
#        extr_time_of_param[1] = 0
                
        
        if OPT_PARAM and len(param_range) > 1:
            #=====================================================================
            # 3) evaluate the embedding's performance with optimized embedding
            #    parameter (this is only done for explicit embeddings)
            #=====================================================================
            mode = 'opt_param'
            
            result_file.write('\n%s (%s)\n' % (kernel.upper(), mode.upper()))
            
            write_eval_info(dataset, embedding_name, kernel, mode)
            
            cross_validation.optimize_embedding_param(grid_clf,
                                                      feature_mat_of_param,
                                                      class_lbls, EXPER_NUM_ITER,
                                                      NUM_OUTER_FOLDS,
                                                      num_inner_folds,
                                                      result_file)                                           
        if not COMPARE_PARAMS:
            result_file.close()
            continue
        
        
        if OPT_PARAM:
            result_file.write('\n')
                
                                                                
        if COMPARE_PARAMS:
            #=====================================================================
            # 4) evaluate the embedding's performance for each embedding
            #    parameter
            #=====================================================================
            for param in param_range:
                if not embedding_is_implicit:
                    write_feature_mat_dim_and_extr_time(param,
                                                        feature_mat_of_param,
                                                        extr_time_of_param,
                                                        result_file)
                else:
                    write_kernel_mat_dim_and_kernel_comp_time(param,\
                               kernel_mat_of_param, kernel_mat_comp_time_of_param,
                               result_file)
               
                result_file.write('\n%s\n' % kernel.upper())
                
                write_eval_info(dataset, embedding_name, kernel)
                
                if not embedding_is_implicit:
                    feature_mat = feature_mat_of_param[param]
                    cross_validation.cross_val(grid_clf, feature_mat, class_lbls,
                                               embedding_is_implicit,
                                               EXPER_NUM_ITER, NUM_OUTER_FOLDS,
                                               result_file)
                else:
#                    kernel_mat == feature_mat.dot(feature_mat.T) # !!
#                    kernel_mat = pairwise_kernels(feature_mat)
                    kernel_mat = kernel_mat_of_param[param]
                    cross_validation.cross_val(grid_clf, kernel_mat, class_lbls,
                                               embedding_is_implicit,
                                               EXPER_NUM_ITER, NUM_OUTER_FOLDS,
                                               result_file)

            
        result_file.close()

script_exec_end_time = time.time()
script_exec_time = script_exec_end_time - script_exec_start_time

print '\nThe evaluation of the emedding method(s) took %.1f seconds.' %\
                                                                  script_exec_time
                                                                  
                                                                  
                                                                  
