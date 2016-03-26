"""
Evaluation of embedding methods by means of cross-validation.
"""

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-03-02"


import inspect
import numpy as np
import sys
import time

from os.path import abspath, dirname, join
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.grid_search import GridSearchCV

# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import utils


def optimize_embedding_param(clf, feature_mat_of_param, class_lbls,
                             exper_num_iter, num_outer_folds, num_inner_folds,
                             result_file):
    """
    Perform cross-validation in order to find the best embedding parameter.
    
    The best classifier is determined by exhaustive search over the range of
    embedding parameters and over the grid of SVM parameters specified by
    grid_clf. The scores are obtained via cross-validation on the training
    data. The parameter data_mat can be either a feature matrix or a kernel
    matrix.
    """
    cross_val_start_time = time.time()

    mean_scores_on_test_data = []
    for i in xrange(exper_num_iter):
        scores_on_test_data = []
        outer_cv = KFold(len(class_lbls), num_outer_folds, shuffle = True)
                                                               
        for j, (train_indices, test_indices) in enumerate(outer_cv):
            best_param_on_train_data = -1
            best_score_on_train_data = 0.0
            opt_clf = None

#            for param, feature_mat in feature_mat_of_param.iteritems():
            for param in sorted(feature_mat_of_param.iterkeys()):
                feature_mat = feature_mat_of_param[param]

                if isinstance(clf, GridSearchCV):
                    clf.fit(feature_mat[train_indices], class_lbls[train_indices])
                    
                    sub_clf = clf.best_estimator_
                    
                    if isinstance(param, int):
                        print('param = %d, i = %d, j = %d: params = %s'
                              % (param, i, j, clf.best_params_))
                    elif isinstance(param, float):
                        print('param = %.2f, i = %d, j = %d: params = %s'
                              % (param, i, j, clf.best_params_))                        
                    score_on_train_data = sub_clf.score(feature_mat[test_indices],
                                                        class_lbls[test_indices])
                    if score_on_train_data > best_score_on_train_data:
                        opt_clf = clf
                else:
                    # clf is an instance of LinearSVC
                    score_on_train_data = cross_val_score(
                        clf, feature_mat[train_indices],
                        class_lbls[train_indices],
                        cv = num_inner_folds).mean()
                
                if isinstance(param, int):
                    print('param = %d, i = %d, j = %d: score = %.2f'
                          % (param, i, j, score_on_train_data))
                elif isinstance(param, float):
                    print('param = %.2f, i = %d, j = %d: score = %.2f'
                          % (param, i, j, score_on_train_data))
                                                             
                if score_on_train_data > best_score_on_train_data:
                    best_score_on_train_data = score_on_train_data
                    best_param_on_train_data = param
                    opt_clf = clf
             
            if isinstance(clf, GridSearchCV):
                clf = opt_clf
                
            best_data_mat = feature_mat_of_param[best_param_on_train_data]
            clf.fit(best_data_mat[train_indices], class_lbls[train_indices])
            score_on_test_data = clf.score(best_data_mat[test_indices],
                                           class_lbls[test_indices])
            scores_on_test_data.append(score_on_test_data)
            if isinstance(best_param_on_train_data, int):
                print('-> score on test data = %.2f (best param = %d)\n'
                      % (score_on_test_data, best_param_on_train_data))
            elif isinstance(best_param_on_train_data, float):
                print('-> score on test data = %.2f (best param = %.2f)\n'
                      % (score_on_test_data, best_param_on_train_data))
       
        mean_score_on_test_data = np.mean(scores_on_test_data) 
        mean_scores_on_test_data.append(mean_score_on_test_data)

        print('-------------------------------------------------------------')
        print('RESULT for i = %d: %.7f' % (i, mean_score_on_test_data))
        print('-------------------------------------------------------------\n')                                           
                                                        
    cross_val_end_time = time.time()
    cross_val_time = cross_val_end_time - cross_val_start_time
    
    print('-------------------------------------------------------------')
    sys.stdout.write('TOTAL RESULT: ')
    utils.write('%.3f (+/-%.3f) in %.1f seconds\n'
                % (np.mean(mean_scores_on_test_data),
                   np.std(mean_scores_on_test_data),
                   cross_val_time), result_file)
    print('-------------------------------------------------------------\n')

    
def cross_val(grid_clf, data_mat, class_lbls, embedding_is_implicit,
              exper_num_iter, num_outer_folds, result_file):
    """
    Perform cross-validation for fixed embedding parameter.
    
    The best classifier is determined by exhaustive search over the range
    of embedding parameters and over the grid of SVM parameters specified
    by grid_clf.
    
    The best classifier is determined by exhaustive search over the grid of
    SVM parameters specified by grid_clf. The scores are obtained via
    cross-validation on the training data. The parameter data_mat can be
    either a feature matrix or a kernel matrix.
    """
    cross_val_start_time = time.time()

    mean_scores_on_test_data = []   
    for i in xrange(exper_num_iter):
        scores_on_test_data = []
        outer_cv = KFold(len(class_lbls), num_outer_folds, shuffle = True)
        
        for j, (train_indices, test_indices) in enumerate(outer_cv):
            if not embedding_is_implicit:
                # explicit embedding
                grid_clf.fit(data_mat[train_indices], class_lbls[train_indices])
            else:
                # implicit embedding
                grid_clf.fit(data_mat[np.ix_(train_indices, train_indices)],
                             class_lbls[train_indices])
                
            opt_clf = grid_clf.best_estimator_
                
            print('i = %d, j = %d: params = %s' % (i, j, grid_clf.best_params_))
            
            if not embedding_is_implicit:
                # explicit embedding                 
                score_on_test_data = opt_clf.score(data_mat[test_indices],
                                                   class_lbls[test_indices])
            else:
                # implicit embedding
                score_on_test_data = opt_clf.score(
                    data_mat[np.ix_(test_indices, train_indices)],
                    class_lbls[test_indices])       
        
            scores_on_test_data.append(score_on_test_data)

            # !!
#            x = 0
#            clf2 = grid_clf.estimator
#            scores_on_test_data = cross_val_score(clf2, data_mat, class_lbls,
#                                                  cv = outer_cv)
                                                  
        mean_score_on_test_data = np.mean(scores_on_test_data) 
        mean_scores_on_test_data.append(mean_score_on_test_data)
        print('%d) score: %.7f' % (i, mean_score_on_test_data))
    print('')
      
    cross_val_end_time = time.time()
    cross_val_time = cross_val_end_time - cross_val_start_time
          
    sys.stdout.write('RESULT: ')
    utils.write('%.3f (+/-%.3f) in %.1f seconds\n'
                % (np.mean(mean_scores_on_test_data),
                   np.std(mean_scores_on_test_data),
                   cross_val_time), result_file)
    sys.stdout.write('\n')
    
