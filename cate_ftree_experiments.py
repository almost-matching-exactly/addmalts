import pandas as pd
import numpy as np
import wasserstein_trees as wstree
from util import CATE
import pickle as pkl
from multiprocessing import Pool
import sys
from functools import partial

def wass_tree_meta_predict(X_valid, wass_tree_treated, wass_tree_control, treatment_var):
    y_pred = []
    for i in range(X_valid.shape[0]):
        if X_valid.loc[i, treatment_var] == 1:
            y_pred.append(wass_tree_control.predict(X_valid.loc[i:i, :].assign(treatment = 0))[0, :])
        else:
            y_pred.append(wass_tree_treated.predict(X_valid.loc[i:i, :].assign(treatment = 1))[0, :])
    y_pred = np.array(y_pred)
    return y_pred

### Wasserstein random forest experiments
def wass_tree_parallel(dataset_iteration, dataset_directory):
    print(dataset_iteration, end = ' ')
    seed = 2020 + 1000 * dataset_iteration

    # read dataset
    X_train = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/X_train.csv')
    X_est = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/X_est.csv')
    y_train = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/y_train.csv')
    y_est = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/y_est.csv')
    

    X_train = X_train.reset_index().drop('index', axis = 1)
    X_est = X_est.reset_index().drop('index', axis = 1)
    y_train = y_train.to_numpy()
    y_est = y_est.to_numpy()

    # split data into treated and control
    # train control random forest
    control_indexes = np.where(X_train['A'].values == 0)[0]
    wass_tree_control = wstree.wass_forest(X = X_train.query('A == 0'), 
                    y = y_train[control_indexes, :],
                    y_quantile_id=False,
                    min_samples_split=None,
                    max_depth=5,
                    depth=None,
                    node_type=None,
                    n_trees=1,
                    seed=999,
                    n_samples_min=None)

    # train treated random forest
    treated_indexes = np.where(X_train['A'].values == 1)[0]
    wass_tree_treated = wstree.wass_forest(X = X_train.query('A == 1'), 
                    y = y_train[treated_indexes, :],
                    y_quantile_id=False,
                    min_samples_split=None,
                    max_depth=5,
                    depth=None,
                    node_type=None,
                    n_trees=1,
                    seed=999,
                    n_samples_min=None)
    
    # save random forest models
    control_file_name = open(dataset_directory + '/dataset_' + str(seed) + '/wass_treeControl.pkl', 'wb')
    pkl.dump(wass_tree_control, control_file_name)
    treated_file_name = open(dataset_directory + '/dataset_' + str(seed) + '/wass_treeTreated.pkl', 'wb')
    pkl.dump(wass_tree_treated, treated_file_name)

    # impute counterfactuals
    y_wass_tree_bary = wass_tree_meta_predict(X_est, wass_tree_control=wass_tree_control, wass_tree_treated=wass_tree_treated, treatment_var='A')
    
    # estimate the ITE with imputed counterfactuals
    ITE_wass_tree = []
    for i in range(X_est.shape[0]):
        ITE_wass_tree.append(
            CATE(
                y_obs = y_est[i, :],
                y_cf = y_wass_tree_bary[i, :],
                observed_treatment = X_est['A'].values[i],
                y_obs_qtl_id = False, 
                y_cf_qtl_id = True
            )
        )
    ITE_wass_tree = np.array(ITE_wass_tree)
    ATE_wass_tree = ITE_wass_tree.mean(axis = 1)
    print(dataset_iteration, ': WTree ATE min =', ATE_wass_tree.min(), ' ATE max = ', ATE_wass_tree.max())
    
    wass_tree_ITE_df = pd.DataFrame(ITE_wass_tree, columns = np.linspace(0, 1, ITE_wass_tree.shape[0]))
    wass_tree_ITE_df.to_csv(dataset_directory + '/dataset_' + str(seed) + '/wass_tree_CATE.csv')

def main(argv):
    dataset_directory = sys.argv[1]
    dataset_iterations_to_conduct = range(0, 100)
    with Pool(processes = 25) as pool:
        pool.map(partial(wass_tree_parallel, dataset_directory = dataset_directory),
                dataset_iterations_to_conduct)


if __name__ == '__main__':
    main(sys.argv)