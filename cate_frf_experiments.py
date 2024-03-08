import pandas as pd
import numpy as np
import wasserstein_trees as wstree
from util import CATE
import pickle as pkl
from multiprocessing import Pool
import sys
from functools import partial

def wrf_meta_predict(X_valid, wrf_treated, wrf_control, treatment_var):
    y_pred = []
    for i in range(X_valid.shape[0]):
        if X_valid.loc[i, treatment_var] == 1:
            y_pred.append(wrf_control.predict(X_valid.loc[i:i, :].assign(treatment = 0))[0, :])
        else:
            y_pred.append(wrf_treated.predict(X_valid.loc[i:i, :].assign(treatment = 1))[0, :])
    y_pred = np.array(y_pred)
    return y_pred

### Wasserstein random forest experiments
def wrf_parallel(dataset_iteration, dataset_directory):
    print(dataset_iteration)
    seed = 2020 + 1000 * dataset_iteration

    # read dataset
    # maltspro_df = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/X.csv')
    # y = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/Y.csv').to_numpy()
    X_train = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/X_train.csv')
    X_est = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/X_est.csv')
    y_train = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/y_train.csv')
    y_est = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/y_est.csv')
    
    # split into training and estimation datasets: 20% for training, 80% for estimation

    X_train = X_train.reset_index().drop('index', axis = 1)
    X_est = X_est.reset_index().drop('index', axis = 1)
    y_train = y_train.to_numpy()
    y_est = y_est.to_numpy()

    print('training control...', dataset_iteration)
    # split data into treated and control
    # train control random forest
    wrf_control = wstree.wass_forest(X = X_train.query('A == 0'), 
                    y = y_train[X_train.query('A == 0').index.values, :],
                    y_quantile_id=False,
                    min_samples_split=None,
                    max_depth=20,
                    depth=None,
                    node_type=None,
                    n_trees=100,
                    seed=999,
                    n_samples_min=None)

    # train treated random forest
    print('training treated...', dataset_iteration)
    wrf_treated = wstree.wass_forest(X = X_train.query('A == 1'), 
                    y = y_train[X_train.query('A == 1').index.values, :],
                    y_quantile_id=False,
                    min_samples_split=None,
                    max_depth=20,
                    depth=None,
                    node_type=None,
                    n_trees=100,
                    seed=999,
                    n_samples_min=None)

    # save random forest models
    control_file_name = open(dataset_directory + '/dataset_' + str(seed) + '/wrfControl.pkl', 'wb')
    pkl.dump(wrf_control, control_file_name)
    treated_file_name = open(dataset_directory + '/dataset_' + str(seed) + '/wrfTreated.pkl', 'wb')
    pkl.dump(wrf_treated, treated_file_name)
    

    # impute counterfactuals
    print('imputing barycenters...', dataset_iteration)
    y_wrf_bary = wrf_meta_predict(X_est, wrf_control=wrf_control, wrf_treated=wrf_treated, treatment_var='A')

    # estimate the CATE with imputed counterfactuals
    print('estimating CATE...', dataset_iteration)
    CATE_wrf = []
    for i in range(X_est.shape[0]):
        CATE_wrf.append(
            CATE(
                y_obs = y_est[i, :],
                y_cf = y_wrf_bary[i, :],
                observed_treatment = X_est['A'].values[i],
                y_obs_qtl_id = False, 
                y_cf_qtl_id = True
            )
        )
    CATE_wrf = np.array(CATE_wrf)
    ATE_wrf = CATE_wrf.mean(axis = 1)
    print(dataset_iteration, ': True ATE min =', ATE_wrf.min(), ' ATE max = ', ATE_wrf.max())
    
    wrf_CATE_df = pd.DataFrame(CATE_wrf, columns = np.linspace(0, 1, CATE_wrf.shape[0]))
    wrf_CATE_df.to_csv(dataset_directory + '/dataset_' + str(seed) + '/wrf_CATE.csv')
    
def main(argv):
    dataset_directory = sys.argv[1]
    wrf_parallel(dataset_iteration=seed, dataset_directory=dataset_directory)
    dataset_iterations_to_conduct = range(0, 100)
    with Pool(processes = 25) as pool:
        pool.map(partial(wrf_parallel, dataset_directory = dataset_directory),
                dataset_iterations_to_conduct)

if __name__ == '__main__':
    main(sys.argv)