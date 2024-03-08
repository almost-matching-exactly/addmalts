import pandas as pd
import numpy as np
from util import CATE
import pickle as pkl
from multiprocessing import Pool
from sklearn.ensemble import RandomForestRegressor
import sys
from functools import partial
def ps_predict(treatment_query, X_query_ps, treatment_est, y_est, y_est_qtl_id, ps_est, k, n_samples_min):
    '''
    description
    -----------
    find k points with closest treatment propensities to X_query's and compute barycenter of their outcomes
    
    parameters
    ----------
    X_query : vector describing X's covariates
    treatment_query : treatment assignment that we want to impute for
    X_query_ps : propensity score of the query point
    y_est : observed outcome for each unit in estimation set
    y_est_qtl_id : True if y_est is quantile functions
    ps_est : propensity score of each unit in estimation set
    n_samples_min : min number of samples for each outcome vector
    '''
    
    knn_id = np.argsort(np.abs(X_query_ps - ps_est))[:k]
    
    
    
    knn_y = []
    # if outcomes are already quantile functions, just append together
    if y_est_qtl_id == True:
        for i in knn_id:
            knn_y.append(y_est[i, :])
    else:
        quantile_values = np.linspace(start = 0, stop = 1, num = n_samples_min)
        y_est_qtl = np.apply_along_axis(arr = y_est,
                                        axis = 1,
                                        func1d = lambda x: np.quantile(x[x == x], q = quantile_values)
                                        )
        for i in knn_id:
            knn_y.append(y_est_qtl[i, :])
    knn_y = np.array(knn_y)
    y_bary = np.mean(knn_y, axis = 0) # barycenter is col means of quantile functions
    
    return y_bary

### Propensity score matching experiments
def ps_parallel(dataset_iteration, dataset_directory):
    print(dataset_iteration)
    seed = 2020 + 1000 * dataset_iteration

    # read dataset
    X_train = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/X_train.csv')
    X_est = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/X_est.csv')
    y_train = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/y_train.csv')
    y_est = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/y_est.csv')

    X_train = X_train.reset_index()
    X_est = X_est.reset_index()
    y_train = y_train.to_numpy()
    y_est = y_est.to_numpy()


    print('fitting prop score...', dataset_iteration)
    # train propensity score model : random forest regression
    ps_model = RandomForestRegressor()
    ps_model.fit(X = X_train.drop('A', 1).to_numpy(), y = X_train['A'].values)
    ps_values = ps_model.predict(X_est.drop('A',1).to_numpy())
    
    n_samples_min = np.apply_along_axis(arr = y_est,
                                    axis = 1,
                                    func1d = lambda x: x[x == x].shape[0]).min()
    
    print('imputing barycenter...', dataset_iteration)
    # impute counterfactuals 
    y_ps_bary = []
    for i in range(X_est.shape[0]):
        y_ps_bary.append(
            ps_predict(
                treatment_query = 1 - X_est['A'].values[i], # impute counterfactual
                X_query_ps = ps_values[i],
                treatment_est = X_est['A'].values, 
                y_est = y_est, 
                y_est_qtl_id = False, # observed outcomes were not quantile function
                ps_est = ps_values,
                k = 10, # take 10 nearest neighbors
                n_samples_min=n_samples_min
                )
        )
    y_ps_bary = np.array(y_ps_bary)
    
    print('fitting CATE...', dataset_iteration)
    
    # measure P(A > B | A ~ Y_i(1), B ~ Y_i(0)) for units i in estimation set
    CATE_ps = []
    for i in range(X_est.shape[0]):
        CATE_ps.append(
            CATE(
                y_obs = y_est[i, :],
                y_cf = y_ps_bary[i, :],
                observed_treatment = X_est['A'].values[i],
                y_obs_qtl_id = False, 
                y_cf_qtl_id = True
            )
            )
    CATE_ps = np.array(CATE_ps)
    ATE_ps = CATE_ps.mean(axis = 1)
    # print(dataset_iteration, ': MALTSPro ATE =', ATE_malts)
    pd.DataFrame(CATE_ps, columns=np.linspace(0, 1, CATE_ps.shape[0])).\
        to_csv(dataset_directory + '/dataset_' + str(seed) + '/rf_ps_CATE.csv')
    
def main(argv):
    dataset_directory = sys.argv[1]
    dataset_iterations_to_conduct = range(0, 100)
    with Pool(processes = 25) as pool:
        pool.map(partial(ps_parallel, dataset_directory = dataset_directory),
                dataset_iterations_to_conduct)


if __name__ == '__main__':
    main(sys.argv)