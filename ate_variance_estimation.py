import pandas as pd
import numpy as np
import wasserstein_trees as wstree
import pickle as pkl
import sys
from multiprocessing import Pool
from functools import partial
from scipy.special import ndtri # to get z score

## train add malts model
def read_addmalts_model(directory):
    am_model = pkl.load(open(directory + '/malts_model.pkl', 'rb'))
    return am_model


def abadie_imbdens(seed, dataset_directory):
    print(dataset_directory, seed, 'reading data')
    directory = dataset_directory + '/dataset_' + str(2020 + 1000 * seed)
    ## get training, calibration, and estimation sets
    X_train = pd.read_csv(directory + '/X_train.csv')
    X_est  = pd.read_csv(directory + '/X_est.csv')
    y_train = pd.read_csv(directory + '/y_train.csv').to_numpy()
    y_est  = pd.read_csv(directory + '/y_est.csv').to_numpy()
    ate_true = pd.read_csv(directory + '/tau_ate.csv').to_numpy()[0]

    ## train add malts model
    am_model = read_addmalts_model(directory)

    ## fix add malts distance metric
    if dataset_directory == './experiments/trunc_normal_variance':
        am_model.M_opt = np.array([1, 2])
        am_model.Mc = np.array([1,2])
        am_model.Md = np.array([])
        am_model.Mq = np.array([])
    elif dataset_directory == './experiments/trunc_normal_constants':
        am_model.M_opt = np.array([1, 1])
        am_model.Mc = np.array([1,1])
        am_model.Md = np.array([])
        am_model.Mq = np.array([])
    elif dataset_directory == './experiments/trunc_normal_complex': # use learned distance metric
        am_model.M_opt = am_model.M_opt
    elif dataset_directory == './experiments/trunc_normal_quadratic':
        am_model.M_opt = np.array([1, 2, 3, 4, 5, 6])
        am_model.Mc = np.array([1, 2, 3, 4, 5, 6])
        am_model.Md = np.array([])
        am_model.Mq = np.array([])
    elif dataset_directory == './experiments/trunc_normal_linear_same':
        am_model.M_opt = np.array([1, 2])
        am_model.Mc = np.array([1, 2])
        am_model.Md = np.array([])
        am_model.Mq = np.array([])
    elif dataset_directory == './experiments/trunc_normal_linear_diff':
        am_model.M_opt = np.array([1, 2])
        am_model.Mc = np.array([1, 2])
        am_model.Md = np.array([])
        am_model.Mq = np.array([])
    elif dataset_directory == './experiments/trunc_normal_variance_quadratic':
        am_model.M_opt = np.array([1, 2, 3, 4, 5, 6])
        am_model.Mc = np.array([1, 2, 3, 4, 5, 6])
        am_model.Md = np.array([])
        am_model.Mq = np.array([])
    else:
        raise Exception(f'{dataset_directory} not found')
    ## convert outcomes to quantile functions
    qtl_grid = np.linspace(0, 1, am_model.n_samples_min)
    y_train_qtl = np.apply_along_axis(func1d= lambda x: np.quantile(x, qtl_grid),
                                    arr = y_train,
                                    axis = 1)
    y_est_qtl = np.apply_along_axis(func1d= lambda x: np.quantile(x, qtl_grid),
                                    arr = y_est,
                                    axis = 1)
    k = 10

    print(dataset_directory, 'getting ate estimate')
    # get matched groups
    mg_df = am_model.get_matched_groups(X_estimation=X_est,
                                        Y_estimation= y_est_qtl,
                                        k = k)

    # count number of times a unit is matched
    n_times_matched = mg_df.groupby('matched_unit')[am_model.treatment].count().sort_index().values
    y_bary = am_model.barycenter_imputation(X_estimation = X_est,
                                            Y_estimation = y_est_qtl,
                                            MG = mg_df, 
                                            qtl_id = True)

    y_hat_treated = y_est_qtl * X_est[am_model.treatment].values.reshape(-1,1) + y_bary * (1 - X_est[am_model.treatment].values.reshape(-1,1))
    y_hat_control = y_est_qtl * (1 - X_est[am_model.treatment].values.reshape(-1,1)) + y_bary * (X_est[am_model.treatment].values.reshape(-1,1))
    ate_hat = (y_hat_treated - y_hat_control).mean(axis = 0)
    new_mg_df = mg_df.query('unit != matched_unit')
    new_mg_df[am_model.treatment] = 1 - new_mg_df[am_model.treatment].values
    new_malts_df_est = X_est.copy()
    new_malts_df_est[am_model.treatment] = 1 - new_malts_df_est[am_model.treatment].values
    y_bary_nn = am_model.barycenter_imputation(X_estimation = X_est,
                                            Y_estimation = y_est_qtl,
                                            MG = new_mg_df,
                                            qtl_id = True
                                                )
    mse_predicted_outcome = k/(k + 1) * (y_est_qtl - y_bary_nn)**2
    variance_hat = ((y_hat_treated - y_hat_control - ate_hat)**2).mean(axis = 0) + (((n_times_matched/k)**2 + ((2 * k - 1)/k) * (n_times_matched/k)).reshape(-1,1) * mse_predicted_outcome).mean(axis = 0)
    
    print(directory, 'getting bias')
    # estimated_bias
    np.random.seed(999)
    wrf_control = wstree.wass_forest(X = X_train.query(am_model.treatment + ' == 0'), 
                        y = y_train_qtl[np.where(X_train[am_model.treatment].values == 0)[0], :],
                        y_quantile_id=True,
                        min_samples_split=None,
                        max_depth=20,
                        depth=None,
                        node_type=None,
                        n_trees=100,
                        seed=999,
                        n_samples_min=None)

    # train treated random forest
    wrf_treated = wstree.wass_forest(X = X_train.query(am_model.treatment + ' == 1'), 
                        y = y_train_qtl[np.where(X_train[am_model.treatment].values == 1)[0], :],
                    y_quantile_id=True,
                    min_samples_split=None,
                    max_depth=20,
                    depth=None,
                    node_type=None,
                    n_trees=100,
                    seed=999,
                    n_samples_min=None)
    y_wrf_control = []
    y_wrf_treated = []
    for i in X_est.index.values:
        y_wrf_control.append(wrf_control.predict(X_est.loc[i:i].assign(T = 0)))
        y_wrf_treated.append(wrf_treated.predict(X_est.loc[i:i].assign(T = 1)))
    y_wrf_control = np.array(y_wrf_control)
    y_wrf_treated = np.array(y_wrf_treated)
    estimated_bias = np.zeros(shape = qtl_grid.shape)
    for i in X_est.index.values:
        # find nearest neighbors
        nn_i = mg_df.query('unit == ' + str(i)).query(am_model.treatment + ' == unit_treatment').query('unit != matched_unit')
        treatment_i = X_est.loc[i, am_model.treatment]
        bias_i = []
        for j in nn_i.matched_unit.values:
            if treatment_i == 1: # if unit i is treated, get control conditional mean
                bias_i.append(y_wrf_control[i, :] - y_wrf_control[j, :])
            if treatment_i == 0: # if unit i is control, get treated conditional mean
                bias_i.append(y_wrf_treated[i, :] - y_wrf_treated[j, :])
        bias_i = np.array(bias_i).mean(axis = 0)
        estimated_bias += ((2 * treatment_i - 1) * bias_i).flatten()
    estimated_bias = estimated_bias / X_est.shape[0]

    alpha = 0.05

    lower_bound = (ate_hat - estimated_bias) - ndtri(1 - alpha/2) * np.sqrt(1/X_est.shape[0]) * np.sqrt(variance_hat)
    upper_bound = (ate_hat - estimated_bias) + ndtri(1 - alpha/2) * np.sqrt(1/X_est.shape[0]) * np.sqrt(variance_hat)

    pd.DataFrame({'lower_bound' : lower_bound, 
                  'upper_bound' : upper_bound, 
                  'estimated_bias' : estimated_bias,
                  'variance_hat' : variance_hat,
                  'ate_hat' : ate_hat}).to_csv(directory + '/abadie_imbens_pw_intervals.csv')
    pointwise_coverage = ((ate_true <= upper_bound) * (ate_true >= lower_bound)).mean()
    print(dataset_directory, seed, pointwise_coverage)
    return pointwise_coverage
    

def main(argv):
    dataset_directory = sys.argv[1]
    dataset_iterations_to_conduct = range(100)
    coverage_list = []
    for dataset_iteration in dataset_iterations_to_conduct:
        coverage_list.append(abadie_imbdens(dataset_iteration, dataset_directory))
    coverage = np.vstack(coverage_list)
    pd.DataFrame(coverage, columns = ['pointwise_coverage']).to_csv(dataset_directory + '/coverage_smoothness.csv', index = False)
    
if __name__ == '__main__':
    main(sys.argv)