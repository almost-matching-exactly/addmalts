import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from sklearn.preprocessing import PolynomialFeatures
import os
import sys, getopt
from multiprocessing import Pool
from functools import partial
import itertools

# trunc_normal_linear_same, trunc_normal_linear_diff, trunc_normal_complex, trunc_normal_variance
def mu_1(x_i, error, dataset_directory):
    if dataset_directory == './experiments/trunc_normal_constants':
        return 10
    elif dataset_directory == './experiments/trunc_normal_linear_same':
        return 10 + x_i[0] + 2 * x_i[1] + error
    elif dataset_directory == './experiments/trunc_normal_linear_diff':
        return 10 + x_i[0] + 2 * x_i[1] + 10 + error
    elif dataset_directory == './experiments/trunc_normal_complex':
        return 10 * np.sin(np.pi * x_i[0] * x_i[1]) + 20 * (x_i[2] - 0.5)**2 + 10 * x_i[3] + 5 * x_i[4] + 7 + x_i[2] * np.cos(np.pi * x_i[0] * x_i[1]) + error
    elif dataset_directory == './experiments/trunc_normal_quadratic':
        poly = PolynomialFeatures(degree = 2)
        poly_features = poly.fit_transform(x_i[1:5].reshape(1, -1))
        return 10 + x_i[0:5].sum() + 10 + poly_features.sum() + error
    elif dataset_directory == './experiments/trunc_normal_variance':
        return 10 + x_i[0] + 2 * x_i[1] + error
    elif dataset_directory == './experiments/trunc_normal_variance_quadratic':
        poly = PolynomialFeatures(degree = 2)
        poly_features = poly.fit_transform(x_i[0:5].reshape(1, -1))
        return 10 + x_i[0:5].sum() + 10 + poly_features.sum() + error
    
def mu_0(x_i, error, dataset_directory):
    if dataset_directory == './experiments/trunc_normal_constants':
        return 0
    elif dataset_directory == './experiments/trunc_normal_linear_same':
        return 10 + x_i[0] + 2 * x_i[1] + error
    elif dataset_directory == './experiments/trunc_normal_linear_diff':
        return 10 + x_i[0] + 2 * x_i[1] + error
    elif dataset_directory == './experiments/trunc_normal_complex':
        return 10 * np.sin(np.pi * x_i[0] * x_i[1]) + 20 * (x_i[2] - 0.5)**2 + 10 * x_i[3] + 5 * x_i[4] + error
    elif dataset_directory == './experiments/trunc_normal_quadratic':
        return 10 + x_i[0:5].sum() + error
    elif dataset_directory == './experiments/trunc_normal_variance':
        return 10 + x_i[0] + 2 * x_i[1] + error
    elif dataset_directory == './experiments/trunc_normal_variance_quadratic':
        return 10 + x_i[0:5].sum() + error
    
def sigma2_1(x_i, error, dataset_directory):
    if dataset_directory == './experiments/trunc_normal_constants':
        return 1
    elif dataset_directory == './experiments/trunc_normal_linear_same':
        return 1
    elif dataset_directory == './experiments/trunc_normal_linear_diff':
        return 1
    elif dataset_directory == './experiments/trunc_normal_complex':
        return np.abs(10 + x_i[0] + 2 * x_i[1] + error)
    elif dataset_directory == './experiments/trunc_normal_quadratic':
        return 1
    elif dataset_directory == './experiments/trunc_normal_variance':
        return np.abs(10 + x_i[0] + 2 * x_i[1] + 10 + error)
    elif dataset_directory == './experiments/trunc_normal_variance_quadratic':
        poly = PolynomialFeatures(degree = 2)
        poly_features = poly.fit_transform(x_i[0:5].reshape(1, -1))
        return np.abs(10 + x_i[0:5].sum() + 10 + poly_features.sum() + error)
    
def sigma2_0(x_i, error, dataset_directory):
    if dataset_directory == './experiments/trunc_normal_constants':
        return 1
    elif dataset_directory == './experiments/trunc_normal_linear_same':
        return 1
    elif dataset_directory == './experiments/trunc_normal_linear_diff':
        return 1
    elif dataset_directory == './experiments/trunc_normal_complex':
        return np.abs(10 + x_i[0] + 2 * x_i[1] + error)
    elif dataset_directory == './experiments/trunc_normal_quadratic':
        return 1
    elif dataset_directory == './experiments/trunc_normal_variance':
        return np.abs(10 + x_i[0] + 2 * x_i[1] + error)
    elif dataset_directory == './experiments/trunc_normal_variance_quadratic':
        return np.abs(10 + x_i[0:5].sum() + error)
    
def quantile_trunc_normal(x_i, error, q, t, dataset_directory, a = -3, b = 3):
    '''
    find q-th quantile of truncated normal distribution
    '''
    mu = mu_1(x_i, error, dataset_directory) * t + mu_0(x_i, error, dataset_directory) * (1 - t)
    sigma2 = sigma2_1(x_i, error, dataset_directory) * t + sigma2_0(x_i, error, dataset_directory) * (1 - t)
    sigma = np.sqrt(sigma2)
    return truncnorm.ppf(q = q, a = a, b = b, loc = mu, scale = sigma)


def make_data(dataset_iteration, dataset_directory):
    print(dataset_directory, dataset_iteration)
    seed = 2020 + 1000 * dataset_iteration
    np.random.seed(seed)
    n = 1500 # have 1500 units
    m = 1001 # have 1001 outcomes per unit
    
    p_vars = 2
    if dataset_directory == './experiments/trunc_normal_complex':
        p_vars = 6

    X = np.random.uniform(low = -1, high = 1, size = [n, p_vars]) 
    minmax_X_grid = np.array([np.array(i) for i in itertools.product([-1, 1], repeat=p_vars)])
    epsilon = np.random.normal(loc = 0, scale = 1, size = n)
    # assign treatment
    treatment = np.random.binomial(n = 1, p = 1/(1 + np.exp(-1 * (X[:, 0] + X[:, 1]))), size = n)
    # get true quantile functions
    qtl_grid = np.linspace(0, 1, m)
    Q1 = np.array([quantile_trunc_normal(x_i = X[i, :], error = epsilon[i], q = qtl_grid, dataset_directory = dataset_directory, t = 1) for i in range(n)])
    Q0 = np.array([quantile_trunc_normal(x_i = X[i, :], error = epsilon[i], q = qtl_grid, dataset_directory = dataset_directory, t = 0) for i in range(n)])
    # get true CATE
    tau_i = Q1 - Q0
    # get true ATE
    Q1_grid = np.array([quantile_trunc_normal(x_i = minmax_X_grid[i, :], error = 0, q = qtl_grid, dataset_directory = dataset_directory, t = 1) for i in range(minmax_X_grid.shape[0])])
    Q0_grid = np.array([quantile_trunc_normal(x_i = minmax_X_grid[i, :], error = 0, q = qtl_grid, dataset_directory = dataset_directory, t = 0) for i in range(minmax_X_grid.shape[0])])
    tau_ate = (Q1_grid - Q0_grid).mean(axis = 0)
    
    
    # get outcome samples
    y1 = np.array([quantile_trunc_normal(x_i = X[i, :], 
                                        error = epsilon[i],
                                        q = np.random.uniform(0, 1, size = m), 
                                        dataset_directory = dataset_directory,
                                        t = 1) for i in range(n)])
    y0 = np.array([quantile_trunc_normal(x_i = X[i, :], 
                                        error = epsilon[i], 
                                        q = np.random.uniform(0, 1, size = m), 
                                        dataset_directory = dataset_directory,
                                        t = 0) for i in range(n)])
    y = np.array([y1[i, :] * treatment[i] + y0[i, :] * (1 - treatment[i]) for i in range(n)])
    # split into training and testing
    np.random.seed(999)
    train_calib_est_assign = np.random.choice(a = [0, 1, 2], size = n, replace = True, p = [0.6, 0, 0.4])
    train_indexes = np.where(train_calib_est_assign == 0)[0]
    calib_indexes = np.where(train_calib_est_assign == 1)[0]
    est_indexes = np.where(train_calib_est_assign == 2)[0]

    X_train = X[train_indexes, :]
    X_calib = X[calib_indexes, :]
    X_est = X[est_indexes, :]

    y_train = y[train_indexes, :]
    y_calib = y[calib_indexes, :]
    y_est = y[est_indexes, :]

    treatment_train = treatment[train_indexes]
    treatment_calib = treatment[calib_indexes]
    treatment_est   = treatment[est_indexes]

    tau_train = tau_i[train_indexes, :]
    tau_calib = tau_i[calib_indexes, :]
    tau_est   = tau_i[est_indexes, :]

    # write information out
    if 'dataset_' + str(seed) not in os.listdir(dataset_directory):
        os.mkdir(dataset_directory + '/dataset_' + str(seed))

    pd.DataFrame(X_train, columns = ['X' + str(i) for i in range(p_vars)]).assign(A = treatment_train).to_csv(dataset_directory + '/dataset_' + str(seed) + '/X_train.csv', index = False)
    pd.DataFrame(X_calib, columns = ['X' + str(i) for i in range(p_vars)]).assign(A = treatment_calib).to_csv(dataset_directory + '/dataset_' + str(seed) + '/X_calib.csv', index = False)
    pd.DataFrame(X_est, columns = ['X' + str(i) for i in range(p_vars)]).assign(A = treatment_est).to_csv(dataset_directory + '/dataset_' + str(seed) + '/X_est.csv', index = False)

    pd.DataFrame(y_train, columns = qtl_grid).to_csv(dataset_directory + '/dataset_' + str(seed) + '/y_train.csv', index = False)
    pd.DataFrame(y_calib, columns = qtl_grid).to_csv(dataset_directory + '/dataset_' + str(seed) + '/y_calib.csv', index = False)
    pd.DataFrame(y_est, columns = qtl_grid).to_csv(dataset_directory + '/dataset_' + str(seed) + '/y_est.csv', index = False)

    pd.DataFrame(tau_train, columns = qtl_grid).to_csv(dataset_directory + '/dataset_' + str(seed) + '/ITE_train.csv', index = False)
    pd.DataFrame(tau_calib, columns = qtl_grid).to_csv(dataset_directory + '/dataset_' + str(seed) + '/ITE_calib.csv', index = False)
    pd.DataFrame(tau_est, columns = qtl_grid).to_csv(dataset_directory + '/dataset_' + str(seed) + '/ITE_est.csv', index = False)

    pd.DataFrame(tau_ate.reshape(1, m), columns = qtl_grid).to_csv(dataset_directory + '/dataset_' + str(seed) + '/tau_ate.csv', index = False)

def main(argv):
    dataset_directory = sys.argv[1]
    if 'experiments' not in os.listdir('./'):
        os.mkdir('./experiments/')
    if dataset_directory not in os.listdir('./experiments/'):
        os.mkdir('./experiments/' + dataset_directory)
    dataset_directory = './experiments/' + dataset_directory
    dataset_iterations_to_conduct = range(0, 100)
    with Pool(processes = 25) as pool:
        pool.map(partial(make_data, dataset_directory=dataset_directory), dataset_iterations_to_conduct)

if __name__ == '__main__':
    main(sys.argv)