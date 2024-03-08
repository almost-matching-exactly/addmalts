import numpy as np
import pandas as pd
import pyaddmalts as pam
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.stats import truncnorm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score
from multiprocessing import Pool


def make_pi(xv, yv):
    return 1/(1 + np.exp(-(0.5 * xv + 0.5 * yv))) * ((xv >= -0.5) + (yv >= -0.5))# + (yv <= -0.5) + (yv >= 0.5))

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
    
def sigma2_1(x_i, error, dataset_directory):
    if dataset_directory == './experiments/trunc_normal_constants':
        return 1
    elif dataset_directory == './experiments/trunc_normal_linear_same':
        return 1
    elif dataset_directory == './experiments/trunc_normal_linear_diff':
        return 1
    elif dataset_directory == './experiments/trunc_normal_complex':
        return 1
    elif dataset_directory == './experiments/trunc_normal_quadratic':
        return 1
    elif dataset_directory == './experiments/trunc_normal_variance':
        return np.abs(10 + x_i[0] + 2 * x_i[1] + error)
    
def sigma2_0(x_i, error, dataset_directory):
    if dataset_directory == './experiments/trunc_normal_constants':
        return 1
    elif dataset_directory == './experiments/trunc_normal_linear_same':
        return 1
    elif dataset_directory == './experiments/trunc_normal_linear_diff':
        return 1
    elif dataset_directory == './experiments/trunc_normal_complex':
        return 1
    elif dataset_directory == './experiments/trunc_normal_quadratic':
        return 1
    elif dataset_directory == './experiments/trunc_normal_variance':
        return np.abs(10 + x_i[0] + 2 * x_i[1] + 10 + error)
    

def quantile_trunc_normal(x_i, error, q, t, dataset_directory, a = -3, b = 3):
    '''
    find q-th quantile of truncated normal distribution
    '''
    mu = mu_1(x_i, error, dataset_directory) * t + mu_0(x_i, error, dataset_directory) * (1 - t)
    sigma2 = sigma2_1(x_i, error, dataset_directory) * t + sigma2_0(x_i, error, dataset_directory) * (1 - t)
    sigma = np.sqrt(sigma2)
    return truncnorm.ppf(q = q, a = a, b = b, loc = mu, scale = sigma)


def positivity_exp(seed):
    print(seed)
    np.random.seed(2020 + 1000 * seed)
    n_units = 2500
    p_vars = 2
    m = 1001
    dataset_directory = './experiments/trunc_normal_linear_same'
    X = np.random.uniform(low = -1, high = 1, size = [n_units, p_vars]) 
    minmax_X_grid = np.array([np.array(i) for i in itertools.product([-1, 1], repeat=p_vars)])
    epsilon = np.random.normal(loc = 0, scale = 1, size = n_units)
    # assign treatment
    pi = make_pi(X[:, 0], X[:, 1])
    treatment = np.random.binomial(n = 1, p = pi, size = n_units)
    # get true quantile functions
    qtl_grid = np.linspace(0, 1, m)
    Q1 = np.array([quantile_trunc_normal(x_i = X[i, :], error = epsilon[i], q = qtl_grid, dataset_directory = dataset_directory, t = 1) for i in range(n_units)])
    Q0 = np.array([quantile_trunc_normal(x_i = X[i, :], error = epsilon[i], q = qtl_grid, dataset_directory = dataset_directory, t = 0) for i in range(n_units)])

    y1 = np.array([quantile_trunc_normal(x_i = X[i, :], 
                                        error = epsilon[i],
                                        q = np.random.uniform(0, 1, size = m), 
                                        dataset_directory = dataset_directory,
                                        t = 1) for i in range(n_units)])
    y0 = np.array([quantile_trunc_normal(x_i = X[i, :], 
                                        error = epsilon[i], 
                                        q = np.random.uniform(0, 1, size = m), 
                                        dataset_directory = dataset_directory,
                                        t = 0) for i in range(n_units)])
    y = np.array([y1[i, :] * treatment[i] + y0[i, :] * (1 - treatment[i]) for i in range(n_units)])

    # split into training and estimation
    np.random.seed(999)
    train_indexes = np.random.binomial(n = 1, p = 0.6, size = n_units)
    est_indexes = 1 - train_indexes

    train_indexes = np.where(train_indexes == 1)[0]
    est_indexes = np.where(est_indexes == 1)[0]

    X_train = X[train_indexes, :]
    X_est = X[est_indexes, :]

    treatment_train = treatment[train_indexes]
    treatment_est = treatment[est_indexes]

    y_train = y[train_indexes, :]
    y_est = y[est_indexes, :]
    pi_est = pi[est_indexes]
    prune_est = (pi_est == 0)

    # train add malts
    addmalts = pam.pyaddmalts(X = pd.DataFrame(X_train, columns = [f'X_{_}' for _ in range(X_train.shape[1])]).assign(A = treatment_train),
                                y = y_train, 
                                treatment = 'A', 
                                discrete = [],
                                C = 0.1,
                                k = 10, # 10 NN matching 
                                y_qtl_id = False)
    print('fitting')
    addmalts.fit(method = 'SLSQP')


    # save addmalts
    # pkl_file = open(dataset_directory + '/dataset_' + str(seed) + '/malts_model.pkl', 'wb')


    mg_df = addmalts.get_matched_groups(X_estimation=pd.DataFrame(X_est, columns = [f'X_{_}' for _ in range(X_train.shape[1])]).assign(A = treatment_est),
                                    Y_estimation= y_est,
                                    k = 10)

    distances = mg_df.query('unit_treatment != A').groupby('unit').distance.mean()
    threshold = np.quantile(distances, 0.75) + 1.5 * (np.quantile(distances, 0.75) - np.quantile(distances, 0.25))
    prune_addmalts = distances >= threshold


    lasso_model = LogisticRegressionCV()
    lasso_model.fit(X_train, treatment_train)
    propensity_lasso = lasso_model.predict_proba(X_est)[:, 1]
    prune_lasso = (propensity_lasso <= 0.1) + (propensity_lasso >= 0.9)


    param_grid = {
        'n_estimators': [20, 25, 100, 200]
    }
    rf_model = RandomForestRegressor()

    grid_clf = GridSearchCV(rf_model, param_grid, cv=10)
    grid_clf.fit(X_train, treatment_train)

    propensity_rf = grid_clf.best_estimator_.predict(X_est)
    prune_rf = (propensity_rf <= 0.1) + (propensity_rf >= 0.9)

    # accuracy
    addmalts_accuracy = (prune_est == prune_addmalts).mean()
    lasso_accuracy = (prune_est == prune_lasso).mean()
    rf_accuracy = (prune_est == prune_rf).mean()

    # recall score
    addmalts_recall = recall_score(prune_est, prune_addmalts)
    lasso_recall = recall_score(prune_est, prune_lasso)
    rf_recall = recall_score(prune_est, prune_rf)


    # precision
    addmalts_precision = precision_score(prune_est, prune_addmalts)
    lasso_precision = precision_score(prune_est, prune_lasso)
    rf_precision = precision_score(prune_est, prune_rf)

    print(seed, addmalts_accuracy, lasso_accuracy, rf_accuracy, addmalts_precision, lasso_precision, rf_precision, addmalts_recall, lasso_recall, rf_recall)
    return pd.DataFrame({'prune_est' : prune_est,
                         'prune_addmalts' : prune_addmalts,
                         'prune_lasso' : prune_lasso,
                         'prune_rf' : prune_rf,
                         'ps_lasso' : propensity_lasso,
                         'ps_rf' : propensity_rf,
                         'X0' : X_est[:, 0],
                         'X1' : X_est[:, 1],
                         'df' : seed})

def main():
    dataset_iterations_to_conduct = range(100)
    with Pool(processes = 40) as pool:
        positivity_list = pool.map(positivity_exp, dataset_iterations_to_conduct)

    positivity_df = pd.concat(positivity_list, axis = 0)
    positivity_df.to_csv('./experiments/positivity.csv', index = False)
    
    
if __name__ == '__main__':
    main()