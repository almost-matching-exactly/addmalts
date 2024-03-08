import pandas as pd
import numpy as np
from util import sample_quantile
import pickle as pkl
from multiprocessing import Pool
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys
from functools import partial

class ols_quantile():
    def __init__(self, qtl_grid, treatment_var):
        '''
        description
        -----------
        frechet mean regression
        
        inputs
        ------
        qtl_grid : numpy array with quantiles to estimate outcome quantile function
        treatment_var : string with name of treatment variable
        '''
        self.qtl_grid = np.array(qtl_grid)
        self.treatment_var = treatment_var
    
    def fit(self, X_train, Y_train):
        '''
        description
        -----------
        train linear outcome model by fitting linear regression for each quantile
        
        inputs
        ------
        X_train : pandas dataframe with treatment variable and covariates
        Y_train : N_train by m numpy array with samples from outcome distribution
        '''
        # split data into control and treated data
        A_train = X_train[self.treatment_var].values
        X_train = X_train.drop(self.treatment_var, axis = 1)
        
        control = np.where(A_train == 0)[0]
        X_control = X_train.iloc[control, :]
        Y_control = Y_train[control, :]
        
        treated = np.where(A_train == 1)[0]
        X_treated = X_train.iloc[treated, :]
        Y_treated = Y_train[treated, :]
        
        # for each quantile, train a linear model
        self.OLS_control_list = []
        self.OLS_treated_list = []
        for q in range(self.qtl_grid.shape[0]):
            OLS_control = LinearRegression()
            OLS_control.fit(X_control, Y_control[:, q])
            self.OLS_control_list.append(OLS_control)
            
            OLS_treated = LinearRegression()
            OLS_treated.fit(X_treated, Y_treated[:, q])
            self.OLS_treated_list.append(OLS_treated)
        
        # save models
        self.OLS_control_list = np.array(self.OLS_control_list)
        self.OLS_treated_list = np.array(self.OLS_treated_list)
    
    def predict_control(self, X_est):
        
        # predict control potential outcome distribution
        X_est = X_est.drop(self.treatment_var, axis = 1)
        Y_predict_control = np.ones(shape = [X_est.shape[0], self.qtl_grid.shape[0]])
        
        for q in range(self.qtl_grid.shape[0]):
            Y_predict_control[:, q] = self.OLS_control_list[q].predict(X_est).reshape(Y_predict_control[:, q].shape)
        
        Y_predict_control = Y_predict_control.reshape([X_est.shape[0], self.qtl_grid.shape[0]])
        Y_predict_control = np.array(Y_predict_control)
        return Y_predict_control
            
    def predict_treated(self, X_est):
        
        # predict treated potential outcome distribution
        X_est = X_est.drop(self.treatment_var, axis = 1)
        Y_predict_treated = np.ones(shape = [X_est.shape[0], self.qtl_grid.shape[0]])
        
        for q in range(self.qtl_grid.shape[0]):
            Y_predict_treated[:, q] = self.OLS_treated_list[q].predict(X_est).reshape(Y_predict_treated[:, q].shape)
        
        Y_predict_treated = Y_predict_treated.reshape([X_est.shape[0], self.qtl_grid.shape[0]])
        return Y_predict_treated
        
        
def ols_meta_predict(X_valid, wass_ols, treatment_var):
    '''
    description
    -----------
    predict counterfactual outcome
    
    inputs
    ------
    X_valid : pandas dataframe of covariates for units that must be imputed
    wass_ols : frechet regression model
    treatment_var : string with name of treatment variable
    '''
    
    y_pred = []
    A = X_valid[treatment_var].values
    A_repeat = np.array([np.repeat(A_i, repeats = wass_ols.qtl_grid.shape[0], axis = 0) for A_i in A])
    X_valid[treatment_var] = 0
    
    # predict control outcome for all units
    y_pred_control = wass_ols.predict_control(X_valid)
    X_valid[treatment_var] = 1
    
    # predict treated outcome for all units
    y_pred_treated = wass_ols.predict_treated(X_valid)
    print(y_pred_treated.shape, y_pred_control.shape)
    
    # isolate counterfactual outcome
    y_pred = (1 - A_repeat) * y_pred_treated + A_repeat * y_pred_control
    return y_pred

class doubly_robust_method():
    def __init__(self, qtl_grid, treatment_var, propensity_model):
        '''
        description
        -----------
        doubly robust estimator
        
        inputs
        ------
        qtl_grid : numpy array with quantiles to estimate outcome quantile function
        treatment_var : string with name of treatment variable
        propensity_model : scikit learn classification model 
        '''
        self.qtl_grid = np.array(qtl_grid)
        self.treatment_var = treatment_var
        self.propensity_model = propensity_model
        
    def prop_score(self, X_train):
        '''
        description
        -----------
        Estimate propensity score using a logistic regression model
        
        inputs
        ------
        X_train : a dataframe containing the treatment variable and all covariates
        
        returns
        -------
        Saves propensity score model as class attribute
        '''
        # LR = LogisticRegression(penalty = 'none') # no regularization
        A = X_train[self.treatment_var].values
        X = X_train.drop(self.treatment_var, axis = 1)
        self.propensity_model.fit(X, A)
        
    
    def outcome_regression(self, X_train, Y_train):
        '''
        description
        -----------
        Fit outcome regression model using Frechet mean regression
        
        inputs
        ------
        X_train: a dataframe containing the treatment variable and all covariates
        Y_train: a n_train by m matrix of observed samples from outcome distribution
        '''
        wass_reg = ols_quantile(qtl_grid = self.qtl_grid, treatment_var = self.treatment_var)
        wass_reg.fit(X_train = X_train, Y_train = Y_train)
        self.outcome_reg_model = wass_reg
    
    def fit(self, X_train, Y_train):
        self.prop_score(X_train)
        self.outcome_regression(X_train, Y_train)
        
    def estimate_CATE(self, X_est, Y_est, reference_distribution):
        '''
        description
        -----------
        Perform augmented inverse propensity weighting with outcome regression and propensity score models
        
        inputs
        ------
        X_est : a dataframe containing the treatment variable and all covariates for estimation set
        Y_est : a n by m matrix of observed samples from outcome distribution for estimation set
        reference_distribution : a 2D array mapping samples from reference distribution to density of sample
			-- reference distribution _must_ be continuous
			-- col 1 is sample
			-- col 2 is prob of observing sample
        '''
        
        # estimate treated and control outcomes
        A = X_est[self.treatment_var].values
        X_est[self.treatment_var] = 1
        Y_hat_treated = self.outcome_reg_model.predict_treated(X_est=X_est)
        X_est[self.treatment_var] = 0
        Y_hat_control = self.outcome_reg_model.predict_control(X_est=X_est)
        
        # construct quantile function out of Y_est
        Y_est_qtl = np.array([np.quantile(Y_est[i, :], q = self.qtl_grid) for i in range(Y_est.shape[0])])
        
        # construct Y_hat \circ \lambda
        def outcome_model_w_reference(outcome):
            return np.array([sample_quantile(quantile_fn=outcome, quantile = q) for q in reference_distribution[1, :]])
        
        Y_hat_lambda = np.apply_along_axis(func1d = outcome_model_w_reference,
                                            axis = 1,
                                            arr = Y_est_qtl
                                            )
        
        # construct Y_hat_treated \circ \lambda
        m_hat_treated = np.apply_along_axis(func1d = outcome_model_w_reference,
                                            axis = 1,
                                            arr = Y_hat_treated
                                            )
        
        # construct Y_hat_control \circ \lambda
        m_hat_control = np.apply_along_axis(func1d = outcome_model_w_reference,
                                            axis = 1,
                                            arr = Y_hat_control
                                            )
        
        # construct propensity score
        prop_score = self.propensity_model.predict_proba(X_est.drop(self.treatment_var, axis = 1))[:, 1]
        prop_score_matrix = np.array([np.repeat(pi_i, repeats = m_hat_control.shape[1], axis = 0) for pi_i in prop_score])
        treatment_matrix  = np.array([np.repeat(A_i, repeats = m_hat_control.shape[1], axis = 0) for A_i in A])
        # get components of doubly robust
        mu_hat_treated = m_hat_treated + ((treatment_matrix == 1)/prop_score_matrix) * (Y_hat_lambda - m_hat_treated)
        mu_hat_control = m_hat_control + ((treatment_matrix == 0)/prop_score_matrix) * (Y_hat_lambda - m_hat_control)
        
        # estimate CATE
        CATE = mu_hat_treated - mu_hat_control
        
        # return CATEs
        return CATE
        
        
        
### Wasserstein regression experiments
def doubly_robust_parallel(dataset_iteration, dataset_directory):
    print(dataset_iteration, end = ' ')
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

    # maltspro_df = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/X.csv')
    # y = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/Y.csv').to_numpy()

    # turn into quantile functions
    #   1. find #quantiles
    n_samples_min = np.apply_along_axis(
                arr = np.vstack([y_train, y_est]),
                axis = 1,
                func1d = lambda x: x[x == x].shape[0]).min()
    #   2. estimate quantile function at grid
    qtls = range(n_samples_min)
    y_train = np.apply_along_axis(
                    arr = y_train,
                    axis = 1,
                    func1d = lambda x: np.quantile(
                        a = x, 
                        q = np.linspace(start = 0, stop = 1, num = n_samples_min)
                        )
                    ).reshape(y_train.shape)
    
    y_est = np.apply_along_axis(
                    arr = y_est,
                    axis = 1,
                    func1d = lambda x: np.quantile(
                        a = x, 
                        q = np.linspace(start = 0, stop = 1, num = n_samples_min)
                        )
                    ).reshape(y_est.shape)

    # train control and treated regressions
    logit = LogisticRegression(penalty='none')
    doubly_robust = doubly_robust_method(qtl_grid=np.linspace(0, 1, n_samples_min), treatment_var='A', propensity_model=logit)
    doubly_robust.fit(X_train, y_train)
    CATE_doubly_robust = doubly_robust.estimate_CATE(X_est, 
                                                    y_est,
                                                    reference_distribution=np.vstack([np.linspace(0, 1, n_samples_min), 
                                                                                    np.linspace(0, 1, n_samples_min)]))


    # save doubly robust models
    treated_file_name = open(dataset_directory + '/dataset_' + str(seed) + '/doubly_robust_linps.pkl', 'wb')
    pkl.dump(doubly_robust, treated_file_name)

    ATE_doubly_robust = CATE_doubly_robust.mean(axis = 1)
    print(dataset_iteration, ': DR ATE min =', ATE_doubly_robust.min(), ' ATE max = ', ATE_doubly_robust.max())

    print('saving data')
    CATE_doubly_robust_df = pd.DataFrame(CATE_doubly_robust, columns = np.linspace(0, 1, CATE_doubly_robust.shape[1]))
    CATE_doubly_robust_df.to_csv(dataset_directory + '/dataset_' + str(seed) + '/dr_linps_CATE.csv')
  
def main(argv):
    dataset_directory = sys.argv[1]
    dataset_iterations_to_conduct = range(0, 100)
    with Pool(processes = 25) as pool:
        pool.map(partial(doubly_robust_parallel, dataset_directory = dataset_directory),
                dataset_iterations_to_conduct)


if __name__ == '__main__':
    main(sys.argv)