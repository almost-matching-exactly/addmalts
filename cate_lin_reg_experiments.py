import pandas as pd
import numpy as np
from util import CATE
import pickle as pkl
from multiprocessing import Pool
from sklearn.linear_model import LinearRegression
import sys
from functools import partial

class ols_quantile():
    def __init__(self, qtl_grid, treatment_var):
        self.qtl_grid = np.array(qtl_grid)
        self.treatment_var = treatment_var
    
    def fit(self, X_train, Y_train):
        A_train = X_train[self.treatment_var].values
        X_train = X_train.drop(self.treatment_var, axis = 1)
        
        control = np.where(A_train == 0)[0]
        X_control = X_train.iloc[control, :]
        Y_control = Y_train[control, :]
        
        treated = np.where(A_train == 1)[0]
        X_treated = X_train.iloc[treated, :]
        Y_treated = Y_train[treated, :]
        
        self.OLS_control_list = []
        self.OLS_treated_list = []
        for q in self.qtl_grid:
            OLS_control = LinearRegression()
            OLS_control.fit(X_control, Y_control[:, q])
            self.OLS_control_list.append(OLS_control)
            
            OLS_treated = LinearRegression()
            OLS_treated.fit(X_treated, Y_treated[:, q])
            self.OLS_treated_list.append(OLS_treated)
        self.OLS_control_list = np.array(self.OLS_control_list)
        self.OLS_treated_list = np.array(self.OLS_treated_list)
    
    def predict_control(self, X_est):
        X_est = X_est.drop(self.treatment_var, axis = 1)
        Y_predict_control = np.ones(shape = [X_est.shape[0], self.qtl_grid.shape[0]])
        print(self.OLS_control_list.shape)
        for q in self.qtl_grid:
            Y_predict_control[:, q] = self.OLS_control_list[q].predict(X_est).reshape(Y_predict_control[:, q].shape)
        
        Y_predict_control = Y_predict_control.reshape([X_est.shape[0], self.qtl_grid.shape[0]])
        
        Y_predict_control = np.array(Y_predict_control)
        return Y_predict_control
            
    def predict_treated(self, X_est):
        X_est = X_est.drop(self.treatment_var, axis = 1)
        Y_predict_treated = np.ones(shape = [X_est.shape[0], self.qtl_grid.shape[0]])
        print(self.OLS_treated_list.shape)
        for q in self.qtl_grid:
            Y_predict_treated[:, q] = self.OLS_treated_list[q].predict(X_est).reshape(Y_predict_treated[:, q].shape)
        # Y_predict_treated = np.apply_along_axis(func1d = lambda model: model.predict(X_est), axis = 0, arr = self.OLS_treated_list)
        Y_predict_treated = Y_predict_treated.reshape([X_est.shape[0], self.qtl_grid.shape[0]])
        return Y_predict_treated
        
        
def ols_meta_predict(X_valid, lin_reg, treatment_var):
    y_pred = []


    A = X_valid[treatment_var].values
    A_repeat = np.array([np.repeat(A_i, repeats = lin_reg.qtl_grid.shape[0], axis = 0) for A_i in A])
    X_valid[treatment_var] = 0
    y_pred_control = lin_reg.predict_control(X_valid)
    X_valid[treatment_var] = 1
    y_pred_treated = lin_reg.predict_treated(X_valid)
    print(y_pred_treated.shape, y_pred_control.shape)
    y_pred = (1 - A_repeat) * y_pred_treated + A_repeat * y_pred_control
    return y_pred

### Wasserstein regression experiments
def lin_reg_parallel(dataset_iteration, dataset_directory):
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

    print('training')
    
    # train control and treated regressions
    lin_reg = ols_quantile(qtl_grid=qtls, treatment_var='A')
    lin_reg.fit(X_train, y_train)


    # save OLS models
    treated_file_name = open(dataset_directory + '/dataset_' + str(seed) + '/lin_reg.pkl', 'wb')
    pkl.dump(lin_reg, treated_file_name)

    print('imputing counterfactuals')
    # impute counterfactuals
    y_lin_reg_bary = ols_meta_predict(X_est, lin_reg, treatment_var = 'A').reshape(y_est.shape)

    print('estimating treatment effects')
    # estimate the CATE with imputed counterfactuals
    CATE_lin_reg = []
    for i in range(X_est.shape[0]):
        CATE_lin_reg.append(
            CATE(
                y_obs = y_est[i, :],
                y_cf = y_lin_reg_bary[i, :],
                observed_treatment = X_est['A'].values[i],
                y_obs_qtl_id = False, 
                y_cf_qtl_id = True
            )
        )
    CATE_lin_reg = np.array(CATE_lin_reg)
    ATE_lin_reg = CATE_lin_reg.mean(axis = 1)
    print(dataset_iteration, ': WassOLS ATE min =', ATE_lin_reg.min(), ' ATE max = ', ATE_lin_reg.max())

    print('saving data')
    lin_reg_CATE_df = pd.DataFrame(CATE_lin_reg, columns = np.linspace(0, 1, CATE_lin_reg.shape[0]))
    lin_reg_CATE_df.to_csv(dataset_directory + '/dataset_' + str(seed) + '/lin_reg_CATE.csv')
  
def main(argv):
    dataset_directory = sys.argv[1]
    dataset_iterations_to_conduct = range(0, 100)
    with Pool(processes = 25) as pool:
        pool.map(partial(lin_reg_parallel, dataset_directory = dataset_directory),
                dataset_iterations_to_conduct)


if __name__ == '__main__':
    main(sys.argv)