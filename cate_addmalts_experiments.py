import pyaddmalts as pam
from util import CATE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle as pkl
from multiprocessing import Pool
import time
from functools import partial

def addmalts_parallel(dataset_iteration, dataset_directory):
    print(dataset_iteration)
    seed = 2020 + 1000 * dataset_iteration
    
    # read dataset
    X_train = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/X_train.csv')
    X_est = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/X_est.csv')
    y_train = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/y_train.csv')
    y_est = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/y_est.csv')


    y_train = y_train.to_numpy()
    y_est = y_est.to_numpy()

    if 'dist_cov' not in dataset_directory:
        X_train = X_train.reset_index()
        X_est = X_est.reset_index()
        Q_x_train = np.array([[[]]])
        Q_x_est = np.array([[[]]])
    else:
        # take inverse of Xs to get upper bounds
        upper_train = (X_train.drop('A', axis = 1).to_numpy() + 1)/2
        upper_est = (X_est.drop('A', axis = 1).to_numpy() + 1)/2

        # make quantile covariate using the first feature's upper bound
        Q_x_train = np.linspace(0, upper_train[:, 0], y_train.shape[1]).transpose().reshape(upper_train.shape[0], 1, y_train.shape[1])
        Q_x_est = np.linspace(0, upper_est[:, 0], y_est.shape[1]).transpose().reshape(upper_est.shape[0], 1, y_est.shape[1])

        X_train = X_train[['A'] + ['X' + str(_) for _ in range(1, upper_train.shape[1])]]
        X_est = X_est[['A'] + ['X' + str(_) for _ in range(1, upper_train.shape[1])]]

    # run MALTSPro
    print(dataset_directory, dataset_iteration, 'initializing...', time.time())
    addmalts = pam.pyaddmalts(X = X_train,
                            y = y_train, 
                            X_quantile_fns = Q_x_train,
                            treatment = 'A', 
                            discrete = [],
                            C = 1,
                            k = 10, # k NN matching 
                            y_qtl_id = False)

    print(dataset_directory, dataset_iteration, 'fitting...', time.time())
    print(addmalts.fit(method = 'SLSQP'))
    print(dataset_directory, dataset_iteration, 'finished fitting...', time.time())

    # save addmalts
    pkl_file = open(dataset_directory + '/dataset_' + str(seed) + '/malts_model.pkl', 'wb')
    pkl.dump(addmalts, file = pkl_file)

    # with open(dataset_directory + '/dataset_' + str(seed) + '/malts_model.pkl', 'rb') as f:
    #     addmalts = pkl.load(f)
    # print(dataset_iteration, 'getting matched groups')
    # get matched groups
    mg_df = addmalts.get_matched_groups(X_estimation=X_est,
                                        Y_estimation= y_est,
                                        X_qtl_estimation=Q_x_est,
                                        k = 10)
    
    print(dataset_directory, dataset_iteration, 'getting CATE', time.time())
    

    y_bary = addmalts.barycenter_imputation(X_estimation = X_est,
                                            Y_estimation = y_est,
                                            MG = mg_df, 
                                            qtl_id = False)

    CATE_addmalts = []
    for i in range(X_est.shape[0]):
        CATE_addmalts.append(
            CATE(
                y_obs = y_est[i, :],
                y_cf = y_bary[i, :],
                observed_treatment = X_est['A'].values[i],
                y_obs_qtl_id = False, 
                y_cf_qtl_id = True
            )
        )
    CATE_addmalts = np.array(CATE_addmalts)
    
    addmalts_ITE_df = pd.DataFrame(CATE_addmalts, columns = np.linspace(0, 1, CATE_addmalts.shape[0]))
    addmalts_ITE_df.to_csv(dataset_directory + '/dataset_' + str(seed) + '/addmalts_CATE.csv')

    print(dataset_directory, dataset_iteration, 'done', time.time())
    del(addmalts)
    del(mg_df)
    del(CATE_addmalts)
    
def main(argv):
    dataset_directory = sys.argv[1]
    dataset_iterations_to_conduct = range(0, 100)
    with Pool(processes = 40) as pool:
        pool.map(partial(addmalts_parallel, dataset_directory = dataset_directory),
                dataset_iterations_to_conduct)


if __name__ == '__main__':
    main(sys.argv)