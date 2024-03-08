#!/usr/bin/env python
# coding: utf-8

# # Methods and Code base for Wasserstein Decision Trees and Wasserstein Random Forests


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import wasserstein_dist, wasserstein2_barycenter


class wass_node:
    def __init__(self, X, y, y_quantile_id = False, min_samples_split=None, max_depth=None, depth=None, node_type=None, n_samples_min = None):
        # save data
        self.X = X.reset_index().drop('index', axis = 1)
        self.y = y
        self.features = X.columns
        
        self.y_quantile_id = y_quantile_id
        # find min number of samples in any y_i
        if n_samples_min is None:
            self.n_samples_min = np.apply_along_axis(
                arr = self.y,
                axis = 1,
                func1d = lambda x: x[x == x].shape[0]).min()
        else:
            self.n_samples_min = n_samples_min
        
        # save hyperparameters: defaults are 20 and 5 unless specified otherwise
        if min_samples_split is None:
            self.min_samples_split = 20
        else:
            self.min_samples_split = min_samples_split
        if max_depth is None:
            self.max_depth = 5
        else:
            self.max_depth = max_depth
#         self.max_depth = self.max_depth if max_depth is not None else 5
        
        # current node characteristics: defaults are 0 depth and 'root' node
        if depth is None:
            self.depth = 0
        else:
            self.depth = depth
        if node_type is None:
            self.node_type = 'root'
        else:
            self.node_type = node_type
        
        # number of units in the node
        self.n_units = X.shape[0]
        
        # get barycenter of current node
        self.barycenter = wasserstein2_barycenter(sample_array_1_through_n = y,
                                                  qtl_id=y_quantile_id,  
                                                  n_samples_min = self.n_samples_min, 
                                                  weights = np.repeat(a = 1/self.n_units, repeats = self.n_units)
                                                 )
        
        # calculate mean squared wasserstein error: 1/N sum_i W_2(bary, y_i)^2
        self.mswe = self.self_mswe_calc()
        
        # initialize feature/value to split on
        self.best_feature = None
        self.best_feature_split_val = None
        
        # initializing child nodes
        self.left_node = None
        self.right_node = None
        
    def self_mswe_calc(self):
        mswe = np.apply_along_axis(arr = self.y, 
            axis = 1, 
            func1d = lambda x: wasserstein_dist(sample_array1 = x, 
                                                sample_array2 = self.barycenter, 
                                                p = 2, 
                                                n_samples_min = self.n_samples_min, 
                                                array1_quantile = self.y_quantile_id,
                                                array2_quantile = True
                                               ) ** 2
           ).mean()
        return mswe

        
        
    def _mswe(self, y_true, y_pred, y_true_quantile_id = False, y_pred_quantile_id = True):
        squared_wass_dist_array = np.apply_along_axis(
            arr = y_true,
            axis = 1,
            func1d = lambda x: wasserstein_dist(sample_array1 = x,
                                                sample_array2 = y_pred, 
                                                p = 2,
                                                n_samples_min = self.n_samples_min, 
                                                array1_quantile = y_true_quantile_id,
                                                array2_quantile = y_pred_quantile_id, 
                                               ) ** 2
        )
        mswe = squared_wass_dist_array.mean()
        return mswe
    
    def best_split(self):
        
        mswe_base = self.mswe
        best_feature = None
        best_feature_split_val = None
        best_mswe = np.inf
        
        for feature in self.features:
            # split each feature at its median
            feature_split_val = self.X[feature].mean()
            
            # left node would be whenever x_feature <= median
            X_left = self.X.loc[self.X[feature] <= feature_split_val]
            X_right = self.X.loc[self.X[feature] > feature_split_val]
            
            y_left = self.y[X_left.index]
            y_right = self.y[X_right.index]
            
            if X_left.shape[0] > 0 and X_right.shape[0] > 0:
                # calculate barycenter of left and right nodes
                y_left_bary = wasserstein2_barycenter(sample_array_1_through_n = y_left,
                                                      weights = np.repeat(a = 1/X_left.shape[0], repeats = X_left.shape[0]), 
                                                      n_samples_min = self.n_samples_min,
                                                      qtl_id = self.y_quantile_id
                                                     )
                # calculate barycenter of left and right nodes
                y_right_bary = wasserstein2_barycenter(sample_array_1_through_n = y_right,
                                                      weights = np.repeat(a = 1/X_right.shape[0], repeats = X_right.shape[0]), 
                                                      n_samples_min = self.n_samples_min,
                                                      qtl_id = self.y_quantile_id
                                                     )

                left_mswe = np.apply_along_axis(
                    arr = y_left, 
                    axis = 1, 
                    func1d = lambda x: wasserstein_dist(sample_array1 = x, 
                                                        sample_array2 = y_left_bary, 
                                                        p = 2, 
                                                        n_samples_min = self.n_samples_min, 
                                                        array1_quantile = self.y_quantile_id,
                                                        array2_quantile = True
                                                       ) ** 2
                   ).mean()
                # calculate mswe for left and right nodes
                right_mswe = np.apply_along_axis(
                    arr = y_right, 
                    axis = 1, 
                    func1d = lambda x: wasserstein_dist(sample_array1 = x, 
                                                        sample_array2 = y_right_bary, 
                                                        p = 2, 
                                                        n_samples_min = self.n_samples_min, 
                                                        array1_quantile = self.y_quantile_id,
                                                        array2_quantile = True
                                                       ) ** 2
                   ).mean()

                # average both mswe together
                total_mswe = ((X_left.shape[0]) * left_mswe + (X_right.shape[0]) * right_mswe)/self.X.shape[0]

                # check if we improved mswe
                if total_mswe < best_mswe:
                    best_feature = feature
                    best_feature_split_val = feature_split_val
                    best_mswe = total_mswe
#                 print(feature, total_mswe)
        return (best_feature, best_feature_split_val)
    
    def grow_tree(self):
        if (self.depth < self.max_depth) and (self.n_units > self.min_samples_split):
            best_feature, best_feature_split_val = self.best_split()
            if best_feature is not None:
                self.best_feature = best_feature
                self.best_feature_split_val = best_feature_split_val
                
                X_left = self.X.loc[self.X[best_feature] <= best_feature_split_val]
                X_right = self.X.loc[self.X[best_feature] > best_feature_split_val]

                y_left = self.y[X_left.index, :]
                y_right = self.y[X_right.index, :]

                left = wass_node(
                    X = X_left,
                    y = y_left,
                    y_quantile_id = self.y_quantile_id,
                    min_samples_split = self.min_samples_split,
                    max_depth = self.max_depth,
                    depth = self.depth + 1,
                    node_type = 'left_node'
                )

                if left is not None:
                    self.left_node = left
                    try:
                        self.left_node.grow_tree()
                    except:
                        print(self.left_node.X.shape)

                right = wass_node(
                    X = X_right,
                    y = y_right,
                    y_quantile_id = self.y_quantile_id,
                    min_samples_split = self.min_samples_split,
                    max_depth = self.max_depth,
                    depth = self.depth + 1,
                    node_type = 'right_node'
                )

                if right is not None:
                    self.right_node = right
                    self.right_node.grow_tree()
                
    def print_info(self, width=4):
        """
        Method to print the infromation about the tree
        """
        # Defining the number of spaces 
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const
        
        if self.node_type == 'root':
            print("Root")
        elif self.node_type == 'left_node':
            if self.best_feature is not None:
                print(str(spaces) + " Split rule: " + str(self.best_feature) + " <= " + str(self.best_feature_split_val))
        else:
            if self.best_feature is not None:
                print(str(spaces) + "Split rule: " + str(self.best_feature) + " > " + str(self.best_feature_split_val))
        print(' ' * const +  f"| MSWE of the node: {round(self.mswe, 5)}")
        print(' ' * const + f"  | Count of observations in node: {self.n_units}")
#         print(f"{' ' * const}   | Prediction of node: {round(self.ymean, 3)}")   

    def print_tree(self):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info() 
        
        if self.left_node is not None: 
            self.left_node.print_tree()
        
        if self.right_node is not None:
            self.right_node.print_tree()
              
    def predict(self, X_valid):
        node = self
        y_pred = []
        for col in self.X.columns:
            if col not in X_valid.columns:
              raise Exception(col + ' is not a valid column')
        for i in X_valid.index.values:
            while (node.left_node is not None) and (node.right_node is not None):
                if X_valid.loc[i, node.best_feature] <= node.best_feature_split_val:
                    node = node.left_node
                else:
                    node = node.right_node
            y_pred_i = wasserstein2_barycenter(sample_array_1_through_n = node.y,
                    weights = np.repeat(1/node.y.shape[0], node.y.shape[0]),
                    n_samples_min = node.n_samples_min, 
                    qtl_id = node.y_quantile_id)
            y_pred.append(y_pred_i)
        return y_pred

class wass_forest:
    def __init__(self, X, y, y_quantile_id = False, min_samples_split=None, max_depth=None, depth=None, node_type=None, n_trees = 20, seed = 999, n_samples_min = None):
        self.X = X
        self.y = y
        self.y_quantile_id = y_quantile_id
        
        # find min number of samples in any y_i
        if n_samples_min is None:
            self.n_samples_min = np.apply_along_axis(
                arr = self.y,
                axis = 1,
                func1d = lambda x: x[x == x].shape[0]).min()
        else:
            self.n_samples_min = n_samples_min
        
        # convert Y to quantile function
        if self.y_quantile_id == False:
            quantiles = np.linspace(0, 1, self.n_samples_min)
            self.y = np.apply_along_axis(
                arr = self.y,
                axis = 1,
                func1d = lambda x: np.quantile(a = y, q = quantiles))
        
        self.y_quantile_id = True
        # save hyperparameters: defaults are 20 and 5 unless specified otherwise
        if min_samples_split is None:
            self.min_samples_split = 20
        else:
            self.min_samples_split = min_samples_split
        if max_depth is None:
            self.max_depth = 5
        else:
            self.max_depth = max_depth
#         self.max_depth = self.max_depth if max_depth is not None else 5
        
        # current node characteristics: defaults are 0 depth and 'root' node
        if depth is None:
            self.depth = 0
        else:
            self.depth = depth
        if node_type is None:
            self.node_type = 'root'
        else:
            self.node_type = node_type
            
        self.trees = []
        
        # for each new tree
        for i in range(n_trees):
            # bootstrap data: choose index
            bootstrap_ids = np.random.choice(a = range(self.X.shape[0]), size = self.X.shape[0], replace = True)
            bootstrap_X = []
            bootstrap_y = []
            for index in bootstrap_ids:
                bootstrap_X.append(pd.DataFrame(self.X.iloc[index, :]).transpose())
                bootstrap_y.append(self.y[index, :])
            bootstrap_X_df = pd.concat(bootstrap_X, axis = 0)
            bootstrap_y_np = np.array(bootstrap_y)
            
            # grow tree
            wass_tree = wass_node(
                X = bootstrap_X_df,
                y = bootstrap_y_np,
                y_quantile_id = self.y_quantile_id,
                min_samples_split = self.min_samples_split,
                max_depth = self.max_depth, 
                n_samples_min = self.n_samples_min
            )
            
            wass_tree.grow_tree()
            
            # save tree
            self.trees.append(wass_tree)
        
    def predict(self, X_valid):
        # initialize empty list to store predictions from individual trees
        y_pred = []
        # get each prediction for each tree
        for wass_tree in self.trees:
            y_pred.append(wass_tree.predict(X_valid))
        # compute the barycenter
        y_pred_np = np.array(y_pred) # should be a T by p matrix, where T is #trees
        y_bary = wasserstein2_barycenter(
            sample_array_1_through_n = y_pred_np,
            weights = np.repeat(1/len(self.trees), len(self.trees)),
            n_samples_min = self.n_samples_min,
            qtl_id = True)
        return y_bary
