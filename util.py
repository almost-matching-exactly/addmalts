import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def wasserstein_dist(sample_array1, sample_array2, p, n_samples_min, array1_quantile = False, array2_quantile = False):
    '''
    description
    -----------
    calculate pairwise wasserstein distance between all the sampled distributions of one array and all the sampled distributions of another
    Wasserstein distance for dists U, V on real line: 
        int_{0}^1 (F^{-1}_U(q) - F^{-1}_V(q)) dq, where F^{-1}_A reps quantile function of rv A

    inputs
    ------
    sample_array1: N x S_max1 np array of floats/ints
        N is number of units in the array and S_max1 is the maximum number of samples across any of the N dists
    sample_array2: N x S_max2 np array of floats/ints
        N is number of units in the array and S_max2 is the maximum number of samples across any of the N dists
    p: order of wasserstein distance

    returns
    -------
    Wasserstein-p distance between sample_array1 and sample_array2
    '''
    # if the point is scalar, evaluate only at the 1st quantile
    if n_samples_min == 1:
        quantile_values = np.array([1])
        quantile_diffs = [1]
    else:
        quantile_values = np.linspace(start = 0, stop = 1, num = n_samples_min)
        # find width of each quantile window
        quantile_diffs = np.absolute(quantile_values[1:] - quantile_values[:-1])
        quantile_diffs = np.hstack([[0], quantile_diffs])
    if array1_quantile:
    	quantile_array1 = sample_array1
    else:
        quantile_array1 = np.quantile(sample_array1[~np.isnan(sample_array1)], q = quantile_values)
    if array2_quantile:
        quantile_array2 = sample_array2
    else:
    	quantile_array2 = np.quantile(sample_array2[~np.isnan(sample_array2)], q = quantile_values)
     
    # calculate distance between quantile
    dist = ((np.absolute(quantile_array1 - quantile_array2)**p * quantile_diffs).sum())**(1/p)
    return dist

def pairwise_wasserstein(sample_array1, sample_array2, p, n_samples_min):
    '''
    description
    -----------
    calculate pairwise wasserstein distance between all the sampled distributions of one array and all the sampled distributions of another
    Wasserstein distance for dists U, V on real line: 
        int_{0}^1 (F^{-1}_U(q) - F^{-1}_V(q)) dq, where F^{-1}_A reps quantile function of rv A

    inputs
    ------
    sample_array1: N x S_max1 np array of floats/ints
        N is number of units in the array and S_max1 is the maximum number of samples across any of the N dists
    sample_array2: N x S_max2 np array of floats/ints
        N is number of units in the array and S_max2 is the maximum number of samples across any of the N dists
    p: order of wasserstein distance

    returns
    -------
    a N x N matrix where entry (i,j) represents Wasserstein-p distance between unit i and unit j
    '''

    # initialize return matrix s.t. entry (i,j) represents wasserstein-p distance 
        # between dist i of sample_array1 and dist j of sample_array2
    emd_matrix = np.ones((sample_array1.shape[0], sample_array2.shape[0]))
    for i in range(emd_matrix.shape[0]):
        for j in range(emd_matrix.shape[1]):
            # gather distributions i and j
            sample_array_i = sample_array1[i, :]
            sample_array_j = sample_array2[j, :]
            # calculate wasserstein-p distance
            emd_matrix[i, j] = wasserstein_dist(sample_array1 = sample_array_i,
                                                sample_array2 = sample_array_j, 
                                                p = p,
                                                n_samples_min = n_samples_min)

    return emd_matrix

def wasserstein_distance_unit_covs(x1, x2, weights, n_samples_min):
	distance = 0
	for j in range(x1.shape[0]):
		distance += weights[j] * wasserstein_dist(sample_array1=x1[j, :],
												sample_array2=x2[j, :],
												p = 1, 
												n_samples_min=n_samples_min, 
												array1_quantile=True, 
												array2_quantile=True)
	return distance

def wasserstein_distance_matrix(qtl_fn_matrix, weights):
	'''
	description
	-----------
	Calculate distance between all units' distributional covariates
	d(x_1, x_2) = \sum_{covs j} w_j W_1(x_{1,j}, x_{2,j}) 
	            = \sum_{covs j} w_j \int_0^1 |x_{1,j}(q) - x_{2,j}(q)| dq
		    	= \sum_{covs j} w_j \sum_{q = 0...1} |x_{1,j,q} - x_{2,j,q}|\Delta_q
	
	inputs
	------
	qtl_fn_matrix : N by P by Q matrix where
		N is number of units
		P is number of distributional covariates
		Q is number of discrete quantiles each covariate is evaluated at *with equal size bins*
		each entry is unit i's covariate j's empirical quantile q
	weights : P by 1 array where
		P is number of distributional covariates
	
	returns
	-------
	N by N matrix representing distributional distance between all units
	'''
	dist_matrix = np.zeros((qtl_fn_matrix.shape[0], qtl_fn_matrix.shape[0]))
	n_samples_min = qtl_fn_matrix.shape[2]
	for i1 in range(qtl_fn_matrix.shape[0]):
		for i2 in range(qtl_fn_matrix.shape[0]):
			dist_matrix[i1, i2] = wasserstein_distance_unit_covs(x1 = qtl_fn_matrix[i1, :, :],
																	x2 = qtl_fn_matrix[i2, :, :],
																	weights = weights,
																	n_samples_min= n_samples_min
																	)
	return dist_matrix



def wasserstein2_barycenter(sample_array_1_through_n, weights, n_samples_min, qtl_id):
	'''
	description
	-----------
	compute the wasserstein-2 barycenter

	inputs
	------
	sample_array_1_through_n : N x Smax numpy array with all of the samples from the distributional outcome
		N is number of units and Smax is the max number of samples from any distribution
		if some unit has S < Smax samples, then its entry of y should have (Smax - S) NA values
					OR
		N x Q matrix where Q is number of quantiles the quantile function is evaluated at
	weight : N x 1 array specifying the weight to place on each unit's distribution when taking average
	n_samples_min : minimum number of samples from any of the N distributions
	qtl_id : boolean set to True if sample_array_1_through_n is quantile function

	returns
	-------
	n_samples_min x 1 array such that entry i is (i/n_samples_min)-th quantile from barycenter
	'''
	# compute empirical quantile functions for each distribution 
	if qtl_id == False:
		qtls_1_through_n = np.apply_along_axis(
					arr = sample_array_1_through_n,
					axis = 1,
					func1d = lambda x: np.quantile(
						a = x, 
						q = np.linspace(start = 0, stop = 1, num = n_samples_min)
						)
					)
	else:
		qtls_1_through_n = sample_array_1_through_n

	# take quantile level euclidean average weighted by weights
	bcenter_qtl = np.average(a = qtls_1_through_n,
								weights = weights, axis = 0)

	# return barycenter quantile function
	return bcenter_qtl



def sample_quantile(quantile_fn, quantile):
    '''
    description
    -----------
    linearly interpolate quantile function and return value of a given quantile
    
    parameters
    ----------
    quantile_fn : numpy array with values of quantile function at specified quantiles
    quantile : value of quantile
    n_qtls : size of quantile function
    
    returns
    -------
    quantile function evaluated at specified quantile
    '''
    n_qtls = quantile_fn.shape[0] - 1
    quantile_index = quantile * n_qtls
    quantile_floor = int(np.floor(quantile_index))
    quantile_ceil  = int(np.ceil(quantile_index))
    if quantile_floor == quantile_ceil == quantile_index:
        return(quantile_fn[quantile_floor])
    else:
        return np.sum([quantile_fn[quantile_floor] * (quantile_index - quantile_floor), quantile_fn[quantile_ceil] * (quantile_ceil - quantile_index)])



def lin_et_al_CATE(y_obs, y_cf, observed_treatment, reference_distribution, y_obs_qtl_id = False, y_cf_qtl_id = False):
		'''
		description
		-----------
		Compute Y_i^-1(1) \circ \lambda(t) - Y_i^{-1}(0) \circ \lambda(t)
		Please see Lin et al (2023) for more details on notation: https://arxiv.org/abs/2101.01599

		parameters
		----------
		y_obs : array of samples/quantiles for the true observed outcome in estimation set
		y_cf : array of quantile function for counterfactual outcome
		observed_treatment : boolean that is True iff treated outcome observed, False otherwise
		reference_distribution : a 2D array mapping samples from reference distribution to density of sample
			-- reference distribution _must_ be continuous
			-- col 1 is sample
			-- col 2 is prob of observing sample
		y_obs_qtl_id : boolean that is True iff y_estimation is a quantile function

		returns
		-------
		E[Y_i(1)^{-1}(\lambda(t)) - Y_i(0)^{-1}(\lambda(t))], 0 <= t <= 1
		'''
		# what values are quantile functions eval'd at?
		quantiles = np.linspace(start = 0, stop = 1, num = y_cf.shape[0])
		# if the given array for observed values is not a quantile function, turn it into a emp qtl fn
		if y_obs_qtl_id:
			y_estimation = y_obs
		else:
			y_estimation = np.quantile(y_obs, quantiles)
		# if the given array for counterfactual values is not a quantile function, turn it into a emp qtl fn
		if y_cf_qtl_id:
			y_cf = y_cf
		else:
			y_cf = np.quantile(y_cf, quantiles)
		# initialize objects to save data
		ylambda_treated = []
		ylambda_control = []
  
		# Find Y(1) \circ lambda and Y(0) \circ lambda
		if observed_treatment == 1:
			for i in reference_distribution[1, :]:
				ylambda_treated.append(sample_quantile(quantile_fn = y_estimation, quantile = i))
				ylambda_control.append(sample_quantile(quantile_fn = y_cf, quantile = i))
		else:
			for i in reference_distribution[1, :]:
				ylambda_treated.append(sample_quantile(quantile_fn = y_cf, quantile = i))
				ylambda_control.append(sample_quantile(quantile_fn = y_estimation, quantile = i))
		# calculate Y(1) \circ lambda - Y(0) \circ lambda
		ylambda_treated = np.array(ylambda_treated)
		ylambda_control = np.array(ylambda_control)
		ylambda_treated_minus_control = ylambda_treated - ylambda_control
		return_array = np.array([reference_distribution[0, :], ylambda_treated_minus_control])
		return return_array


def CATE(y_obs, y_cf, observed_treatment, y_obs_qtl_id = False, y_cf_qtl_id = False):
    '''
    description
    -----------
    Compute CATE: E[ F_{Y(1)}^{-1}(q) - F_{Y(0)}^{-1}(q) | X]

    parameters
    ----------
    y_obs : array of samples/quantiles for the true observed outcome in estimation set
    y_cf : array of quantile function for counterfactual outcome
    observed_treatment : boolean that is True iff treated outcome observed, False otherwise
    y_obs_qtl_id : boolean that is True iff y_estimation is a quantile function

    returns
    -------
    E[Y_i(1)^{-1}(q) - Y_i(0)^{-1}(q) | X], 0 <= q <= 1
    '''
    quantiles = np.linspace(start = 0, stop = 1, num = y_cf.shape[0])
    reference_distribution = np.array([quantiles, quantiles])
    lin_et_al_result = lin_et_al_CATE(y_obs, 
                                        y_cf, 
                                        observed_treatment, 
                                        reference_distribution, 
                                        y_obs_qtl_id = y_obs_qtl_id, 
                                        y_cf_qtl_id = y_cf_qtl_id)
    cate = lin_et_al_result[:, 1]
    return cate
