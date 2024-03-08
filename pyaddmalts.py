import scipy.optimize as opt
from util import *

# class for pymaltspro 
class pyaddmalts:
	def __init__(self, X, y, y_qtl_id, treatment, X_quantile_fns = np.array([[[]]]), discrete = [], C = 1, k = 10, reweight = False, id_name = 'index', norm = 2):
		'''
		description
		-----------
		a class for running malts on distribution functions

		inputs
		------
		X : N x (p + 2) pandas dataframe with all of the input features and treatment variable
			N is number of units and p is number of input covariates
				1 extra column for treatment variable
				1 extra column for unit id s.t. row 1 is also unit 1
			make sure indices are in order 1, 2, ..., N so it aligns with y
		y : N x Smax numpy array with all of the samples from the distributional outcome
			N is number of units and Smax is the max number of samples from any distribution
			if some unit has S < Smax samples, then its entry of y should have (Smax - S) NA values
		treatment : string with name of treatment column
		# id_name : string with  name of id column
		discrete : list with names of discrete columns
		C : coefficient of regularization term
		k : number of neighbors to match with in caliper matching
		reweight : whether to reweight outcomes when combining
		norm : the type of norm to use for regularization; default is 2 norm
		'''
		# initialize data
		self.C = C
		self.norm = norm
		self.k = k	
		self.reweight = reweight
		self.n = X.shape[0]
		self.treatment = treatment
		self.id_name = id_name
		self.discrete = discrete
		self.continuous = list(set(X.columns).difference(set([treatment]+discrete+[self.id_name])))
		if X_quantile_fns.shape[2] > 0:
			self.distributional = list(range(X_quantile_fns.shape[1]))
		else:
			self.distributional = []
		self.p = len(self.discrete) + len(self.distributional) + len(self.continuous) # number of non-treatment input features
		# split data into control and treated units
		self.X_T = X.loc[X[treatment] == 1]
		self.X_C = X.loc[X[treatment] == 0]
		# split X dfs into discrete and continuous covariates
		self.Xc_T = self.X_T[self.continuous].to_numpy()
		self.Xc_C = self.X_C[self.continuous].to_numpy()
		self.Xd_T = self.X_T[self.discrete].to_numpy()
		self.Xd_C = self.X_C[self.discrete].to_numpy()

		# split covariates that are distributions into control/treated units
		self.Xq_T = X_quantile_fns[np.where(X[treatment] == 1)[0], :, :]
		self.Xq_C = X_quantile_fns[np.where(X[treatment] == 0)[0], :, :]

		self.y = y
		# N x Smax vectors Y
		# find the minimum number of samples taken from outcome dist across all units
		self.n_samples_min = np.apply_along_axis(
				arr = self.y,
				axis = 1,
				func1d = lambda x: x[x == x].shape[0]).min()
		self.quantile_values = np.linspace(start = 0, stop = 1, num = self.n_samples_min)
		self.quantile_diffs = self.quantile_values[1:] - self.quantile_values[:-1]

		self.Y_T = self.y[np.where(X[treatment] == 1)[0], :]
		self.Y_C = self.y[np.where(X[treatment] == 0)[0], :]
		if y_qtl_id:
			self.Y_T_quantiles = self.Y_T
			self.Y_C_quantiles = self.Y_C
		else:
			self.Y_T_quantiles = np.apply_along_axis(
					arr = self.Y_T,
					axis = 1,
					func1d = lambda x: np.quantile(a = x[x == x], # remove NaN
										q = self.quantile_values)
					)
			self.Y_C_quantiles = np.apply_along_axis(
					arr = self.Y_C,
					axis = 1,
					func1d = lambda x: np.quantile(a = x[x == x],  
										q = self.quantile_values)
					)
		# store wasserstein distances in N_T x N_T (or N_C x N_C) matrix to avoid recomputing
		# self.Wass1_Y_T = pairwise_wasserstein(self.Y_T, self.Y_T, p = 1, n_samples_min = self.n_samples_min)
		# self.Wass1_Y_C = pairwise_wasserstein(self.Y_C, self.Y_C, p = 1, n_samples_min = self.n_samples_min)

		# pulled straight from pymalts2 code
		# Dc_T represents distance between continuous covariates for treatment units
		# Dc_C represents distance between continuous covariates for control units

		self.Dc_T = np.ones((self.Xc_T.shape[0],self.Xc_T.shape[1],self.Xc_T.shape[0])) * self.Xc_T.T
		# (X_{ik} - X^T_{jk})^2 --> compare kth cov between unit i and unit j
		self.Dc_T = (self.Dc_T - self.Dc_T.T) # each entry is a 3d matrix (i.e., cube) with entry (i,j,k) reps unit i, unit j, k-th covariate ==> check ordering of (i,j,k) 
		self.Dc_C = np.ones((self.Xc_C.shape[0],self.Xc_C.shape[1],self.Xc_C.shape[0])) * self.Xc_C.T # same thing for all controls
		self.Dc_C = (self.Dc_C - self.Dc_C.T) 

		# Dd_T represents distance between discrete covariates for treatment units
		# Dd_C represents distance between discrete covariates for control units
		self.Dd_T = np.ones((self.Xd_T.shape[0],self.Xd_T.shape[1],self.Xd_T.shape[0])) * self.Xd_T.T
		self.Dd_T = (self.Dd_T != self.Dd_T.T) 
		self.Dd_C = np.ones((self.Xd_C.shape[0],self.Xd_C.shape[1],self.Xd_C.shape[0])) * self.Xd_C.T
		self.Dd_C = (self.Dd_C != self.Dd_C.T) 

		# Dq_T represents distance between quantile functions for treatment units
		# Dq_C represents distance between quantile functions for control units
		if len(self.distributional) > 0:
			self.Dq_T = np.ones((self.Xq_T.shape[0], self.Xq_T.shape[1], self.Xq_T.shape[0]))
			self.Dq_C = np.ones((self.Xq_C.shape[0], self.Xq_C.shape[1], self.Xq_C.shape[0]))
			for cov in range(len(self.distributional)):
				self.Dq_T[:, cov, :]= wasserstein_distance_matrix(qtl_fn_matrix=self.Xq_T[:, [cov], :], weights = np.repeat(1, self.Xq_T.shape[1]))
				self.Dq_C[:, cov, :] = wasserstein_distance_matrix(qtl_fn_matrix=self.Xq_C[:, [cov], :], weights = np.repeat(1, self.Xq_T.shape[1]))
		else:
			self.Dq_T = np.array([]).reshape((self.Xc_T.shape[0], 0, self.Xc_T.shape[0]))
			self.Dq_C = np.array([]).reshape((self.Xc_C.shape[0], 0, self.Xc_C.shape[0]))

    # choose what kind of nearest neighbor we want; as of rn, it's just traditional knn
	def threshold(self,x):
		'''
		description
		-----------
		chooses the k nearest neighbors between a given unit x and the rest in dataset

		input
		-----
		x : N_x x p array of covariates for the N_x units of interest

		returns
		-------
		N_x x k array with indexes of the k-nn for each unit
		'''
		# traditional knn; if we want to use exp(...), have to update this code: can take gradient of that
		k = self.k
		for i in range(x.shape[0]):
		    row = x[i,:]
		    row1 = np.where( row < row[np.argpartition(row,k+1)[k+1]],1,0)
		    x[i,:] = row1
		return x
    
    # calculates distance between two units _given_ a specified distance metric -- not being used right now
	def distance(self,Mc,Md, Mq, xc1,xd1,xc2,xd2, xq1, xq2):
		'''
		description
		-----------
		calculate the distance between two unit's covariates given a specified distance metric
		not being used currently
		'''
		dc = np.dot((Mc**2)*(xc1-xc2),(xc1-xc2))
		dd = np.sum((Md**2)*xd1!=xd2)
		dq = wasserstein_distance_unit_covs(x1 = xq1, 
				      						x2 = xq2,
											weights = Mq, 
											n_samples_min=xq1.shape[1])

		return dc+dd+dq

	def calcW_T(self,Mc,Md,Mq):
		'''
		description
		-----------
		weight matrix for each treated unit's outcome given the stretch (Mc, Md)
		gives the objective function to reweight the importance of units to a single imputation

		inputs
		------
		Mc : matrix of how to stretch each unit's continuous covariates
		Md : matrix of how to stretch each unit's discrete covariates
		Mq : matrix of how to stretch each unit's distributional covariates

		returns
		-------
		the weight matrix
		'''
	    #this step is slow
		Dc = np.sum( ( self.Dc_T * (Mc.reshape(-1,1)) )**2, axis=1)
		Dd = np.sum( ( self.Dd_T * (Md.reshape(-1,1)) )**2, axis=1)
		Dq = np.sum( ( self.Dq_T * (Mq.reshape(-1,1)) )**2, axis = 1 )
		W = self.threshold( (Dc + Dd + Dq) )
		W = W / (np.sum(W,axis=1)-np.diag(W)).reshape(-1,1)
		return W  

	def calcW_C(self,Mc,Md,Mq):
		'''
		description
		-----------
		return weight matrix for each control unit's outcome given the stretch (Mc, Md)
		gives the objective function to reweight the importance of units to a single imputation

		inputs
		------
		Mc : matrix of how to stretch each unit's continuous covariates
		Mc : matrix of how to stretch each unit's discrete covariates

		returns
		-------
		the weight matrix
		'''
	    #this step is slow
		Dc = np.sum( ( self.Dc_C * (Mc.reshape(-1,1)) )**2, axis=1)
		Dd = np.sum( ( self.Dd_C * (Md.reshape(-1,1)) )**2, axis=1)
		Dq = np.sum( ( self.Dq_T * (Mq.reshape(-1,1)) )**2, axis = 1 )
		W = self.threshold( (Dc + Dd) )
		W = W / (np.sum(W,axis=1)-np.diag(W)).reshape(-1,1)
		return W



	# combination of both W_C and W_T
	def Delta_(self,Mc,Md,Mq):
		'''
		description
		-----------
		calculate Delta for treated and control outcomes using `calcW_T` and `calcW_C`
			Delta_t = 1/N_t sum_{units i w/ treatment t} W1(f_Yi, wass2bary_i)
			wass2bary_i = argmin_{v_i} sum_{l : t_i = t_l} lambda_i W2(f_Yl, v_i)
			lambda_i = exp(-d_M(x_i, x_l))/[sum_{k : t_i = t_k} -d_M(x_i, x_k)]

		inputs
		------
		Mc : parameters of distance function such that dist btw cont covs a_c, b_c is
			d(a_c, b_c) = || Mc * a_c - Mc * b_c ||_2
		Md : parameters of distance function such that dist btw disc covs a_d, b_d is
			d(a_d, b_d) = sum_{discrete covs j}Md 1(a_{d,j} == b_{d,j})

		returns weighted or unweighted 
		'''
		self.W_T = self.calcW_T(Mc,Md,Mq)
		self.W_C = self.calcW_C(Mc,Md,Mq)
		self.delta_T = np.ones(shape = self.Y_T.shape[0])
		self.delta_C = np.ones(shape = self.Y_C.shape[0])

		for i in range(self.delta_T.shape[0]):
			Y_T_bary_i = wasserstein2_barycenter(
					sample_array_1_through_n = self.Y_T_quantiles,
					weights = self.W_T[i, :],
					n_samples_min = self.n_samples_min,
					qtl_id = True
					)
			#W1(f_yi, wass2bary(distributions f_y1...f_yN_T, weights W_T[i]))
			self.delta_T[i] = wasserstein_dist(
				sample_array1 = self.Y_T_quantiles[i, :],
				sample_array2 = Y_T_bary_i,
				p = 1, 
				n_samples_min = self.n_samples_min,
				array1_quantile=True,
				array2_quantile = True
				)

		for i in range(self.delta_C.shape[0]):
			#W1(f_yi, wass2bary(distributions f_y1...f_yN_C, weights W_C[i]))
			Y_C_bary_i = wasserstein2_barycenter(
					sample_array_1_through_n = self.Y_C_quantiles,
					weights = self.W_C[i, :],
					n_samples_min = self.n_samples_min,
					qtl_id = True
					)
			self.delta_C[i] = wasserstein_dist(
				sample_array1 = self.Y_C_quantiles[i, :],
				sample_array2 = Y_C_bary_i,
				p = 1, 
				n_samples_min = self.n_samples_min,
				array1_quantile = True,
				array2_quantile = True)

		self.delta_T = self.delta_T.mean()
		self.delta_C = self.delta_C.mean()
		
		# self.delta_T = np.sum((self.Y_T - (np.matmul(self.W_T,self.Y_T) - np.diag(self.W_T)*self.Y_T))**2)
		# self.delta_C = np.sum((self.Y_C - (np.matmul(self.W_C,self.Y_C) - np.diag(self.W_C)*self.Y_C))**2)
		if self.reweight == False:
		    return self.delta_T + self.delta_C
		elif self.reweight == True:
		    return (self.Y_T.shape[0] + self.Y_C.shape[0])*(self.delta_T/self.Y_T.shape[0] + self.delta_C/self.Y_C.shape[0])

	def objective(self, M):
		'''
		description
		-----------
		calculate objective function: min_M Delta_T + Delta_C + FrobeniusNorm(M)

		inputs
		------
		M : 1 x p vector specifying the weight to place on each cov
			discrete covariates come before continuous covariates

		returns
		-------
		calculated objective function
		'''
		Mc = np.abs(M[ :len(self.continuous)])
		Md = np.abs(M[len(self.continuous):(len(self.continuous) + len(self.discrete))])
		Mq = np.abs(M[(len(self.continuous) + len(self.discrete)): ])

		delta = self.Delta_(Mc, Md, Mq)
		reg = self.C * ( np.linalg.norm(Mc,ord=self.norm)**2 + np.linalg.norm(Md,ord=self.norm)**2 + np.linalg.norm(Mq,ord=self.norm))
		# ask harsh why we need cons1 and cons2
		cons1 = 0 * ( (np.sum(Mc) + np.sum(Md) + np.sum(Mq)) - self.p )**2
		cons2 = 1e+25 * np.sum( ( np.concatenate((Mc,Md,Mq)) < 0 ) )
		return delta + reg + cons1 + cons2

	# fits -- like sklearn fit
	def fit(self,method='COBYLA', M_init = None):
		'''
		description
		-----------
		find argument (i.e., param values) that minimize objective fn

		inputs
		------
		method: string specifying optimization method to use
		
		returns
		-------
		returns best fitting param values
		'''
		if M_init is None:
			M_init = np.ones((self.p,))
		res = opt.minimize( self.objective, x0=M_init,method=method )
		self.M = np.abs(res.x)
		self.Mc = np.abs(self.M[:len(self.continuous)])
		self.Md = np.abs(self.M[len(self.continuous):(len(self.continuous) + len(self.discrete))])
		self.Mq = np.abs(self.M[(len(self.continuous) + len(self.discrete)):])
		self.M_opt = pd.DataFrame(self.M.reshape(1,-1),columns=self.continuous+self.discrete+self.distributional,index=['Diag'])
		return res
	
	
	def get_matched_groups(self, X_estimation, Y_estimation, k, X_qtl_estimation = np.array([[[]]])):
		# split df into continuous and discrete variables
		Xc = X_estimation[self.continuous].to_numpy()
		Xd = X_estimation[self.discrete].to_numpy()
		Xq = X_qtl_estimation
		treatment = X_estimation[self.treatment].to_numpy()

		# get distance between all units
		Dc = np.zeros( (X_estimation.shape[0], X_estimation.shape[0]) )
		Dd = np.zeros( (X_estimation.shape[0], X_estimation.shape[0]) )
		Dq = np.zeros( (X_estimation.shape[0], X_estimation.shape[0]) )
		for i in range(X_estimation.shape[0]):
			for j in range(X_estimation.shape[0]):
				Dc[i,j] = np.sqrt(((Xc[i, :] - Xc[j, :]) * (Xc[i, :] - Xc[j, :]) * self.Mc).sum())
				Dd[i,j] = ((Xd[i, :] != Xd[j, :]) * self.Md).sum()
		if len(self.distributional) > 0:
			Dq = wasserstein_distance_matrix(qtl_fn_matrix=Xq, weights = self.Mq)

		D = Dc + Dd + Dq
		control_idxs = np.where(treatment == 0)[0]
		treated_idxs = np.where(treatment == 1)[0]
		D_control = D[:, control_idxs]
		D_treated = D[:, treated_idxs]
		# for each query unit
		index = X_estimation.shape[0]
		mg_list = []
		for unit in range(X_estimation.shape[0]):
			# find k closest control units
			control_mg_index = np.argpartition(D_control[unit, :], k)[:k]
			# control_mg_index = control_idxs[control_mg_index]
			control_mg = X_estimation.loc[control_idxs[control_mg_index], :]
			control_mg['distance'] = D_control[unit, control_mg_index]
			control_mg['matched_unit'] = control_idxs[control_mg_index]
			# find k closest treated units
			treated_mg_index = np.argpartition(D_treated[unit, :], k)[:k]
			treated_mg = X_estimation.loc[treated_idxs[treated_mg_index], :]
			treated_mg['distance'] = D_treated[unit, treated_mg_index]
			treated_mg['matched_unit'] = treated_idxs[treated_mg_index]
			# make mg df : rename variables to match with barycenter imputation
			mg = pd.concat([control_mg, treated_mg], axis = 0)
			mg['unit_treatment'] = treatment[unit]
			mg = mg.sort_values('distance', ascending=True)
			mg.index = np.repeat(unit, mg.shape[0])
			mg['unit'] = np.repeat(unit, mg.shape[0])
			mg_list.append(mg)
		MG = pd.concat(mg_list, axis = 0)
		return MG


	def barycenter_imputation(self, X_estimation, Y_estimation, MG, qtl_id):
	    Y_counterfactual = []
	    for i in X_estimation.index.values:
	        # make a holder list for adding matched units' outcomes
	        matched_unit_ids = MG.query('unit == ' + str(i)).query(self.treatment + ' != unit_treatment').matched_unit.values
	        matched_unit_outcomes = Y_estimation[matched_unit_ids, :]
	        y_i_counterfactual = wasserstein2_barycenter(
	            sample_array_1_through_n = matched_unit_outcomes, 
	            weights = np.repeat(1/matched_unit_outcomes.shape[0], matched_unit_outcomes.shape[0]),
	            n_samples_min=self.n_samples_min,
		    qtl_id=qtl_id
	        )
	        Y_counterfactual.append(y_i_counterfactual)
	    return np.array(Y_counterfactual)


	def estimate_CATE(self, X_est, y_est, k = 1, y_est_qtl_id = False):
		# get matched groups
		mg_df = self.get_matched_groups(X_estimation=X_est,
                                        Y_estimation= y_est,
                                        k = k)
		# impute barycenters
		y_bary = self.barycenter_imputation(X_estimation = X_est,
				      						Y_estimation = y_est,
                                            MG = mg_df, 
                                            qtl_id = y_est_qtl_id)
		
        # take difference in quantile fn's
		CATE_list = []
		for i in range(X_est.shape[0]):
			CATE_list.append(
				CATE(
					y_obs = y_est[i, :],
					y_cf = y_bary[i, :],
					observed_treatment = X_est[self.treatment].values[i],
					reference_distribution = np.vstack([np.linspace(0, 1, self.n_samples_min),
														np.linspace(0, 1, self.n_samples_min)]),
					y_obs_qtl_id = self.y_qtl_id, 
					y_cf_qtl_id = True
				)
			)
		CATE_list = np.array(CATE_list)
		return CATE_list
	
	def mise(self, y_pred, y_true):
		'''
		description
		-----------
		Given function families u_i, v_i approximate 1/n sum_{i = 1}^n (int_{t} |u_i(t) - v_i(t)|^2 dt)

		parameters
		----------
		y_pred : array from predicted vector
		y_true : array from true vector

		returns
		-------
		float representing mean integrated squared error between two vectors
		'''
		return ((y_pred - y_true)**2).sum(axis = 1).mean()
