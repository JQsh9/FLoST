# dependencies python3
import numpy as np
from scipy.io import loadmat
import math
import random
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
#from scipy.special import boxcox, inv_boxcox
from scipy.linalg import svd
from scipy.ndimage import uniform_filter, maximum_filter, gaussian_filter

#from tqdm import tqdm

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
from bayes_opt import BayesianOptimization
# my functions 
from generate_data import *
from preprocess_data import *
from completion_alg import *
from smoothing import *
from check_results import *

class TEC_data():
    def __init__(self, dates, preprocess=False,r=5, k=100, missing_rate = 0.5, cutted = True,
                 n_quantiles = 10000,# parms for quantile transformer
                 ):
        self.tec_VISTA, self.tec_MF_lt = generate_tensor_vista(dates)
        if cutted:
            self.tec_VISTA = self.tec_VISTA[60:150, 90:330, :]
            self.tec_MF_lt = self.tec_MF_lt[60:150, 90:330, :]
            #self.tec_VISTA = self.tec_VISTA[60:130, 100:320, :]
            #self.tec_MF_lt = self.tec_MF_lt[60:130, 100:320, :]
        self.missing_rate = missing_rate
        self.preprocess = preprocess
        if preprocess == False:
            self.tensor_dropped = randomly_drop_pixels(self.tec_VISTA, self.missing_rate)
        elif preprocess == 'log':
            self.tensor_dropped = randomly_drop_pixels(np.log(self.tec_VISTA), self.missing_rate)
        elif preprocess == 'scale':
            self.tensor_dropped = randomly_drop_pixels(self.tec_VISTA, self.missing_rate)
            self.tensor_dropped,self.scaler = robust_scaler(self.tensor_dropped)
        elif preprocess == 'quantile':
            self.tensor_dropped = randomly_drop_pixels(self.tec_VISTA, self.missing_rate)
            flattern_sorted = np.sort(self.tensor_dropped.reshape(-1, 1))
            self.qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal', random_state=2025)
            flattern_sorted = self.qt.fit_transform(flattern_sorted)
            self.tensor_dropped = flattern_sorted.reshape(self.tensor_dropped.shape)
        self.tensor_train, self.val_idx= holdout_validation(self.tensor_dropped, val_ratio=0.10, seed=0)
        
        self.k, self.r = k, r
        self.sv1, self.sv2 = get_singular_values(self.tensor_dropped,self.r,self.k)

    
    def run_completion(self,lam1,lam2, train=False):
        if train:
            self.tensor_hat = sparse_low_rank(self.tensor_train, self.missing_rate, self.k, lam1, lam2)
        else:
            self.tensor_hat = sparse_low_rank(self.tensor_dropped, self.missing_rate, self.k, lam1, lam2)
        # inverse processing
        if self.preprocess == 'log':
            self.tensor_hat = np.exp(self.tensor_hat)
        elif self.preprocess == 'scale':
            self.tensor_hat = inverse_robust_scaler(self.tensor_hat, self.scaler)
        elif self.preprocess == 'quantile':
            shape = self.tensor_hat.shape
            self.tensor_hat = self.qt.inverse_transform(self.tensor_hat.reshape(-1, 1)).reshape(shape)
        else: pass


    def tune_lam2(self, 
                  lower, upper, lam1, metric = 'rmse',
                  init_points=10, n_iter=50,
                  verbose = 2,random_state=0):
        def black_box_function(x,y):
            self.run_completion(lam1=lam1, lam2=[x,y])
            if metric == 'rse':
                return -np.linalg.norm(self.tensor_hat - self.tec_VISTA)/np.linalg.norm(self.tec_VISTA)
            elif metric == 'rmse':
                return -np.linalg.norm(self.tensor_hat - self.tec_VISTA)/math.sqrt(self.tec_VISTA.size)
        pbounds = {'x': (lower, upper), 'y': (lower, upper)}
        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,random_state=random_state,verbose=verbose,)
        optimizer.maximize(init_points=init_points,n_iter=n_iter)
        self.lam2_opt = [optimizer.max['params']['x'], optimizer.max['params']['y']]
        self.error= optimizer.max['target']

    def tune_lam1(self,
                  lam2, metric = 'rmse',
                  init_points=5, n_iter=50,
                  verbose = 0,random_state=0,):
        tuned = []
        for i in range(2, self.k+1, 2):
            def black_box_function(x,y):
                lam1_tune = tuned + [x,y] + self.sv2[i:]
                self.run_completion(lam1=lam1_tune, lam2=lam2,)
                if metric == 'rse':
                    return -np.linalg.norm(self.tensor_hat - self.tec_VISTA)/np.linalg.norm(self.tec_VISTA)
                elif metric == 'rmse':
                    return -np.linalg.norm(self.tensor_hat - self.tec_VISTA)/math.sqrt(self.tec_VISTA.size)
            pbounds = {'x': (self.sv2[i-2]-1000, self.sv1[i-2]+1000), 
                       'y': (self.sv2[i-1]-1000, self.sv1[i-1]+1000)}
            optimizer = BayesianOptimization(
                f=black_box_function,
                pbounds=pbounds,random_state=random_state,verbose=verbose,)
            optimizer.maximize(init_points=init_points,n_iter=n_iter)
            tuned.append(optimizer.max['params']['x'])
            tuned.append(optimizer.max['params']['y'])
        self.lam1_opt = tuned
        self.error = optimizer.max['target']

    def tune_lam1_parallel(self, tune_start, n_tune,
                  lam2, metric = 'rmse',
                  init_points=5, n_iter=50,
                  verbose = 0,random_state=0,):
        tuned = self.sv2[:tune_start-2]
        for i in range(tune_start, tune_start+n_tune, 2):
            def black_box_function(x,y):
                lam1_tune = tuned + [x,y] + self.sv2[i:]
                self.run_completion(lam1=lam1_tune, lam2=lam2,)
                if metric == 'rse':
                    return -np.linalg.norm(self.tensor_hat - self.tec_VISTA)/np.linalg.norm(self.tec_VISTA)
                elif metric == 'rmse':
                    return -np.linalg.norm(self.tensor_hat - self.tec_VISTA)/math.sqrt(self.tec_VISTA.size)
            pbounds = {'x': (self.sv2[i-2]-1000, self.sv1[i-2]+1000), 
                       'y': (self.sv2[i-1]-1000, self.sv1[i-1]+1000)}
            optimizer = BayesianOptimization(
                f=black_box_function,
                pbounds=pbounds,random_state=random_state,verbose=verbose,)
            optimizer.maximize(init_points=init_points,n_iter=n_iter)
            tuned.append(optimizer.max['params']['x'])
            tuned.append(optimizer.max['params']['y'])
        self.lam1_opt = tuned[tune_start-2:]
        self.error = optimizer.max['target']

    def validation_tune_lam2(self, 
                  lower, upper, lam1, metric = 'rmse',
                  init_points=10, n_iter=50,
                  verbose = 2,random_state=0):  
        def black_box_function(x,y):
            self.run_completion(lam1=lam1, lam2=[x,y], train=True)
            if metric == 'rse':
                return -np.linalg.norm(self.tensor_hat[self.val_idx]-self.tensor_dropped[self.val_idx])/np.linalg.norm(self.tensor_dropped[self.val_idx])
            elif metric == 'rmse':
                return -np.linalg.norm(self.tensor_hat[self.val_idx] - self.tensor_dropped[self.val_idx])/math.sqrt(self.tensor_hat.size)
        pbounds = {'x': (lower, upper), 'y': (lower, upper)}
        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,random_state=random_state,verbose=verbose,)
        optimizer.maximize(init_points=init_points,n_iter=n_iter)
        self.lam2_opt = [optimizer.max['params']['x'], optimizer.max['params']['y']]
        self.error= optimizer.max['target']

    def validation_tune_lam1_parallel(self, tune_start, n_tune,
                  lam2, metric = 'rmse',
                  init_points=5, n_iter=50,
                  verbose = 0,random_state=0,):
        tuned = self.sv2[:tune_start-2]
        for i in range(tune_start, tune_start+n_tune, 2):
            def black_box_function(x,y):
                lam1_tune = tuned + [x,y] + self.sv2[i:]
                self.run_completion(lam1=lam1_tune, lam2=lam2,train=True)
                if metric == 'rse':
                    return -np.linalg.norm(self.tensor_hat[self.val_idx]-self.tensor_dropped[self.val_idx])/np.linalg.norm(self.tensor_dropped[self.val_idx])
                elif metric == 'rmse':
                    return -np.linalg.norm(self.tensor_hat[self.val_idx] - self.tensor_dropped[self.val_idx])/math.sqrt(self.tensor_hat.size)
            pbounds = {'x': (self.sv2[i-2]-1000, self.sv1[i-2]+1000), 
                       'y': (self.sv2[i-1]-1000, self.sv1[i-1]+1000)}
            optimizer = BayesianOptimization(
                f=black_box_function,
                pbounds=pbounds,random_state=random_state,verbose=verbose,)
            optimizer.maximize(init_points=init_points,n_iter=n_iter)
            tuned.append(optimizer.max['params']['x'])
            tuned.append(optimizer.max['params']['y'])
        self.lam1_opt = tuned[tune_start-2:]
        self.error = optimizer.max['target']

    def validation_tune_lam1(self,
                  lam2, metric = 'rmse',
                  init_points=5, n_iter=50,
                  verbose = 0,random_state=0,):
        tuned = []
        for i in range(2, self.k+1, 2):
            def black_box_function(x,y):
                lam1_tune = tuned + [x,y] + self.sv2[i:]
                self.run_completion(lam1=lam1_tune, lam2=lam2,train=True)
                if metric == 'rse':
                    return -np.linalg.norm(self.tensor_hat[self.val_idx]-self.tensor_dropped[self.val_idx])/np.linalg.norm(self.tensor_dropped[self.val_idx])
                elif metric == 'rmse':
                    return -np.linalg.norm(self.tensor_hat[self.val_idx] - self.tensor_dropped[self.val_idx])/math.sqrt(self.tensor_hat.size)
            pbounds = {'x': (self.sv2[i-2]-1000, self.sv1[i-2]+1000), 
                       'y': (self.sv2[i-1]-1000, self.sv1[i-1]+1000)}
            optimizer = BayesianOptimization(
                f=black_box_function,
                pbounds=pbounds,random_state=random_state,verbose=verbose,)
            optimizer.maximize(init_points=init_points,n_iter=n_iter)
            tuned.append(optimizer.max['params']['x'])
            tuned.append(optimizer.max['params']['y'])
        self.lam1_opt = tuned
        self.error = optimizer.max['target']

        '''
    def tune_k(self, lower_k, upper_k, lower_lam2, upper_lam2,
               #init_points=5, n_iter=50, 
               metric = 'rmse'):
        error = 10000
        k_opt = 0
        for k in tqdm(range(lower_k, upper_k+1,10)):
            self.k = k
            self.sv1, self.sv2 = get_singular_values(self.tensor_dropped,self.r,self.k)
            self.tune_lam2(lower_lam2, upper_lam2, self.sv2,
                  init_points=10, n_iter=50,metric = metric, verbose = 0,)
            self.tune_lam1( self.lam2_opt, init_points=5, n_iter=50,metric = metric,)
            if self.error < error:
                error = self.error
                k_opt = k
        return k_opt, error

        def black_box_function(x):
            self.k = x
            self.sv1, self.sv2 = get_singular_values(self.tensor_dropped,self.r,self.k)
            self.tune_lam2(lower_lam2, upper_lam2, self.sv2,
                  init_points=10, n_iter=50,metric = metric, verbose = 0,)
            self.tune_lam1( self.lam2_opt, init_points=5, n_iter=50,metric = metric,)
            return self.error
        pbounds= {'x':(lower_k,upper_k)}
        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,verbose=2,random_state=0,)
        optimizer.maximize(init_points=init_points,n_iter=n_iter)
        self.k = optimizer.max['params']['x']
        self.sv1, self.sv2 = get_singular_values(self.tensor_dropped,self.r,self.k)
        self.tune_lam2(lower_lam2, upper_lam2, self.sv2,
                  init_points=10, n_iter=50,metric = metric, verbose = 0,)
        self.tune_lam1( self.lam2_opt, init_points=5, n_iter=50,metric = metric,)
        '''
        # smoothing
    def smoothing(self, sigma, size):
        self.tensor_hat_mf = smooth_tensor_mean_filter(self.tensor_hat, size)
        self.tensor_hat_maxf = smooth_tensor_maximum_filter(self.tensor_hat, size)
        self.tensor_hat_gf = smooth_tensor_gaussian_filter(self.tensor_hat, sigma)