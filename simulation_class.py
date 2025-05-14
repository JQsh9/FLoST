from RCGD.sparse_low_rank.simulation_functions import *
import time
from bayes_opt import BayesianOptimization
import random

class simulate_data():
    def __init__(self, T0,my_seed, d1, d2, d3,missing_rate=0.5, k=10, s=0.2, r=5):
        self.k =k
        self.r, self.missing_rate, self.d3= r, missing_rate, d3
        self.size = d1 * d2 * d3
        self.T0 = T0
        self.seed = my_seed
        np.random.seed(my_seed)
        self.Tn = T0 + np.random.normal(0,0.1, size=(d1, d2, d3))
        self.Tm = randomly_drop_pixels(self.Tn, missing_rate, seed=my_seed)

        self.mask_test = np.isnan(self.Tm)
    def RCGD(self,):
        time_start = time.time()
        self.T_rcgd = tensor_rcgd(self.Tm, self.r, monitor_seed=self.seed) 
        time_end = time.time()
        self.time_rcgd = time_end - time_start
        self.te_rmse_rcgd = np.linalg.norm(self.T0[self.mask_test] - self.T_rcgd[self.mask_test]) / math.sqrt(self.size)
        self.tr_rmse_rcgd = np.linalg.norm(self.T0[~self.mask_test] - self.T_rcgd[~self.mask_test]) / math.sqrt(self.size)
        

    def tune_lam(self,upper=1000, lower=0, random_state=0, init_points=5, n_iter=50, verbose=0):
        self.T_train, self.val_idx, train_missing= holdout_validation(self.Tm, val_ratio=0.10, seed=self.seed)
        sv1, sv2 = get_singular_values(self.T_train, train_missing, self.r, self.k)
        def black_box_function(x,y):
            T_hat = sparse_low_rank(self.T_train, train_missing, self.k, lam1=sv2, lam2=[x,y])
            return -np.linalg.norm(T_hat[self.val_idx]-self.Tm[self.val_idx])/ math.sqrt(self.size)
        pbounds = {'x': (lower, upper), 'y': (lower, upper)}
        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,random_state=random_state,verbose=verbose,)
        optimizer.maximize(init_points=init_points,n_iter=n_iter)
        self.lam2_opt = [optimizer.max['params']['x'], optimizer.max['params']['y']]

    def FLoST(self,):
        self.tune_lam()
        time_start = time.time()
        self.sv1, self.sv2 = get_singular_values(self.Tm, self.missing_rate, self.r, self.k)
        self.T_flost = sparse_low_rank(self.Tm, self.missing_rate, self.k, lam1=np.dot(self.sv2,1), lam2=self.lam2_opt)
        time_end = time.time()
        self.time_flost = time_end - time_start
        self.te_rmse_flost = np.linalg.norm(self.T0[self.mask_test] - self.T_flost[self.mask_test]) / math.sqrt(self.size)
        self.tr_rmse_flost = np.linalg.norm(self.T0[~self.mask_test] - self.T_flost[~self.mask_test]) / math.sqrt(self.size)

    def FLT(self,):
        self.half_d3 = math.ceil( (self.d3 + 1)/2 )
        self.sv1, self.sv2 = get_singular_values(self.Tm, self.missing_rate, self.r, self.half_d3)
        time_start = time.time()
        self.T_flt = sparse_low_rank(self.Tm, self.missing_rate, self.half_d3, lam1=np.dot(self.sv2,1), lam2=self.lam2_opt)
        time_end = time.time()
        self.time_flt = time_end - time_start
        self.te_rmse_flt = np.linalg.norm(self.T0[self.mask_test] - self.T_flt[self.mask_test]) / math.sqrt(self.size)
        self.tr_rmse_flt = np.linalg.norm(self.T0[~self.mask_test] - self.T_flt[~self.mask_test]) / math.sqrt(self.size)
        

        




        


