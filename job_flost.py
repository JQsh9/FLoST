# dependencies python3
import numpy as np
import math
import random
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
#from scipy.special import boxcox, inv_boxcox
from scipy.io import loadmat
from scipy.linalg import svd
from scipy.ndimage import uniform_filter, maximum_filter, gaussian_filter
# visualization
import matplotlib.pyplot as plt
import seaborn as sns
from bayes_opt import BayesianOptimization

# my functions 
from generate_data import generate_tensor_vista, generate_tensor_simulated
from preprocess_data import randomly_drop_pixels, holdout_validation,get_singular_values,robust_scaler, inverse_robust_scaler
from completion_alg import sparse_low_rank
from smoothing import smooth_tensor_mean_filter, smooth_tensor_gaussian_filter, smooth_tensor_maximum_filter
from check_results import error_plot, compare_maps

from TEC_class import TEC_data

import sys
#k = int(sys.argv[1])
tune_start = int(sys.argv[1])
d3 = 288*6
k=math.ceil( (d3 + 1)/2 )

dates = ['190621', '190622', '190623', '190624', '190625', '190626']

VISTA_1906 = TEC_data(dates, preprocess=False,r=5, k=k, missing_rate = 0.5, cutted = True)
VISTA_1906.validation_tune_lam2(lower=50, upper=3000, lam1=VISTA_1906.sv2, metric = 'rmse',
                  init_points=10, n_iter=50,verbose = 0,random_state=0)
VISTA_1906.validation_tune_lam1_parallel(tune_start, 50,
                                VISTA_1906.lam2_opt)
if tune_start == 2:
    lam_results = VISTA_1906.lam2_opt + VISTA_1906.lam1_opt
else: lam_results = VISTA_1906.lam1_opt

'''
VISTA_1906 = TEC_data(dates, preprocess='quantile',r=5, k=k, missing_rate = 0.5, cutted = True)
VISTA_1906.validation_tune_lam2(lower=50, upper=3000, lam1=VISTA_1906.sv2, metric = 'rmse',
                  init_points=10, n_iter=50,verbose = 0,random_state=0)
VISTA_1906.validation_tune_lam1_parallel(tune_start, 50,
                                VISTA_1906.lam2_opt)
                                if tune_start == 2:
lam_results_quantile = VISTA_1906.lam2_opt + VISTA_1906.lam1_opt
else: lam_results_quantile = VISTA_1906.lam1_opt
import json, pathlib
pathlib.Path(f"lams_{k}_{tune_start}.json").write_text(json.dumps({"lam_results": lam_results, 
                                                                   "lam_results_quantile": lam_results_quantile
                                                                   }))
'''

import json
with open(f"full_lams_{tune_start}.json","w") as f: json.dump(lam_results, f)
