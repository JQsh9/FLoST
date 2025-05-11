import numpy as np
import random
from sklearn.preprocessing import RobustScaler
#from scipy.special import boxcox, inv_boxcox
def randomly_drop_pixels(tensor, drop_rate, seed=2024):
        n1, n2, n3 = tensor.shape
        n = int(n1 * n2 * n3 * drop_rate)
        random.seed(seed)
        drop_indices = random.sample(range(n1 * n2 * n3), n)
        tensor_dropped = tensor.copy()
        tensor_dropped.reshape(-1)[drop_indices] = np.nan
        return tensor_dropped

def holdout_validation(tensor, val_ratio=0.10, seed=0):
    rng = np.random.default_rng(seed)
    obs_flat = np.flatnonzero(~np.isnan(tensor))
    n_val = int(np.ceil(val_ratio * obs_flat.size))
    val_flat = rng.choice(obs_flat, size=n_val, replace=False)
    val_idx  = np.unravel_index(val_flat, tensor.shape)
    train_tensor  = tensor.copy()
    train_tensor[val_idx] = np.nan    
    return train_tensor, val_idx
        
def get_singular_values(tensor,r,k):
    sv1 = []
    sv2 = []
    if np.any(np.isnan(tensor)):
        missing_rate = np.sum(np.isnan(tensor)) / tensor.size
        tensor = np.nan_to_num(tensor.copy(),0)/(1-missing_rate)
    tensor = np.fft.fft(tensor, axis=2)
    for i in range(k):
        U, S , Vh =np.linalg.svd(tensor[:,:,i], full_matrices=False)
        sv1.append(S[r-1])
        #sv2.append(S[r])
        sv2.append(min(S))
    return sv1, sv2

def robust_scaler(tensor # tensor with missing values
     ):
    mask = ~np.isnan(tensor)
    observed_vals = tensor[mask].reshape(-1, 1)
    scaler = RobustScaler(quantile_range=(25, 75))   # default IQR, tweak if needed
    scaled_vals = scaler.fit_transform(observed_vals)
    tensor[mask] = scaled_vals.ravel() 
    return tensor, scaler

def inverse_robust_scaler(tensor_hat, scaler):
   return scaler.inverse_transform(tensor_hat.reshape(-1, 1)).reshape(tensor_hat.shape)
    


'''
def log_boxcox(tensor, missing_rate, lambda_bc = None):
    #flatten = np.log(tensor).flatten()
    flatten = tensor.flatten()
    min=1
    #min = np.min(flatten)
    #flatten = flatten - min + 1
    #tensor_boxcox = (flatten**lambda_bc - 1) / lambda_bc
    tensor_boxcox = boxcox(flatten, lambda_bc)
    tensor_boxcox = tensor_boxcox.reshape(tensor.shape)
    tensor_dropped = randomly_drop_pixels(tensor_boxcox, missing_rate)
    return tensor_dropped,min

def inverse_log_boxcox(tensor, min, lambda_bc=None):
    if lambda_bc is None:
        raise ValueError("lambda_boxcox must be provided for log+boxcox method.") 
    d1, d2, d3 = tensor.shape
    tensor = inv_boxcox(tensor.flatten(), lambda_bc)
    #tensor = tensor + min - 1
    tensor = tensor.reshape(d1, d2, d3)
    #tensor = np.exp(tensor)
    return tensor
'''

        
