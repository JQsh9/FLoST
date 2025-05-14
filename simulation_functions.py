import math
import numpy as np
import random

def generate_tensor_simulated(d1, d2, d3, k = 10, s=0.2,r=5):
    np.random.seed(0)
    tensor = np.random.randn(d1, d2, d3)
    tensor = np.fft.fft(tensor, axis=2)
    for i in range(k):
        U, S, Vh = np.linalg.svd(tensor[:, :, i], full_matrices=False)
        S[r:] = 0
        tensor[:, :, i] = np.dot(U, np.dot(np.diag(S), Vh))
    half_d3 = math.ceil( (d3 + 1)/2 )
    if half_d3> k:
        sparse_part = tensor[:,:,k:half_d3]
        flattened = np.abs(sparse_part).flatten()
        s = int(len(flattened) * s)
        threshold = np.partition(flattened, -s)[-s]
        sparse_part[np.abs(sparse_part) < threshold] = 0
        tensor[:,:,k:half_d3] = sparse_part
    for i in range(half_d3, d3):
        tensor[:, :, i] = np.conj(tensor[:, :, d3 - i])
    tensor = np.fft.ifft(tensor, axis=2).real
    return tensor# tensor_n is the noisy tensor, tensor is the true underlying tensor

def get_singular_values(tensor_m,missing_rate, r,k):
    sv1 = []
    sv2 = []
    tensor = np.nan_to_num(tensor_m.copy(),0)/(1-missing_rate)
    tensor = np.fft.fft(tensor, axis=2)
    for i in range(k):
        U, S , Vh =np.linalg.svd(tensor[:,:,i], full_matrices=False)
        sv1.append(S[r-1])
        sv2.append(S[r])
    return sv1, sv2

def randomly_drop_pixels(tensor, drop_rate, seed=0):
        d1, d2, d3= tensor.shape
        n = int(d1* d2* d3 * drop_rate)
        random.seed(seed)
        drop_indices = random.sample(range(d1* d2*d3), n)
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
    missing_rate = np.sum(np.isnan(train_tensor)) / train_tensor.size
    return train_tensor, val_idx, missing_rate

def sparse_low_rank(tensor, # with missingness pixels
                    missing_rate, k,  lam1,lam2):
    tensor = np.nan_to_num(tensor, nan=0) /(1 - missing_rate)
    tensor = np.fft.fft(tensor, axis=2)
    t_hat = np.zeros(tensor.shape)
    d1, d2, d3 = tensor.shape
    # low rank part
    
    for i in range(k):
        U, S , Vh =np.linalg.svd(tensor[:,:,i], full_matrices=False)
        S = [max(x, 0) for x in S - lam1[i]]
        t_hat[:,:,i] = np.dot(U, np.dot(np.diag(S), Vh))
    half_d3 = math.ceil( (d3 + 1)/2 )
    if half_d3> k:
        # sparse part
        real_part = tensor[:,:,k:half_d3].flatten().real
        imag_part = tensor[:,:,k:half_d3].flatten().imag

        t_hat[:,:,k:half_d3] = np.maximum(real_part - lam2[0], 0).reshape(d1, d2, half_d3 - k )
        t_hat[:,:,k:half_d3] = t_hat[:,:,k:half_d3] + 1j * np.maximum(imag_part - lam2[1], 0).reshape(d1, d2, half_d3 - k)
   # symmetrize the tensor
    for i in range(half_d3, d3):
        t_hat[:, :, i] = np.conj(t_hat[:, :, d3 - i])
    return np.fft.ifft(t_hat, axis=2).real

        

import sys
sys.path.append("tensor_toolbox")
from t_prod import t_prod
from t_svd import t_svd_truncation
from t_svd import t_svd_skinny
from conjugate_transpose import conjugate_transpose
from project_Tl import project_Tl
from ROmega import ROmega
from update import update

def tensor_rcgd(X, # tensor to be completed
                trank, #tubal rank 
                monitor_seed=2024, # seed for random selection of monitoring set
                monitor_size=0.1, # proportion of monitoring set (in observed set)
                tol = 0, # tolerance for stopping criterion
                max_iter=200): 
    """Tensor Riemannian Conjugate Gradient Descent."""

    # validation set used to monitor the Alg.
    o_inds = np.where(~np.isnan(X))
    m_inds = np.where(np.isnan(X))
    monitor_size = int(len(o_inds[0]) * monitor_size)
    random.seed(monitor_seed)
    monitor_inds = random.sample(range(len(o_inds[0])), monitor_size)
    monitor_rows, monitor_cols, monitor_tubes = o_inds[0][monitor_inds], o_inds[1][monitor_inds], o_inds[2][monitor_inds]
    
    blind_inds=(np.concatenate((m_inds[0],monitor_rows)),np.concatenate((m_inds[1], monitor_cols)),np.concatenate((m_inds[2], monitor_tubes)))
   # blind_inds = (np.concatenate((blind_inds[0],t_inds[0])), np.concatenate((blind_inds[1],t_inds[1])), np.concatenate((blind_inds[2],t_inds[2])))

    # Initialization
    Xl = X.copy()
    Xl[blind_inds[0], blind_inds[1], blind_inds[2]] =0
    missing_ratio = len(blind_inds[0]) / Xl.size
    Ul ,Sl, Vl = t_svd_truncation(Xl/(1-missing_ratio), trank)
    Xl = t_prod(Ul, t_prod(Sl, conjugate_transpose(Vl)))    

    for l in range(max_iter):
        G = ROmega(X-Xl,blind_inds)
        ptlG = project_Tl(G, Ul, Vl)

        if l == 0:
            Q = ptlG
            # Step Size
            alpha = np.sum(Q * Q) / np.sum(Q* ROmega(Q,blind_inds)) 
            Xl_new, Ul_new, Sl_new, Vl_new= update(Ul, Sl, Vl, alpha*G)

            Xlnew_monitor = Xl_new[monitor_rows, monitor_cols, monitor_tubes]
            X_true =X[monitor_rows, monitor_cols, monitor_tubes] 
            monitor= np.linalg.norm(Xlnew_monitor-X_true)/ np.linalg.norm(X_true)
            #print(l,monitor)
            Xl = Xl_new; Ul = Ul_new; Sl = Sl_new; Vl = Vl_new
            Q_prev = Q
            continue
        
        ptlQ_prev = project_Tl(Q_prev, Ul, Vl) 
        beta = - np.sum(ptlG *  ROmega(ptlQ_prev,blind_inds)) / np.sum(ptlQ_prev * ROmega(ptlQ_prev,blind_inds))
        Q = ptlG + beta * ptlQ_prev 
        # Step Size
        alpha = np.sum(ptlG * Q) / np.sum(Q* ROmega(Q,blind_inds)) 
        Xl_new, Ul_new, Sl_new, Vl_new= update(Ul, Sl, Vl, alpha*(G+beta*Q_prev))

        # Check for stopping condition
        Xlnew_monitor = Xl_new[monitor_rows, monitor_cols, monitor_tubes] 
        X_true = X[monitor_rows, monitor_cols, monitor_tubes]
        monitor_new= np.linalg.norm(Xlnew_monitor-X_true)/ np.linalg.norm(X_true)
        #print(l,monitor_new)
        if monitor - monitor_new < tol:
            #print('reached the optimal res')
            break
        monitor = monitor_new
        
        Xl = Xl_new; Ul = Ul_new; Sl = Sl_new; Vl = Vl_new
        Q_prev = Q
        
    return Xl