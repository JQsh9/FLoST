import numpy as np
import math
from scipy.linalg import svd

def sparse_low_rank(tensor, # with missingness pixels
                    missing_rate,
                     k,  lam1,lam2):
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

    # sparse part
    real_part = tensor[:,:,k:half_d3].flatten().real
    imag_part = tensor[:,:,k:half_d3].flatten().imag

    t_hat[:,:,k:half_d3] = np.maximum(real_part - lam2[0], 0).reshape(d1, d2, half_d3 - k )
    t_hat[:,:,k:half_d3] = t_hat[:,:,k:half_d3] + 1j * np.maximum(imag_part - lam2[1], 0).reshape(d1, d2, half_d3 - k)
    
    # symmetrize the tensor
    for i in range(half_d3, d3):
        t_hat[:, :, i] = np.conj(t_hat[:, :, d3 - i])
    return np.fft.ifft(t_hat, axis=2).real

