
import numpy as np
from scipy.io import loadmat
import math



def generate_tensor_vista(dates):
    data = loadmat('../data/' + dates[0] + '_VISTA.mat')
    tec_MF_lt = data['tec_MF_lt']
    tec_VISTA = data['imputed']
    for day in range(1,len(dates)):
        data = loadmat('../data/' + dates[day] + '_VISTA.mat')
        tec_MF_lt = np.concatenate((tec_MF_lt, data['tec_MF_lt']), axis=2)
        tec_VISTA = np.concatenate((tec_VISTA, data['imputed']), axis=2)
    return tec_VISTA, tec_MF_lt # tec_VISTA is the complete tensor, tec_MF_lt is with real missingness

def generate_tensor_simulated(d1, d2, d3, k = 10, s=100000,r=5, sigma=0.1,seed = 2024):
    np.random.seed(seed)
    tensor = np.random.rand(d1, d2, d3)
    tensor = np.fft.fft(tensor, axis=2)
    for i in range(k):
        U, S, Vh = np.linalg.svd(tensor[:, :, i], full_matrices=False)
        S[r:] = 0
        tensor[:, :, i] = np.dot(U, np.dot(np.diag(S), Vh))
    half_d3 = math.ceil( (d3 + 1)/2 )
    sparse_part = tensor[:,:,k:half_d3]
    flattened = np.abs(sparse_part).flatten()
    threshold = np.partition(flattened, -s)[-s]
    sparse_part[np.abs(sparse_part) < threshold] = 0
    tensor[:,:,k:half_d3] = sparse_part
    for i in range(half_d3, d3):
        tensor[:, :, i] = np.conj(tensor[:, :, d3 - i])
    tensor = np.fft.ifft(tensor, axis=2).real
    np.random.seed(seed)
    tensor_n = tensor + np.random.normal(0, sigma, (d1, d2, d3))
    return tensor, tensor_n # tensor_n is the noisy tensor, tensor is the true underlying tensor
        
