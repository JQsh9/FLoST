import numpy as np
from scipy.ndimage import uniform_filter, maximum_filter, gaussian_filter

def smooth_tensor_mean_filter(tensor, size=3):
    tensor_filtered = np.zeros_like(tensor)
    for t in range(tensor.shape[2]):
        tensor_filtered[:,:,t] = uniform_filter(tensor[:,:,t], size=size, mode='nearest')
    return tensor_filtered

def smooth_tensor_maximum_filter(tensor, size=3):
    tensor_filtered = np.zeros_like(tensor)
    for t in range(tensor.shape[2]):
        tensor_filtered[:,:,t] = maximum_filter(tensor[:,:,t], size=size, mode='nearest')
    return tensor_filtered

def smooth_tensor_gaussian_filter(tensor,sigma=1.0, truncate=4.0):
    tensor_filtered = np.zeros_like(tensor)
    for t in range(tensor.shape[2]):
        tensor_filtered[:,:,t] = gaussian_filter(tensor[:,:,t], sigma=sigma,truncate=truncate, mode='nearest')
    return tensor_filtered

