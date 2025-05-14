# Written by Jiuqian Shang

import numpy as np

def conjugate_transpose(tensor):
    n1, n2, n3= tensor.shape
    tensor_H = np.zeros((n2, n1, n3), dtype = complex)
    tensor_H[:,:,0] = tensor[:,:,0].T.conj()
    for i in range(1,n3):
        tensor_H[:,:,i] = tensor[:,:,n3-i].T.conj()
    return tensor_H