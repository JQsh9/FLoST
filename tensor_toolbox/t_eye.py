# written by JS

import numpy as np

def t_eye(n,n3):
    # output: dentity tensor of size n*n*n3
    I = np.zeros((n,n,n3))
    I[:,:,0] = np.eye(n)
    return I