# Written by Jiuqian Shang

import numpy as np
import math

def t_prod(A,B):
    [a1,a2,a3] = A.shape
    [b1,b2,b3] = B.shape
    C = np.zeros((a1,b2,b3), dtype = complex)
    A = np.fft.fft(A, axis=2)
    B = np.fft.fft(B, axis=2)
    half_n3 = math.ceil( (a3 + 1)/2 )
    for i in range(half_n3):
        C[:, :, i] = np.dot(A[:, :, i], B[:, :, i])
    
    for i in range(half_n3, a3):
        C[:, :, i] = np.conj(C[:, :, a3 - i])
    C = np.fft.ifft(C, axis=2)
    return C  

# Computation complexity: O(a1 * a2  * b2 * b3)