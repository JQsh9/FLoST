# Written by Jiuqian Shang

import numpy as np
import math

# Tensor orthogonal-triangular decomposition under linear transform
def t_qr(A, r):
    # Input: A n1*r*n3 tensor with tubal rank r
    n1, n2, n3 = A.shape
    if n2 != r:
        raise ValueError("The second dimension of A must be equal to r")
    A_fft = np.fft.fft(A, axis=2)
    Q = np.zeros((n1, r, n3), dtype=complex)
    R = np.zeros((r, r, n3), dtype=complex)

    half_n3 = math.ceil( (n3 + 1)/2 )
    for i in range(half_n3):
        Qi, Ri = np.linalg.qr(A_fft[:,:,i])
        Q[:,:,i] = Qi
        R[:,:,i] = Ri
    
    for i in range(half_n3, n3):
        Q[:, :, i] = np.conj(Q[:, :, n3 - i])
        R[:, :, i] = np.conj(R[:, :, n3 - i])
    
    Q_ifft = np.fft.ifft(Q, axis=2)
    R_ifft = np.fft.ifft(R, axis=2)
    
    return Q_ifft, R_ifft