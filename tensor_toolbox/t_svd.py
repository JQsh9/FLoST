# Written by Jiuqian Shang

import numpy as np
import math
from conjugate_transpose import conjugate_transpose
def t_svd_skinny(A):
    """
    Perform Tensor skinny Singular Value Decomposition (T-SVD) on a third-order tensor A.
    Fourier transform is used to compute the T-SVD.
    'skinny': produces the skinny tensor SVD.
    Let r be the tensor tubal rank of A. Then, A = U*S*V^*, where
    U - n1*r*n3
    S - r*r*n3
    V - n2*r*n3
    """
    n1, n2, n3 = A.shape
    
    # Step 1: Compute the FFT of A along the third dimension
    A_fft = np.fft.fft(A, axis=2)
    
    # Compute tubal rank
    U1, s1, Vh1 = np.linalg.svd(A_fft[:, :, 0], full_matrices=False)
    tol = 1e-10
    trank = np.sum(s1 > tol)

    # Initialize U, S, V
    U = np.zeros((n1, trank, n3), dtype=complex)
    S = np.zeros((trank, trank, n3), dtype=complex)
    V = np.zeros((n2, trank, n3), dtype=complex)
    mrank = []
    mrank.append(trank)

    # Step 2: Compute SVD of each frontal slice of A_fft
    U[:, :, 0] = U1[:, :trank]
    S[:trank, :trank, 0] = np.diag(s1[:trank])
    Vh1r = Vh1[:trank,:]
    V[:, :, 0] = Vh1r.T.conj()
    
    half_n3 = math.ceil( (n3 + 1)/2 )
    for i in range(1, half_n3):
        Ui, Si, Vhi = np.linalg.svd(A_fft[:, :, i], full_matrices=False)
        ri =np.sum(Si > tol)
        U[:, :, i] = Ui[:, :trank]
        S[:ri, :ri, i] = np.diag(Si[:ri])
        Vhir = Vhi[:trank,:]
        V[:, :, i] = Vhir.T.conj()
        mrank.append(ri)
    
    for i in range(half_n3, n3):
        U[:, :, i] = np.conj(U[:, :, n3 - i])
        S[:, :, i] = S[:, :, n3 - i]
        V[:, :, i] = np.conj(V[:, :, n3 - i])
        mrank.append(mrank[n3 - i])
    
    # Step 3: Compute the IFFT of U, S, V along the third dimension
    U_ifft = np.fft.ifft(U, axis=2)
    S_ifft = np.fft.ifft(S, axis=2)
    V_ifft = np.fft.ifft(V, axis=2)

    return U_ifft.real, S_ifft.real, V_ifft.real, mrank

def t_svd_truncation(A, r):
    #skinny truncated SVD

    n1, n2, n3 = A.shape
    
    # Step 1: Compute the FFT of A along the third dimension
    A_fft = np.fft.fft(A, axis=2)
    
    # Initialize U, S, V
    U = np.zeros((n1, r, n3), dtype=complex)
    S = np.zeros((r, r, n3), dtype=complex)
    V = np.zeros((n2, r, n3), dtype=complex)
    
    # Step 2: Compute SVD of each frontal slice of A_fft
    half_n3 = math.ceil( (n3 + 1)/2 )
    for i in range(half_n3):
        Ui, Si, Vhi = np.linalg.svd(A_fft[:, :, i],full_matrices=True)
        U[:, :, i] = Ui[:, :r]
        S[:, :, i] = np.diag(Si[:r])
        Vhir = Vhi[:r,:]
        V[:, :, i] = Vhir.T.conj()
    
    for i in range(half_n3, n3):
        U[:, :, i] = np.conj(U[:, :, n3 - i])
        S[:, :, i] = S[:, :, n3 - i]
        V[:, :, i] = np.conj(V[:, :, n3 - i])
    
    # Step 3: Compute the IFFT of U, S, V along the third dimension
    U_ifft = np.fft.ifft(U, axis=2)
    S_ifft = np.fft.ifft(S, axis=2)
    V_ifft = np.fft.ifft(V, axis=2)
    
    return U_ifft, S_ifft, V_ifft

# Computation complexity: O(n1 * n2 * min(n1, n2) *n3)