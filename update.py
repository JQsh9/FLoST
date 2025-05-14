import numpy as np
import math
import sys
sys.path.append("tensor_toolbox")
from t_prod import t_prod
from conjugate_transpose import conjugate_transpose
from t_qr import t_qr
import time

def update(U, S, V, Z):

    n1, n2, n3 = Z.shape
    r = S.shape[0]
    
    VH = conjugate_transpose(V)
    ZH = conjugate_transpose(Z)

    #start_time = time.time()
    ZHU = t_prod(ZH, U) # n2 * r * n3 Computation: O(n2 * n1 * r * n3)
    #end_time = time.time()
    #print("ZHU time: ", end_time - start_time)

    #start_time = time.time()
    VHZHU = t_prod(VH, ZHU) # r * r * n3 Computation: O(r^2 * n2 * n3)
    #end_time = time.time()  
    #print("VHZHU time: ", end_time - start_time)

    #start_time = time.time()
    Y1 = ZHU - t_prod(V, VHZHU) # n2 * r * n3 Computation: O(r^2 * n2 * n3)
    #end_time = time.time()
    #print("Y1 time: ", end_time - start_time)


    UHZV = conjugate_transpose(VHZHU)
    Y2 = t_prod(Z, V) - t_prod(U, UHZV) # n1 * r * n3 Computation: O(n1 * n2 * r * n3)

    #start_time = time.time()
    Q1, R1 = t_qr(Y1, r)
    Q2, R2 = t_qr(Y2, r)
    #end_time = time.time()
    #print("QR time: ", end_time - start_time)

    #start_time = time.time()
    U_fft = np.fft.fft(U, axis=2) ; VH_fft = np.fft.fft(VH, axis=2)
    upper_left = S + UHZV
    upper_left_fft = np.fft.fft(upper_left, axis=2)
    Q1H = conjugate_transpose(Q1); R1H = conjugate_transpose(R1)
    Q1H_fft = np.fft.fft(Q1H, axis=2); R1H_fft = np.fft.fft(R1H, axis=2)
    Q2_fft = np.fft.fft(Q2, axis=2); R2_fft = np.fft.fft(R2, axis=2)
    #end_time = time.time()
    #print("FFT time: ", end_time - start_time)

    X_new = np.zeros((n1, n2, n3), dtype=complex)
    U_new = np.zeros((n1, r, n3), dtype=complex)
    S_new = np.zeros((r, r, n3), dtype=complex)
    V_new = np.zeros((n2, r, n3), dtype=complex)

    #start_time = time.time()
    half_n3 = math.ceil( (n3 + 1)/2 )
    for i in range(half_n3):
        M = np.concatenate((upper_left_fft[:,:,i], R1H_fft[:,:,i]), axis=1) # M is a matrix on frequency domain
        lower = np.concatenate((R2_fft[:,:,i], np.zeros((r, r), dtype=complex)), axis=1)
        M = np.concatenate((M, lower), axis=0) # 2r * 2r
        left = np.concatenate((U_fft[:,:,i], Q2_fft[:,:,i]), axis=1) # n1 * 2r
        right = np.concatenate((VH_fft[:,:,i], Q1H_fft[:,:,i]), axis=0) # 2r * n2
        # W = left @ M @ right
        Ui, Si, VHi = np.linalg.svd(M, full_matrices=False) # Computation: O(r^3)
        left = left @ Ui[:, :r]
        right = VHi[:r, :] @ right
        # retraction to rank r
        X_new[:,:,i] = left @ np.diag(Si[:r]) @ right
        U_new[:,:,i] = left; 
        S_new[:,:,i] = np.diag(Si[:r]); 
        V_new[:,:,i] = right.T.conj()
    for i in range(half_n3, n3):
        X_new[:, :, i] = np.conj(X_new[:, :, n3 - i])
        U_new[:, :, i] = np.conj(U_new[:, :, n3 - i])
        S_new[:, :, i] = np.conj(S_new[:, :, n3 - i])
        V_new[:, :, i] = np.conj(V_new[:, :, n3 - i])
    
    #end_time = time.time()
    #print("SVD time: ", end_time - start_time)
    
    X_new = np.fft.ifft(X_new, axis=2).real
    U_new = np.fft.ifft(U_new, axis=2).real
    S_new = np.fft.ifft(S_new, axis=2).real
    V_new = np.fft.ifft(V_new, axis=2).real

    return X_new, U_new, S_new, V_new