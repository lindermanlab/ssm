# distutils: extra_compile_args = -O3 -fopenmp 
# distutils: extra_link_args = -fopenmp
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=False

import numpy as np
cimport numpy as np

from cython.parallel import prange

cpdef robust_ar_statistics(double[:, ::1] Ez,
                           double[:, :, ::1] tau,
                           double[:, ::1] x,
                           double[:, ::1] y,
                           double[:, :, :, ::1] J,
                           double[:, :, ::1] h):
    
    cdef int t, T, k, K, d, D, m, n, N 
    T = Ez.shape[0]
    K = Ez.shape[1]
    D = tau.shape[2]
    N = x.shape[1]

    with nogil:
        for k in prange(K):
            for t in range(T):
                for d in range(D):
                    for m in range(N):
                        for n in range(N):
                            J[k, d, m, n] += Ez[t, k] * tau[t, k, d] * x[t, m] * x[t, n]

                        h[k, d, m] += Ez[t, k] * tau[t, k, d] * x[t, m] * y[t, d]                
