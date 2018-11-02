# distutils: extra_compile_args = -O3 -fopenmp 
# distutils: extra_link_args = -fopenmp
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True

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


cpdef _blocks_to_bands_lower(double[:,:,::1] Ad, double[:, :, ::1] Aod):
    """
    Convert a block tridiagonal matrix to the banded matrix representation 
    required for scipy banded solvers. We are using the "lower form."

    C.f. https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solveh_banded.html
    """
    cdef int T, D, t, d, u, i, j, trow, drow
    cdef double[:, ::1] L

    T = Ad.shape[0]
    D = Ad.shape[1]
    L = np.zeros((2 * D, T * D))

    for t in range(T):
        for u in range(2 * D):
            for d in range(D):
                j = t * D + d
                i = u + j

                # Convert i into trow, drow indices
                trow = i // D
                drow = i % D

                if trow >= T:
                  continue 
                
                if t == trow:
                    L[u, j] = Ad[t, drow, d]
                elif t == trow - 1:
                    L[u, j] = Aod[t, drow, d]
                # else: L[u, j] = 0

    return np.asarray(L)


cpdef _blocks_to_bands_upper(double[:,:,::1] Ad, double[:, :, ::1] Aod):
    """
    Convert a block tridiagonal matrix to the banded matrix representation 
    required for scipy banded solvers. We are using the "upper form."

    C.f. https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solveh_banded.html
    """
    cdef int T, D, t, d, u, i, j, trow, drow
    cdef double[:, ::1] U

    T = Ad.shape[0]
    D = Ad.shape[1]
    U = np.zeros((2 * D, T * D))

    for t in range(T):
        for u in range(2 * D):
            for d in range(D):
                j = t * D + d
                i = u + j - (2 * D - 1)

                if i < 0:
                    continue

                # Convert i into trow, drow indices
                trow = i // D
                drow = i % D
                
                if trow >= T:
                    continue

                if t == trow:
                    U[u, j] = Ad[t, drow, d]
                elif t == trow + 1:
                    U[u, j] = Aod[t-1, drow, d]
                # else: U[u, j] = 0

    return np.asarray(U)


cpdef blocks_to_bands(double[:,:,::1] Ad, double[:, :, ::1] Aod, lower=True):
    """
    Convert a block tridiagonal matrix to the banded matrix representation 
    required for scipy banded solvers. 

    C.f. https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solveh_banded.html
    """
    assert Ad.ndim == 3  
    assert Ad.shape[2] == Ad.shape[1]
    assert Aod.ndim == 3
    assert Aod.shape[0] == Ad.shape[0]-1
    assert Aod.shape[1] == Ad.shape[1]
    assert Aod.shape[2] == Ad.shape[1]

    return _blocks_to_bands_lower(Ad, Aod) if lower else _blocks_to_bands_upper(Ad, Aod)
