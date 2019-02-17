# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np

from cython.parallel import prange

# Cython functions for computing sufficient statistics
# of t-distributed AR models
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


# Cython functions to convert between banded and block tridiagonal matrices
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


# Now do the reverse -- convert bands to blocks
cpdef _bands_to_blocks_lower(double[:, ::1] A_banded):
    """
    Convert a banded matrix to a block tridiagonal matrix using the "lower form."

    C.f. https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solveh_banded.html
    """
    cdef int T, D, t, d, u, i, j, trow, drow
    cdef double[:, :, ::1] Ad, Aod

    D = A_banded.shape[0] // 2
    assert A_banded.shape[0] == D * 2
    T = A_banded.shape[1] // D
    assert A_banded.shape[1] == T * D

    Ad = np.zeros((T, D, D))
    Aod = np.zeros((T-1, D, D))

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
                    Ad[t, drow, d] = A_banded[u, j]
                elif t == trow - 1:
                    Aod[t, drow, d] = A_banded[u, j]

    # Fill in the upper triangle of the diagonal blocks
    # for t in range(T):
    #     for d in range(D):
    #         for drow in range(d):
    #             Ad[t, drow, d] = Ad[t, d, drow]

    return np.asarray(Ad), np.asarray(Aod)


cpdef _bands_to_blocks_upper(double[:,::1] A_banded):
    """
    Convert a banded matrix to a block tridiagonal matrix using the "upper form."

    C.f. https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solveh_banded.html
    """
    cdef int T, D, t, d, u, i, j, trow, drow
    cdef double[:, :, ::1] Ad, Aod

    D = A_banded.shape[0] // 2
    assert A_banded.shape[0] == D * 2
    T = A_banded.shape[1] // D
    assert A_banded.shape[1] == T * D

    Ad = np.zeros((T, D, D))
    Aod = np.zeros((T-1, D, D))

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
                    Ad[t, drow, d] = A_banded[u, j]
                elif t == trow + 1:
                    Aod[t-1, drow, d] = A_banded[u, j]
                # else: U[u, j] = 0

    # Fill in the lower triangle of the diagonal blocks
    # for t in range(T):
    #     for drow in range(D):
    #         for d in range(drow):
    #             Ad[t, drow, d] = Ad[t, d, drow]

    return np.asarray(Ad), np.asarray(Aod)


cpdef _transpose_banded(int l, int u, double[:, ::1] A_banded):
    cdef int d, i, dd, j, D, N
    D = A_banded.shape[0]
    N = A_banded.shape[1]

    cdef double[:, ::1] A_banded_T = np.zeros_like(A_banded)

    for d in range(D):
        for j in range(N):
            # Writing entry from
            # A.T[i, j] = A[j, i] = A_banded[u + j - i, i] = A_banded[u + l - D] = A_banded[D - 1 - d]
            i = d + j - l
            if i < 0 or i >= N:
                continue

            A_banded_T[d, j] = A_banded[D-1-d, i]

    return np.asarray(A_banded_T)


cpdef vjp_cholesky_banded_lower(double[:, ::1] L_bar,
                                double[:, ::1] L_banded,
                                double[:, ::1] A_banded,
                                double[:, ::1] A_bar):
    """
    Fill in A_bar with the elementwise gradient of L wrt A times L_bar

    NOTE: L_bar is updated in place!
    """

    cdef int D, N, i, j, k
    D = A_banded.shape[0]
    N = A_banded.shape[1]

    # Compute the gradient
    for i in range(N-1, -1, -1):
        for j in range(i, max(i-D, -1), -1):
            if j == i:
                # A_bar[i, i] = 0.5 * L_bar[i, i] / L[i, i]
                A_bar[0, j] = 0.5 * L_bar[0, j] / L_banded[0, j]
            else:
                # A_bar[i, j] = L_bar[i, j] / L[j, j]
                A_bar[i-j, j] = L_bar[i - j, j] / L_banded[0, j]
                # L_bar[j, j] -= L_bar[i, j] * L[i, j] / L[j, j]
                L_bar[0, j] -= L_bar[i - j, j] * L_banded[i - j, j] / L_banded[0, j]

            for k in range(j-1, max(i-D, -1), -1):
                L_bar[i-k, k] -= A_bar[i-j, j] * L_banded[j-k, k]
                L_bar[j-k, k] -= A_bar[i-j, j] * L_banded[i-k, k]


cpdef _vjp_solve_banded_A(double[:, ::1] A_bar,
                          double[:, ::1] b_bar,
                          double[:, ::1] C_bar,
                          double[:, ::1] C,
                          int u,
                          double[:, ::1] A_banded):

    cdef int D, N, K, d, j, i, k

    D = A_banded.shape[0]
    N = A_banded.shape[1]
    K = C_bar.shape[1]

    # Fill in the gradients of the banded matrix
    for d in range(D):
        for j in range(N):
            i = d + j - u
            if i >= 0 and i < N:
                for k in range(K):
                    A_bar[d, j] -= b_bar[i, k] * C[j, k]


cpdef _vjp_solveh_banded_A(double[:, ::1] A_bar,
                           double[:, ::1] b_bar,
                           double[:, ::1] C_bar,
                           double[:, ::1] C,
                           bint lower,
                           double[:, ::1] A_banded):

    cdef int D, N, K, d, j, i, k

    D = A_banded.shape[0]
    N = A_banded.shape[1]
    K = C_bar.shape[1]

    # Fill in the gradients of the banded matrix
    for j in range(N):
        for d in range(D):
            i = d + j if lower else d + j - D + 1
            if i < 0 or i >= N:
                continue

            for k in range(K):
                A_bar[d, j] -= b_bar[i, k] * C[j, k]
                # If off-diagonal, also include the cross term
                if i != j:
                    A_bar[d, j] -= b_bar[j, k] * C[i, k]
