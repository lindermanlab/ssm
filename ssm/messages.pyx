# distutils: extra_compile_args = -O3
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np

from libc.math cimport log, exp, fmax, INFINITY

cdef double logsumexp(double[::1] x) nogil:
    cdef int i, N
    cdef double m, out

    N = x.shape[0]

    # find the max
    m = -INFINITY
    for i in range(N):
        m = fmax(m, x[i])

    # sum the exponentials
    out = 0
    for i in range(N):
        out += exp(x[i] - m)

    return m + log(out)


cdef dlse(double[::1] a,
          double[::1] out):

    cdef int K, k
    K = a.shape[0]
    cdef double lse = logsumexp(a)

    for k in range(K):
        out[k] = exp(a[k] - lse)


cpdef forward_pass(double[::1] log_pi0,
                   double[:,:,::1] log_Ps,
                   double[:,::1] log_likes,
                   double[:,::1] alphas):

    cdef int T, K, t, k, j
    T = log_likes.shape[0]
    K = log_likes.shape[1]
    assert log_Ps.shape[0] == T-1 or log_Ps.shape[0] == 1
    assert log_Ps.shape[1] == K
    assert log_Ps.shape[2] == K
    assert alphas.shape[0] == T
    assert alphas.shape[1] == K

    cdef double[::1] tmp = np.zeros(K)

    # Trick for handling time-varying transition matrices
    cdef int hetero = (log_Ps.shape[0] == T-1)

    for k in range(K):
        alphas[0, k] = log_pi0[k] + log_likes[0, k]

    for t in range(T - 1):
        for k in range(K):
            for j in range(K):
                tmp[j] = alphas[t, j] + log_Ps[t * hetero, j, k]
            alphas[t+1, k] = logsumexp(tmp) + log_likes[t+1, k]

    return logsumexp(alphas[T-1])

cpdef backward_pass(double[:,:,::1] log_Ps,
                    double[:,::1] log_likes,
                    double[:,::1] betas):

    cdef int T, K, t, k, j
    T = log_likes.shape[0]
    K = log_likes.shape[1]
    assert log_Ps.shape[0] == T-1 or log_Ps.shape[0] == 1
    assert log_Ps.shape[1] == K
    assert log_Ps.shape[2] == K
    assert betas.shape[0] == T
    assert betas.shape[1] == K

    cdef double[::1] tmp = np.zeros(K)

    # Trick for handling time-varying transition matrices
    cdef int hetero = (log_Ps.shape[0] == T-1)

    # Initialize the last output
    for k in range(K):
        betas[T-1, k] = 0

    for t in range(T-2,-1,-1):
        # betal[t] = logsumexp(Al + betal[t+1] + aBl[t+1],axis=1)
        for k in range(K):
            for j in range(K):
                tmp[j] = log_Ps[t * hetero, k, j] + betas[t+1, j] + log_likes[t+1, j]
            betas[t, k] = logsumexp(tmp)


cpdef backward_sample(double[:,:,::1] log_Ps,
                      double[:,::1] log_likes,
                      double[:,::1] alphas,
                      double[::1] us,
                      long[::1] zs):

    cdef int T, K, t, k, j
    cdef double Z, acc

    T = log_likes.shape[0]
    K = log_likes.shape[1]
    assert log_Ps.shape[0] == T-1 or log_Ps.shape[0] == 1
    assert log_Ps.shape[1] == K
    assert log_Ps.shape[2] == K
    assert alphas.shape[0] == T
    assert alphas.shape[1] == K
    assert us.shape[0] == T
    assert zs.shape[0] == T

    cdef double[::1] lpzp1 = np.zeros(K)
    cdef double[::1] lpz = np.zeros(K)

    # Trick for handling time-varying transition matrices
    cdef int hetero = (log_Ps.shape[0] == T-1)

    for t in range(T-1,-1,-1):
        # compute normalized log p(z[t] = k | z[t+1])
        for k in range(K):
            lpz[k] = lpzp1[k] + alphas[t, k]
        Z = logsumexp(lpz)

        # sample
        acc = 0
        zs[t] = K-1
        for k in range(K):
            acc += np.exp(lpz[k] - Z)
            if us[t] < acc:
                zs[t] = k
                break

        # set the transition potential
        if t > 0:
            for k in range(K):
                lpzp1[k] = log_Ps[(t-1) * hetero, k, zs[t]]


cpdef grad_hmm_normalizer(double[:,:,::1] log_Ps,
                          double[:,::1] alphas,
                          double[::1] d_log_pi0,
                          double[:,:,::1] d_log_Ps,
                          double[:,::1] d_log_likes):

    cdef int T, K, t, k, j

    T = alphas.shape[0]
    K = alphas.shape[1]
    assert (log_Ps.shape[0] == T-1) or (log_Ps.shape[0] == 1)
    assert d_log_Ps.shape[0] == log_Ps.shape[0]
    assert log_Ps.shape[1] == d_log_Ps.shape[1] == K
    assert log_Ps.shape[2] == d_log_Ps.shape[2] == K
    assert d_log_pi0.shape[0] == K
    assert d_log_likes.shape[0] == T
    assert d_log_likes.shape[1] == K

    # Initialize temp storage for gradients
    cdef double[::1] tmp1 = np.zeros((K,))
    cdef double[:, ::1] tmp2 = np.zeros((K, K))

    # Trick for handling time-varying transition matrices
    cdef int hetero = (log_Ps.shape[0] == T-1)

    dlse(alphas[T-1], d_log_likes[T-1])
    for t in range(T-1, 0, -1):
        # tmp2 = dLSE_da(alphas[t-1], log_Ps[t-1])
        #      = np.exp(alphas[t-1] + log_Ps[t-1].T - logsumexp(alphas[t-1] + log_Ps[t-1].T, axis=1))
        #      = [dlse(alphas[t-1] + log_Ps[t-1, :, k]) for k in range(K)]
        for k in range(K):
            for j in range(K):
                tmp1[j] = alphas[t-1, j] + log_Ps[(t-1) * hetero, j, k]
            dlse(tmp1, tmp2[k])


        # d_log_Ps[t-1] = vjp_LSE_B(alphas[t-1], log_Ps[t-1], d_log_likes[t])
        #               = d_log_likes[t] * dLSE_da(alphas[t-1], log_Ps[t-1]).T
        #               = d_log_likes[t] * tmp2.T
        #
        # d_log_Ps[t-1, j, k] = d_log_likes[t, k] * tmp2.T[j, k]
        #                     = d_log_likes[t, k] * tmp2[k, j]
        for j in range(K):
            for k in range(K):
                d_log_Ps[(t-1) * hetero, j, k] += d_log_likes[t, k] * tmp2[k, j]

        # d_log_likes[t-1] = d_log_likes[t].dot(dLSE_da(alphas[t-1], log_Ps[t-1]))
        #                  = d_log_likes[t].dot(tmp2)
        for k in range(K):
            d_log_likes[t-1, k] = 0
            for j in range(K):
                d_log_likes[t-1, k] += d_log_likes[t, j] * tmp2[j, k]

    # d_log_pi0 = d_log_likes[0]
    for k in range(K):
        d_log_pi0[k] = d_log_likes[0, k]
