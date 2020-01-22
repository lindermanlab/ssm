import numpy as np

def robust_ar_statistics(double[:, ::1] Ez,
                           double[:, :, ::1] tau,
                           double[:, ::1] x,
                           double[:, ::1] y,
                           double[:, :, :, ::1] J,
                           double[:, :, ::1] h):
    raise NotImplementedError


# Cython functions to convert between banded and block tridiagonal matrices
def _blocks_to_bands_lower(double[:,:,::1] Ad, double[:, :, ::1] Aod):
    raise NotImplementedError


def _blocks_to_bands_upper(double[:,:,::1] Ad, double[:, :, ::1] Aod):
    raise NotImplementedError


# Now do the reverse -- convert bands to blocks
def _bands_to_blocks_lower(double[:, ::1] A_banded):
    raise NotImplementedError


def _bands_to_blocks_upper(double[:,::1] A_banded):
    raise NotImplementedError


def _transpose_banded(int l, int u, double[:, ::1] A_banded):
    raise NotImplementedError


def vjp_cholesky_banded_lower(double[:, ::1] L_bar,
                                double[:, ::1] L_banded,
                                double[:, ::1] A_banded,
                                double[:, ::1] A_bar):
    raise NotImplementedError


def _vjp_solve_banded_A(double[:, ::1] A_bar,
                          double[:, ::1] b_bar,
                          double[:, ::1] C_bar,
                          double[:, ::1] C,
                          int u,
                          double[:, ::1] A_banded):
    raise NotImplementedError


def _vjp_solveh_banded_A(double[:, ::1] A_bar,
                           double[:, ::1] b_bar,
                           double[:, ::1] C_bar,
                           double[:, ::1] C,
                           bint lower,
                           double[:, ::1] A_banded):
    raise NotImplementedError
