import numpy as np


def robust_ar_statistics(Ez,
                         tau,
                         x,
                         y,
                         J,
                         h):
    raise NotImplementedError


# Cython functions to convert between banded and block tridiagonal matrices
def _blocks_to_bands_lower(Ad, Aod):
    raise NotImplementedError


def _blocks_to_bands_upper(Ad, Aod):
    raise NotImplementedError


# Now do the reverse -- convert bands to blocks
def _bands_to_blocks_lower(A_banded):
    raise NotImplementedError


def _bands_to_blocks_upper(A_banded):
    raise NotImplementedError


def _transpose_banded(l, u, A_banded):
    raise NotImplementedError


def vjp_cholesky_banded_lower(L_bar,
                              L_banded,
                              A_banded,
                              A_bar):
    raise NotImplementedError


def _vjp_solve_banded_A(A_bar,
                        b_bar,
                        C_bar,
                        C,
                        u,
                        A_banded):
    raise NotImplementedError


def _vjp_solveh_banded_A(A_bar,
                         b_bar,
                         C_bar,
                         C,
                         lower,
                         A_banded):
    raise NotImplementedError
