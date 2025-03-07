"""Maxvol implementation.

This submodule is taken from maxvol-py.
https://github.com/c-f-h/maxvolpy

The library is not compatible with our version of numpy. We reproduced the code here.
"""

import numpy as np
from scipy.linalg import get_blas_funcs, get_lapack_funcs


def maxvol(
    A: np.ndarray, tol: float = 1.05, max_iters: int = 100, top_k_index: int = -1
):
    """Maxvol implementation.

    Adapted from maxvolpy

    Uses greedy iterative maximization of 1-volume to find good
    `r`-by-`r` submatrix in a given `N`-by-`r` matrix `A` of rank `r`.
       Returns good submatrix and coefficients of expansion
       (`N`-by-`r` matrix) of rows of matrix `A` by rows of good submatrix.

    Args:
       A : numpy.ndarray(ndim=2)
           Real or complex matrix of shape `(N, r)`, `N >= r`.
       tol : float, optional
           Upper bound for infinite norm of coefficients of expansion of
           rows of `A` by rows of good submatrix. Minimum value is 1.
           Default to `1.05`.
       max_iters : integer, optional
           Maximum number of iterations. Each iteration swaps 2 rows.
           Defaults to `100`.
       top_k_index : integer, optional
           Pivot rows for good submatrix will be in range from `0` to
           `(top_k_index-1)`. This restriction is ignored, if `top_k_index`
           is -1. Defaults to `-1`.

    Returns:
       piv : numpy.ndarray(ndim=1, dtype=numpy.int32)
           Rows of matrix `A`, corresponding to submatrix, good in terms
           of 1-volume. Shape is `(r, )`.
       C : numpy.ndarray(ndim=2)
           Matrix of coefficients of expansions of all rows of `A` by good
           rows `piv`. Shape is `(N, r)`.
    """
    if tol < 1:
        tol = 1.0
    N, r = A.shape
    if N <= r:
        return np.arange(N, dtype=np.int32), np.eye(N, dtype=A.dtype)
    if top_k_index == -1 or top_k_index > N:
        top_k_index = N
    if top_k_index < r:
        top_k_index = r
    # set auxiliary matrices and get corresponding *GETRF function
    # from lapack
    B = np.copy(A[:top_k_index], order="F")
    C = np.copy(A.T, order="F")
    H, ipiv, info = get_lapack_funcs("getrf", [B])(B, overwrite_a=1)
    # compute pivots from ipiv (result of *GETRF)
    index = np.arange(N, dtype=np.int32)
    for i in range(r):
        tmp = index[i]
        index[i] = index[ipiv[i]]
        index[ipiv[i]] = tmp
    # solve A = CH, H is in LU format
    B = H[:r]
    # It will be much faster to use *TRSM instead of *TRTRS
    trtrs = get_lapack_funcs("trtrs", [B])
    trtrs(B, C, trans=1, lower=0, unitdiag=0, overwrite_b=1)
    trtrs(B, C, trans=1, lower=1, unitdiag=1, overwrite_b=1)
    # C has shape (r, N) -- it is stored transposed
    # find max value in C
    i, j = divmod(abs(C[:, :top_k_index]).argmax(), top_k_index)
    # set cgeru or zgeru for complex numbers and dger or sger for
    # float numbers
    try:
        ger = get_blas_funcs("geru", [C])
    except RuntimeError:
        ger = get_blas_funcs("ger", [C])
    # set number of iters to 0
    iters = 0
    # check if need to swap rows
    while abs(C[i, j]) > tol and iters < max_iters:
        # add j to index and recompute C by SVM-formula
        index[i] = j
        tmp_row = C[i].copy()
        tmp_column = C[:, j].copy()
        tmp_column[i] -= 1.0
        alpha = -1.0 / C[i, j]
        ger(alpha, tmp_column, tmp_row, a=C, overwrite_a=1)
        iters += 1
        i, j = divmod(abs(C[:, :top_k_index]).argmax(), top_k_index)
    return index[:r].copy(), C.T
