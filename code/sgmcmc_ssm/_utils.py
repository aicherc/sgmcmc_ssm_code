"""

Utilities functions

"""
import numpy as np
import pandas as pd
import logging
import scipy
import scipy.linalg.lapack
from copy import deepcopy

logger = logging.getLogger(name=__name__)


# Random Categorical
def random_categorical(pvals, size=None):
    out = np.random.multinomial(n=1, pvals=pvals, size=size).dot(
            np.arange(len(pvals)))
    return int(out)

# Fixed Wishart
def array_wishart_rvs(df, scale, **kwargs):
    """ Wrapper around scipy.stats.wishart to always return a np.array """
    if np.size(scale) == 1:
        return np.array([[
            scipy.stats.wishart(df=df, scale=scale, **kwargs).rvs()
            ]])
    else:
        return scipy.stats.wishart(df=df, scale=scale, **kwargs).rvs()

def array_invwishart_rvs(df, scale, **kwargs):
    """ Wrapper around scipy.stats.invwishart to always return a np.array """
    if np.size(scale) == 1:
        return np.array([[
            scipy.stats.invwishart(df=df, scale=scale, **kwargs).rvs()
            ]])
    else:
        return scipy.stats.invwishart(df=df, scale=scale, **kwargs).rvs()

def array_invgamma_rvs(shape, scale, **kwargs):
    """ Wrapper around scipy.stats.wishart to always return a np.array """
    if np.size(scale) == 1:
        return np.array([
            scipy.stats.invgamma(a=shape, scale=scale, **kwargs).rvs()
            ])
    else:
        return scipy.stats.invgamma(a=shape, scale=scale, **kwargs).rvs()

# Matrix Normal LogPDF
def matrix_normal_logpdf(X, mean, Lrowprec, Lcolprec):
    """ Numerical stable matrix normal logpdf
    (when cholesky of precision matrices are known)

    Args:
        X (n by m ndarray): random variable instance
        mean (n by m ndarray): mean
        Lrowprec (n by n ndarray): chol of pos def row covariance (i.e. U^{-1})
        Lcolprec (m by m ndarray): chol of pos def col covariance (i.e. V^{-1})
    Returns:
        logpdf = (-1/2*tr(V^{-1}(X-M)U^{-1}(X-M)) - nm/2*log(2pi) +
            m/2*log|U^{-1}| + n/2*log|V^{-1}|)
    """
    n, m = np.shape(X)
    logpdf = -0.5*n*m*np.log(2*np.pi)
    logpdf += -0.5*np.sum(np.dot(Lrowprec.T, np.dot(X-mean, Lcolprec))**2)
    logpdf += m*np.sum(np.log(np.diag(Lrowprec)))
    logpdf += n*np.sum(np.log(np.diag(Lcolprec)))
    return logpdf

def normal_logpdf(X, mean, Lprec):
    """ normal logpdf

    Returns:
        logpdf= -1/2*(X-mean)'*prec*(X-mean) + sum(log(Lprec)) - m/2*log(2pi)
    """
    m = np.size(X)
    delta = np.dot(Lprec, X-mean)
    logpdf = -0.5*m*np.log(2*np.pi)
    logpdf += -0.5*np.dot(delta, delta)
    logpdf += np.sum(np.log(np.diag(Lprec)))
    return logpdf

# Positive Definite Matrix Inverse
def pos_def_mat_inv(mat):
    """ Return inverse(mat)

    Uses LAPACK.DPOTRF and LAPACK.DPOTRI to compute the inverse
        using cholesky decomposition

    See also: https://stackoverflow.com/questions/40703042/more-efficient-way-to-invert-a-matrix-knowing-it-is-symmetric-and-positive-semi

    """
    if np.isscalar(mat):
        return mat ** -1
    zz, info = scipy.linalg.lapack.dpotrf(mat, False, False)
    if info != 0:
        raise RuntimeError("Error in Cholesky Decomposition")
    inv_M, info = scipy.linalg.lapack.dpotri(zz)
    if info != 0:
        raise RuntimeError("Error in Cholesky Inverse")
    inv = np.triu(inv_M) + np.triu(inv_M, k=1).T
    return inv

def pos_def_log_det(mat):
    """ Return log_det(mat)

    Uses LAPACK.DPOTRF to compute the cholesky decomposition

    """
    if np.isscalar(mat):
        return np.log(mat)
    zz, info = scipy.linalg.lapack.dpotrf(mat, False, False)
    if info != 0:
        raise RuntimeError("Error in Cholesky Decomposition")
    logdet_mat = np.sum(np.log(np.diag(zz)))*2.0
    return logdet_mat

def lower_tri_mat_inv(lower_tri_mat):
    """ Return inverse(lower_tri_mat)

    Uses LAPACK.DTRTRI
    """
    if np.isscalar(lower_tri_mat):
        return lower_tri_mat ** -1
    lower_tri_inv, info = scipy.linalg.lapack.dtrtri(lower_tri_mat, True, False)
    if info != 0:
        raise RuntimeError("Error in Lower Triangular Inverse")
    return lower_tri_inv

# Symmetrize Matrices
def sym(mat):
    if np.isscalar(mat):
        return mat
    else:
        return (mat + np.swapaxes(mat, -1, -2))/2.0

# Stability of VAR(p)
def varp_stability_projection(A, eigenvalue_cutoff=0.9999, logger=logger):
    """ Threshold VAR(p) A to have stable eigenvalues """
    m, mp = np.shape(A)
    p = mp//m
    A_stable = A
    F = np.concatenate([A, np.eye(N=m*(p-1), M = m*p)])
    lambduhs = np.linalg.eig(F)[0]
    largest_eigenvalue = np.max(np.abs(lambduhs))
    if largest_eigenvalue > eigenvalue_cutoff:
        logger.info("Thresholding Largest Eigenval F: {0} > {1}".format(
            largest_eigenvalue, eigenvalue_cutoff))
        for ii in range(p):
            A_stable[:, m*ii:m*(ii+1)] *= \
                    (eigenvalue_cutoff/largest_eigenvalue)**(ii+1)
    return A_stable

# Asymptotic precision for VAR(1)
def var_stationary_precision(Qinv, A, num_iters=50):
    """ Approximate the stationary precision matrix of a VAR """
    precision = Qinv
    QinvA = np.dot(Qinv, A)
    AtQinvA = np.dot(A.T, QinvA)
    for ii in range(num_iters):
        precision = Qinv - \
                np.dot(QinvA, np.linalg.solve(precision + AtQinvA, QinvA.T))
    return precision

