import sys
import warnings 

import numpy as np

from collections import namedtuple


convergence = namedtuple('convergence',
                         ('obj', 'rnorm', 'snorm', 'e_pri', 'e_dual'))


def log_likelihood(emp_cov, precision):
    """Gaussian log-likelihood without constant term."""
    return fast_logdet(precision) - np.sum(emp_cov * precision)


def BIC(emp_cov, precision):
    return log_likelihood(emp_cov, precision) - (np.sum(precision != 0)-precision.shape[0])


def EBIC(emp_cov, precision, n=100, epsilon=0.5):
    likelihood = log_likelihood(emp_cov, precision)
    of_nonzero = np.sum(precision != 0) - precision.shape[0]
    penalty = np.log(n)/n*of_nonzero + 4*epsilon* np.log(precision.shape[0])/n * of_nonzero
    return likelihood - penalty


def EBIC_m(emp_cov, precision, n=100, epsilon=0.5):
    likelihood = log_likelihood(emp_cov, precision)
    of_nonzero = np.sum(precision != 0) - precision.shape[0]
    p = precision.shape[0]
    penalty = np.log(n)/n*of_nonzero + 4*epsilon* np.log(p(p-1)/2)/n * of_nonzero
    return likelihood - penalty


def l1_od_norm(precision):
    """L1 norm off-diagonal."""
    return np.abs(precision).sum() - np.abs(np.diag(precision)).sum()


def check_data_dimensions(X, layers=2):
    if len(X) != layers:
        warnings.warn("The maximum number of layers is %d, X must be a list of"
                  " length %d of data matrices." % (layers, layers))
        sys.exit(0)


def update_rho(rho, rnorm, snorm, iteration=None, mu=10, tau_inc=2, tau_dec=2):
    """See Boyd pag 20-21 for details.

    Parameters
    ----------
    rho : float
    """
    if rnorm > mu * snorm:
        return tau_inc * rho
    elif snorm > mu * rnorm:
        return rho / tau_dec
    return rho


def _scalar_product(x, y):
    return (x * y).sum()


def l1_od_norm(precision):
    """L1 norm off-diagonal."""
    return np.abs(precision).sum() - np.abs(np.diag(precision)).sum()
