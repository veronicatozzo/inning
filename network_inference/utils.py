import sys

from collections import namedtuple


convergence = namedtuple('convergence',
                         ('obj', 'rnorm', 'snorm', 'e_pri', 'e_dual'))


def check_data_dimensions(X, layers=2):
    if len(X) != layers:
        sys.error("The maximum number of layers is %d, X must be a list of"
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
