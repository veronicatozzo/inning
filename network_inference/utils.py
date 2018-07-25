import sys

from collections import namedtuple


convergence = namedtuple('convergence',
                         ('obj', 'iter_diff'))


def check_data_dimensions(X, layers=2):
    if len(X) != layers:
        sys.error("The maximum number of layers is %d, X must be a list of"
                  " length %d of data matrices." % (layers, layers))
        sys.exit(0)
