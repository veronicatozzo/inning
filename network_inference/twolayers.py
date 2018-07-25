"""2-layers integrated network inference through FBS with mild differentiability assumptions."""
from __future__ import division

import numpy as np
import warnings
import sys

from functools import partial
from six.moves import map, range, zip
from sklearn.covariance import empirical_covariance
from sklearn.covariance import GraphLasso
from sklearn.utils.extmath import fast_logdet, squared_norm
from sklearn.utils.validation import check_symmetric

from network_inference.utils import check_data_dimensions, convergence

from sklearn.utils.validation import check_array


def _gradient(x, S, n_samples):
    return (S - np.array(map(np.linalg.inv, x))) * n_samples[:, None, None]


def _J(K, alpha1, alpha2, mu, gamma, _lambda, S, n_samples):
    grad_ = _gradient(K, S, n_samples)
    prox_ = prox_FL(x - gamma * grad_, beta * gamma, lamda * gamma)
    return x + alpha * (prox_ - x)


def choose_alpha(_lambda_old, K, S, n_samples,
                 alpha1, alpha2, mu, _lambda, gamma, theta=.99, max_iter=1000):
    """Choose alpha for backtracking.

    References
    ----------
    Salzo S. (2017). https://doi.org/10.1137/16M1073741
    """
    eps = .5
    partial_J = partial(_J, x, beta=beta, lamda=lamda,
                        gamma=gamma, S=S, n_samples=n_samples)
    partial_f = partial(_f, n_samples=n_samples, S=S)
    gradient_ = _gradient(x, S, n_samples)
    for i in range(max_iter):
        iter_diff = partial_J(alpha=alpha) - x
        obj_diff = partial_f(K=partial_J(alpha=alpha)) - partial_f(K=x)
        if obj_diff - _scalar_product_3d(iter_diff, gradient_) <= theta / (gamma * alpha) * squared_norm(iter_diff) + 1e-16:
            return alpha

        alpha *= eps
    return alpha


def log_likelihood(emp_cov, precision):
    """Gaussian log-likelihood without constant term."""
    return fast_logdet(precision) - np.sum(emp_cov * precision)


def _f(n_samples, S, K):
    return np.sum(-n * log_likelihood(emp_cov, precision)
                  for emp_cov, precision, n in zip(S, K, n_samples))


def objective(n_samples, S, K, lamda, beta, psi):
    """Objective function for time-varying graphical lasso."""
    obj = np.sum(-n * log_likelihood(emp_cov, precision)
                 for emp_cov, precision, n in zip(S, K, n_samples))
    obj += lamda * np.sum(map(l1_od_norm, K))
    obj += beta * np.sum(map(psi, K[1:] - K[:-1]))
    return obj


def two_layers_graphical_lasso(
        data_list, alpha1=0.01, alpha2=0.01, mu=0.01, mode='fbs',
        tol=1e-3, rtol=1e-5, max_iter=100, verbose=False, return_n_iter=True,
        return_history=False, compute_objective=False, compute_emp_cov=False):
    """Time-varying graphical lasso solver.

    Solves the following problem via ADMM:
        minimize  trace(S*X) - log det X + lambda*||X||_1

    where S is the empirical covariance of the data
    matrix D (training observations by features).

    Parameters
    ----------
    data_list : list of 2-dimensional matrices.
        Input matrices.
    lamda : float, optional
        Regularisation parameter.
    rho : float, optional
        Augmented Lagrangian parameter.
    alpha : float, optional
        Over-relaxation parameter (typically between 1.0 and 1.8).
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Absolute tolerance for convergence.
    rtol : float, optional
        Relative tolerance for convergence.
    return_history : bool, optional
        Return the history of computed values.

    Returns
    -------
    X : numpy.array, 2-dimensional
        Solution to the problem.
    history : list
        If return_history, then also a structure that contains the
        objective value, the primal and dual residual norms, and tolerances
        for the primal and dual residual norms at each iteration.
        self.precision1_, self.covariance1_,
        self.precision2_, self.covariance2_,
        self.precision_, self.covariance_,
        self.n_iter_
    """
    if compute_emp_cov:
        S = list(map(empirical_covariance, data_list))
    else:
        S = [check_symmetric(c, raise_exception=True) for c in data_list]

    n_samples = np.array([s.shape[0] for s in data_list])
    K = [np.zeros_like(s) for s in S]

    checks = []
    _lambda = 1
    Kold = K.copy()
    for _ in range(max_iter):
        for k in range(S.shape[0]):
            K[k].flat[::K.shape[1] + 1] = 1
        _lambda_old = _lambda

        # choose a gamma
        gamma = .75

        # total variation
        Y = _J(K, alpha1, alpha2, mu, gamma, 1, S, n_samples)
        _lambda = choose_alpha(_lambda_old, K, S, n_samples,
                               alpha1, alpha2, mu, _lambda, gamma)
        _lambda = 1
        K = Kold + _lambda * (Y - Kold)

        check = convergence(
            obj=objective(n_samples, S, K, alpha1, alpha2, mu),
            iter_diff=np.sum([np.linalg.norm(k - kold) for k, kold
                             in zip(K, Kold)]),
        )

        if verbose:
            print("obj: %.4f, iter_diff: %.4f" % check)

        checks.append(check)
        # if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
        #     break
        if check.iter_diff <= rtol:
            break
        Kold = K.copy()
    else:
        warnings.warn("Objective did not converge.")

    if return_history:
        return K, S, checks
    return K, S


class TwoLayersIntegratedGraphicalLasso(GraphLasso):
    """Sparse inverse covariance estimation with an l1-penalized estimator.

    Parameters
    ----------
    alpha1 : positive float, default 0.01
        Regularization parameter for precision matrix on the first layer.
        The higher alpha1, the more regularization, the sparser the inverse
        covariance.

    alpha2 : positive float, default 0.01
        Regularization parameter for precision matrix on the second layer.
        The higher alpha2, the more regularization, the sparser the inverse
        covariance.

    mu : positive float, default 0.01
        Regularization parameter for links between the first and the second
        layer. The higher mu, the more regularization, the sparser the inverse
        covariance.

    tol : positive float, default 1e-4
        Absolute tolerance to declare convergence.

    rtol : positive float, default 1e-4
        Relative tolerance to declare convergence.

    max_iter : integer, default 100
        The maximum number of iterations.

    verbose : boolean, default False
        If verbose is True, the objective function, rnorm and snorm are
        printed at each iteration.

    assume_centered : boolean, default False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.

    compute_objective : boolean, default True
        Choose if compute the objective function during iterations
        (only useful if `verbose=True`).

    mode : {'fbs'}, default 'fbs'
        Minimisation algorithm. At the moment, only 'fbs' is available,
        so this is ignored.

    Attributes
    ----------
    covariance1_ : array-like, shape (n_features1, n_features1)
        Estimated covariance matrix of the first layer.

    precision1_ : array-like, shape (n_features1, n_features1)
        Estimated pseudo inverse matrix of the first layer.

    covariance2_ : array-like, shape (n_features2, n_features2)
        Estimated covariance matrix of the second layer.

    precision2_ : array-like, shape (n_features2, n_features2)
        Estimated pseudo inverse matrix of the second layer.

    connections_: array_like, shape (n_features1, n_features2)
        Estimated part of the global precision matrix.

    covariance_ : array-like,
                   shape (n_features1 + n_features2, n_features1 + n_features2)
        Estimated covariance matrix of the global system.

    precision_ : array-like,
                   shape (n_features1 + n_features2, n_features1 + n_features2)
        Estimated pseudo inverse matrix of the global system.

    n_iter_ : int
        Number of iterations run.

    """

    def __init__(self, alpha1=0.01, alpha2=0.01, mu=0.01, mode='admm', rho=1.,
                 tol=1e-4, rtol=1e-4, max_iter=100,
                 verbose=False, assume_centered=False, compute_objective=True):
        super(GraphLasso, self).__init__(
            tol=tol, max_iter=max_iter, verbose=verbose,
            assume_centered=assume_centered, mode=mode)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.mu = mu
        self.rtol = rtol
        self.compute_objective = compute_objective
        self.covariance1_ = None
        self.covariance2_ = None

    def get_precision(self):
        """Getter for the observed global precision matrix.

        Returns
        -------
        precision_ : array-like,
            The precision matrix associated to the current covariance object.

        """
        return self.get_precision()

    def get_observed_layers_precisions(self):
        """Getter for the observed global precision matrix.

        Returns
        -------
        precision1_ : array-like,
            The precision matrix associated to the current covariance object on
            the first layer.

        precision2_ : array-like,
            The precision matrix associated to the current covariance object on
            the second layer.

        """
        return self.precision1_, self.precision2_

    def fit(self, X, y=None):
        """Fit the GraphLasso model to X.

        Parameters
        ----------
        X : lenght-2 list of array-like of shape (n_samples1, n_features1) and
            (n_samples2, n_features2)
            Data from which to compute the covariance estimates.
        y : (ignored)
        """

        check_data_dimensions(X, layers=2)
        X = [check_array(x, ensure_min_features=2,
                         ensure_min_samples=2, estimator=self) for x in X]

        self.X_train = X
        if self.assume_centered:
            self.location1_ = np.zeros((X[0].shape[0],  X[0].shape[1]))
            self.location2_ = np.zeros((X[1].shape[0],  X[1].shape[1]))
        else:
            self.location1_ = X[0].mean(1).reshape(X[0].shape[0], X[0].shape[1])
            self.location2_ = X[1].mean(1).reshape(X[1].shape[0], X[1].shape[1])

        emp_cov = [empirical_covariance(
            x, assume_centered=self.assume_centered) for x in X]
        self.precision1_, self.covariance1_,
        self.precision2_, self.covariance2_,
        self.precision_, self.covariance_,
        self.n_iter_ = \
            two_layers_graphical_lasso(
                emp_cov, alpha1=self.alpha1, alpha2=self.alpha2,
                mu=self.mu, mode=self.mode,
                tol=self.tol, rtol=self.rtol,
                max_iter=self.max_iter, verbose=self.verbose,
                return_n_iter=True, return_history=False,
                compute_objective=self.compute_objective)
        return self

    def score(self, X_test, y=None):
        """Computes the log-likelihood of a Gaussian data set with
        `self.covariance_` as an estimator of its covariance matrix.

        Parameters
        ----------
        X_test : lenght-2 list of array-like of shape (n_samples1, n_features1)
                 and (n_samples2, n_features2)
            Test data of which we compute the likelihood, where n_samples is
            the number of samples and n_features is the number of features.
            X_test is assumed to be drawn from the same distribution than
            the data used in fit (including centering). The number of features
            must correspond.

        y : not used, present for API consistence purpose.

        Returns
        -------
        res : float
            The likelihood of the data set with `self.covariance1_` and
            `self.covariance1_`as an estimator of its covariance matrix.

        """

        if self.covariance1_ is None and self.covariance2_ is None:
            sys.error("The estimator is not fit on training data.")
            sys.exit(0)

        check_data_dimensions(X_test, layers=2)
        self._check_consistency_with_train(X_test)
        # compute empirical covariance of the test set
        test_cov = np.array([empirical_covariance(
                   x - loc, assume_centered=True)
                   for x, loc
                   in zip(X_test, [self.location1_, self.location2_])])

        res = sum(log_likelihood(S, K) for S, K in zip(
            test_cov, self.get_observed_layers_precision()))

        return res

    def error_norm(self, comp_cov, norm='frobenius', scaling=True,
                   squared=True):
        """Compute the Mean Squared Error between two covariance estimators.
        (In the sense of the Frobenius norm).

        Parameters
        ----------
        comp_cov : array-like, shape = [n_features, n_features]
            The covariance to compare with.

        norm : str
            The type of norm used to compute the error. Available error types:
            - 'frobenius' (default): sqrt(tr(A^t.A))
            - 'spectral': sqrt(max(eigenvalues(A^t.A))
            where A is the error ``(comp_cov - self.covariance_)``.

        scaling : bool
            If True (default), the squared error norm is divided by n_features.
            If False, the squared error norm is not rescaled.

        squared : bool
            Whether to compute the squared error norm or the error norm.
            If True (default), the squared error norm is returned.
            If False, the error norm is returned.

        Returns
        -------
        The Mean Squared Error (in the sense of the Frobenius norm) between
        `self` and `comp_cov` covariance estimators.

        """
        return error_norm_time(self.covariance_, comp_cov, norm=norm,
                               scaling=scaling, squared=squared)

    # def mahalanobis(self, observations):
    #     """Computes the squared Mahalanobis distances of given observations.
    #
    #     Parameters
    #     ----------
    #     observations : array-like, shape = [n_observations, n_features]
    #         The observations, the Mahalanobis distances of the which we
    #         compute. Observations are assumed to be drawn from the same
    #         distribution than the data used in fit.
    #
    #     Returns
    #     -------
    #     mahalanobis_distance : array, shape = [n_observations,]
    #         Squared Mahalanobis distances of the observations.
    #
    #     """
    #     precision1, precision2 = self.get_observed_layers_precision()
    #     # compute mahalanobis distances
    #     sum_ = 0.
    #     for obs, loc in zip(observations, self.location_):
    #         centered_obs = observations - self.location_
    #         sum_ += np.sum(
    #             np.dot(centered_obs, precision1) * centered_obs, 1)
    #
    #     mahalanobis_dist = sum_ / len(observations)
    #     return mahalanobis_dist

    def _check_consistency_with_train(self, X_test):

        for i in len(X_test):
            if X_test[i].shape[1] != self.X_train[i].shape[1]:
                sys.error("The number of features of the %d layer dataset given"
                          "as test is not consistent with the train. Train: %d "
                          "features; Test: %d features" %
                          (i+1, self.X_train[i].shape[1], X_test[i].shape[1]))
                sys.exit(0)
