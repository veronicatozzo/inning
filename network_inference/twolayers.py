"""2-layers integrated network inference through FBS with mild differentiability assumptions."""
from __future__ import division

import numpy as np
import warnings
import sys

from scipy import linalg
from functools import partial

from sklearn.covariance import GraphicalLasso, empirical_covariance
from sklearn.utils.extmath import fast_logdet , squared_norm
from sklearn.utils.validation import check_array, check_random_state

from .utils import check_data_dimensions, convergence, 
                                    update_rho, l1_od_norm
from .utils import _scalar_product, update_rho, convergence
from .utils import log_likelihood
from .prox import prox_logdet, soft_thresholding_od, \
                                    soft_thresholding_sign
from .datasets import is_pos_def, is_pos_semi_def
from ._two_layers_optimization import two_layers_fixed_links_GL, \
                                      two_layers_graphical_lasso


class TwoLayersGraphicalLasso(GraphicalLasso):
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

    tau : positive float, default 0.01
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

    def __init__(self, alpha1=0.01, alpha2=0.01, tau=0.01, mode='admm', rho=1.,
                 tol=1e-4, rtol=1e-4, max_iter=100,
                 verbose=False, assume_centered=False, compute_objective=False,
                 random_state=None):
        super(GraphicalLasso, self).__init__(
            tol=tol, max_iter=max_iter, verbose=verbose,
            assume_centered=assume_centered, mode=mode)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.tau = tau
        self.rho = rho
        self.rtol = rtol
        self.compute_objective = compute_objective
        self.random_state = random_state
        self.covariance1_ = None
        self.covariance2_ = None
        self.precision1_ = None
        self.precision2_ = None
        self.R_ = None
        self.observed1_ = None
        self.observed2_ = None
        self.emp_cov = None
        self.n_iter_ = None

    def get_precision(self):
        """Getter for the observed global precision matrix.

        Returns
        -------
        precision_ : array-like,
            The precision matrix associated to the current covariance object.

        """
        return self.precision1_, self.precision2_

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
        return self.observed1_, self.observed2_

    def fit(self, X, y=None):
        """Fit the GraphLasso model to X.

        Parameters
        ----------
        X : lenght-2 list of array-like of shape (n_samples1, n_features1) and
            (n_samples2, n_features2)
            Data from which to compute the covariance estimates.
        y : (ignored)
        """
        self.random_state = check_random_state(self.random_state)
        check_data_dimensions(X, layers=2)
        X = [check_array(x, ensure_min_features=2,
                         ensure_min_samples=2, estimator=self) for x in X]

        self.X_train = X
        if self.assume_centered:
            self.location1_ = np.zeros((X[0].shape[0],  X[0].shape[1]))
            self.location2_ = np.zeros((X[1].shape[0],  X[1].shape[1]))
        else:
            self.location1_ = X[0].mean(1).reshape(X[0].shape[0], # TODO non sono sicura che sia la direzione giusta
                                                   X[0].shape[1])
            self.location2_ = X[1].mean(1).reshape(X[1].shape[0],
                                                   X[1].shape[1])

        emp_cov = [empirical_covariance(
            x, assume_centered=self.assume_centered) for x in X]
        self.precision1_,  self.precision2_, self.R_,
        self.observed1_, self.observed2_, self.emp_cov,
        self.n_iter_ = \
            two_layers_graphical_lasso(
                emp_cov, alpha1=self.alpha1, alpha2=self.alpha2,
                tau=self.tau, mode=self.mode, rho=self.rho,
                tol=self.tol, rtol=self.rtol,
                max_iter=self.max_iter, verbose=self.verbose,
                return_n_iter=True, return_history=False,
                compute_objective=self.compute_objective,
                n1=X[0].shape[0], n2=X[1].shape[0])
        return self

    # TODO: potrebbe essere che va cambiato con funzioni piu' fighe
    def score(self, X_test, y=None):
        # TODO: look at all the differences in the likelihoods
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

    # def error_norm(self, comp_cov, norm='frobenius', scaling=True,
    #                squared=True):
    #     """Compute the Mean Squared Error between two covariance estimators.
    #     (In the sense of the Frobenius norm).
    #
    #     Parameters
    #     ----------
    #     comp_cov : array-like, shape = [n_features, n_features]
    #         The covariance to compare with.
    #
    #     norm : str
    #         The type of norm used to compute the error. Available error types:
    #         - 'frobenius' (default): sqrt(tr(A^t.A))
    #         - 'spectral': sqrt(max(eigenvalues(A^t.A))
    #         where A is the error ``(comp_cov - self.covariance_)``.
    #
    #     scaling : bool
    #         If True (default), the squared error norm is divided by n_features.
    #         If False, the squared error norm is not rescaled.
    #
    #     squared : bool
    #         Whether to compute the squared error norm or the error norm.
    #         If True (default), the squared error norm is returned.
    #         If False, the error norm is returned.
    #
    #     Returns
    #     -------
    #     The Mean Squared Error (in the sense of the Frobenius norm) between
    #     `self` and `comp_cov` covariance estimators.
    #
    #     """
    #     return error_norm_time(self.covariance_, comp_cov, norm=norm,
    #                            scaling=scaling, squared=squared)

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
                sys.error("The number of features of the %d layer dataset "
                          "given as test is not consistent with the train. "
                          "Train: %d features; Test: %d features" %
                          (i+1, self.X_train[i].shape[1], X_test[i].shape[1]))
                sys.exit(0)


class TwoLayersFixedLinks(GraphicalLasso):
    """Sparse inverse covariance estimation with an l1-penalized estimator.

    Parameters
    ----------
    L : array-like, shape (n_latent, n_features)
        The links between the observed layer and the hidden one.

    eta : positive float, default 0.01
            Regularization parameter for precision matrix.
            The higher eta, the more regularization, the sparser the inverse
            covariance.

    mu : positive float, default 0.01
        Regularization parameter for hidden precision matrix.
        The higher mu, the more regularization, the sparser the inverse
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
    covariance_ : array-like, shape (n_features, n_features)
        Estimated covariance matrix of the first layer.

    precision_ : array-like, shape (n_features, n_features)
        Estimated precision matrix of the first layer.

    hidden_ : array-like, shape (n_latent, n_latent)
        Estimated pseudo inverse matrix of the first layer.

    observed_: array-like, shape (n_features, n_features)
        The estimated observed precision matrix.

    n_iter_ : int
        Number of iterations run.

    """

    def __init__(self, K, eta=0.01, mu=0.01, tau=0.01, rho=1.,
                 tol=1e-4, rtol=1e-4, max_iter=100,
                 verbose=False, assume_centered=False, compute_objective=False,
                 random_state=None):
        super(GraphLasso, self).__init__(
            tol=tol, max_iter=max_iter, verbose=verbose,
            assume_centered=assume_centered, mode=mode)
        self.eta = eta
        self.mu = mu
        self.rho = rho
        self.rtol = rtol
        self.compute_objective = compute_objective
        self.random_state = random_state
        self.covariance_ = None
        self.precision_ = None
        self.hidden_ = None
        self.observed_ = None
        self.emp_cov = None
        self.n_iter_ = None

    def get_precision(self):
        """Getter for the observed global precision matrix.

        Returns
        -------
        precision_ : array-like,
            The precision matrix associated to the current covariance object.

        """
        return self.observed_


    def fit(self, X, y=None):
        """Fit the GraphLasso model to X.

        Parameters
        ----------
        X : array-like shape (n_samples, n_features) 
            Data from which to compute the covariance estimate.
        y : (ignored)
        """
        self.random_state = check_random_state(self.random_state)
        check_data_dimensions(X, layers=2)
        X = [check_array(x, ensure_min_features=2,
                         ensure_min_samples=2, estimator=self) for x in X]

        self.X_train = X
        if self.assume_centered:
            self.location_ = np.zeros((X.shape[0],  X.shape[1]))
        else:
            self.location_ = X.mean(1).reshape(X.shape[0], 
                                               X.shape[1])

        emp_cov = [empirical_covariance(
            x, assume_centered=self.assume_centered) for x in X]
        self.precision_,  self.hidden_, \
        self.observed_,  self.emp_cov, \
        self.n_iter_ = two_layers_fixed_links_GL(
                        emp_cov, L, eta=self.eta, mu=self.mu,
                        rho=self.rho, tol=self.tol, rtol=self.rtol,
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

        if self.covariance_ is None :
            sys.error("The estimator is not fit on training data.")
            sys.exit(0)

        check_data_dimensions(X_test, layers=2)
        # compute empirical covariance of the test set
        test_cov = empirical_covariance(x - self.location_, assumed_centered=True)

        res = log_likelihood(test_cov, self.get_precision())
        return res

   