"""2-layers integrated network inference through FBS with mild differentiability assumptions."""
from __future__ import division

import numpy as np
import warnings
import sys

# from functools import partial
from six.moves import map, range, zip
from sklearn.covariance import empirical_covariance
from sklearn.covariance import GraphLasso
from sklearn.utils.extmath import fast_logdet  # , squared_norm
from sklearn.utils.validation import check_array, check_random_state
# from sklearn.utils.validation import check_symmetric

from network_inference.utils import check_data_dimensions, convergence, update_rho
from network_inference.prox import prox_logdet, soft_thresholding_od, \
                                    soft_thresholding_sign


def log_likelihood(emp_cov, precision):
    """Gaussian log-likelihood without constant term."""
    return fast_logdet(precision) - np.sum(emp_cov * precision)


def objective(emp_cov, K1, K2, A1, A2, R, alpha1, alpha2, tau, rho) : # TODO
    """Objective function for time-varying graphical lasso."""
    obj = np.sum(-n * log_likelihood(emp_cov, precision)
                 for emp_cov, precision, n in zip(S, K, n_samples))
    obj += lamda * np.sum(map(l1_od_norm, K))
    obj += beta * np.sum(map(psi, K[1:] - K[:-1]))
    return obj


def _R_update(K1, K2, A1, A2, R, U1, U2, rho, tau, max_iter):
    gamma = 1e-7

    for iter_ in range(max_iter):
        R_old = R.copy()
        invK1 = np.linalg.pinv(K1)
        invK2 = np.linalg.pinv(K2)
        invK1R = invK1.dot(R)
        RinvK2 = R.dot(invK2)

        gradient = -2*rho*(invK1R.dot(K2 - A2 - R.T.dot(invK1R) + U2) +
                           (K1 - A1 - RinvK2.dot(R.T) + U1).dot(RinvK2))

        R = R - gamma*gradient
        R = soft_thresholding_sign(R, tau*gamma)
        if np.linalg.norm(R_old - R)/np.linalg.norm(R) < 1e-5:
            break
    else:
        warnings.warn("The update of the matrix R did not converge.")
    return R


def two_layers_graphical_lasso(
        data_list, alpha1=0.01, alpha2=0.01, tau=0.01, mode='admm', rho=1.,
        tol=1e-3, rtol=1e-5, max_iter=100, verbose=False, return_n_iter=True,
        return_history=False, compute_objective=False, compute_emp_cov=False,
        random_state=None, n1=None, n2=None):
    """Time-varying graphical lasso solver.

    Solves the following problem via ADMM:
        minimize  trace(S*X) - log det X + lambda*||X||_1

    where S is the empirical covariance of the data
    matrix D (training observations by features).

    Parameters
    ----------
    data_list : list of 2-dimensional matrices.
        Input matrices.
    alpha1 : float, optional
        Regularisation parameter.
    alpha2 : float, optional
        Augmented Lagrangian parameter.
    tau : float, optional
        Augmented Lagrangian parameter.
    mode: string, optional
        Method for the minimization of the problem.
    rho: float, optional
        Over-relaxation parameter (typically between 1.0 and 1.8)
    tol : float, optional
        Absolute tolerance for convergence.
    rtol : float, optional
        Relative tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    verbose: bool, optional
        Print info of the status.
    return_n_iter: bool, optional
        If True the number of iterations is returned.
    return_history : bool, optional
        Return the history of computed values.
    compute_objective: bool, optional
        Compute the objective at each iteration.
    compute_emp_cov: bool, optional
        Compute the empirical covariance of the input data.

    Returns
    -------
    precision1: numpy.array, 2-dimensional
        Network of the first layer.
    precision2: numpy.array, 2-dimensional
        Network of the second layer.
    relations: numpy.array, 2_dimensional
        The links between the two layers.
    empirical_covariance: list of 2-dimensional numpy.array
        The computed empirical covariance of the input data.
    n_iter: int
        Number of total iterations.
    history : list
        If return_history, then also a structure that contains the
        objective value, the primal and dual residual norms, and tolerances
        for the primal and dual residual norms at each iteration.
    """

    random_state = check_random_state(random_state)
    if compute_emp_cov:
        n1, n2 = [data_list[i].shape[0] for i in range(len(data_list))]
        emp_cov = [empirical_covariance(
            x, assume_centered=False) for x in data_list]
    else:
        emp_cov = data_list

    A1 = emp_cov[0].copy()
    A2 = emp_cov[1].copy()
    K1 = np.zeros_like(emp_cov[0])
    K2 = np.zeros_like(emp_cov[1])
    R = random_state.rand(K1.shape[0], K2.shape[0])
    U1 = K1.copy()
    U2 = K2.copy()

    checks = []
    for iteration_ in range(max_iter):
        A1_old = A1.copy()
        A2_old = A2.copy()

        # update K1
        M = A1 + np.linalg.multi_dot((R, np.linalg.pinv(K2), R.T)) + U1
        K1 = soft_thresholding_od(M, lamda=alpha1 / rho)
        print(K1)
        # update K2
        M = A2 + np.linalg.multi_dot((R.T, np.linalg.pinv(K1), R)) + U2
        K2 = soft_thresholding_od(M, lamda=alpha2 / rho)

        # update A1
        M = K1 - np.linalg.multi_dot((R, np.linalg.pinv(K2), R.T)) + U1
        M += M.T
        M /= 2.
        A1 = prox_logdet(data_list[0] - rho * M, lamda=1. / (rho*n1))

        # update A2
        M = K2 - np.linalg.multi_dot((R.T, np.linalg.pinv(K1), R)) + U2
        M += M.T
        M /= 2.
        A2 = prox_logdet(data_list[1] - rho * M, lamda=1. / (rho*n2))

        # update R
        R = _R_update(K1, K2, A1, A2, R, U1, U2, rho, tau,
                      max_iter=500)

        # update residuals
        RK2R = np.linalg.multi_dot((R, np.linalg.pinv(K2), R.T))
        RK1R = np.linalg.multi_dot((R.T, np.linalg.pinv(K1), R))
        U1 += A1 - K1 + RK2R
        U2 += A2 - K2 + RK1R

        # diagnostics, reporting, termination checks
        obj = objective(emp_cov, K1, K2, A1, A2, R, alpha1, alpha1, tau, rho) \
            if compute_objective else np.nan
        rnorm = np.sqrt(np.linalg.norm(A1 - K1 + RK2R)**2 +
                        np.linalg.norm(A2 - K2 + RK1R)**2)
        snorm = rho * np.sqrt(np.linalg.norm(A1 - A1_old)**2 +
                              np.linalg.norm(A2 - A2_old)**2)
        check = convergence(
            obj=obj, rnorm=rnorm, snorm=snorm,
            e_pri=(np.sqrt(A1.size + A2.size) * tol + rtol *
                   max(np.sqrt(np.linalg.norm(A1)**2 + np.linalg.norm(A2)**2 +
                               np.linalg.norm(U1)**2 + np.linalg.norm(U2)**2),
                       np.sqrt(np.linalg.norm(K1 - RK2R)**2 +
                               np.linalg.norm(K2 - RK1R)**2))),
            e_dual=(np.sqrt(A1.size + A2.size) * tol + rtol * rho *
                    np.sqrt(np.linalg.norm(U1)**2 + np.linalg.norm(U2)**2))
        )

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % check)

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break
        rho_new = update_rho(rho, rnorm, snorm, iteration=iteration_)
        # scaled dual variables should be also rescaled
        U1 *= rho / rho_new
        U2 *= rho / rho_new
        rho = rho_new
    else:
        warnings.warn("Objective did not converge.")

    return_list = [K1, K2, R, K1 - RK2R, K2 - RK1R, emp_cov]
    if return_n_iter:
        return_list.append(iteration_)
    if return_history:
        return_list.append(checks)
    return return_list


class TwoLayersGraphicalLasso(GraphLasso):
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
        super(GraphLasso, self).__init__(
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
                sys.error("The number of features of the %d layer dataset given"
                          "as test is not consistent with the train. Train: %d "
                          "features; Test: %d features" %
                          (i+1, self.X_train[i].shape[1], X_test[i].shape[1]))
                sys.exit(0)
