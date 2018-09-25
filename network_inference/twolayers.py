"""2-layers integrated network inference through FBS with mild differentiability assumptions."""
from __future__ import division

import numpy as np
import warnings
import operator
import sys

from scipy import linalg
from functools import partial
from itertools import product
from collections import Sequence

from sklearn.covariance import GraphicalLasso, empirical_covariance
from sklearn.utils.extmath import fast_logdet , squared_norm
from sklearn.utils.validation import check_array, check_random_state
from sklearn.utils import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import check_cv
from sklearn.datasets import make_sparse_spd_matrix

from .utils import check_data_dimensions, convergence,\
                                    update_rho, l1_od_norm
from .utils import _scalar_product, update_rho, convergence
from .utils import log_likelihood, BIC, EBIC, EBIC_m
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

    def __init__(self, L, eta=0.01, mu=0.01, tau=0.01, rho=1.,
                 tol=1e-4, rtol=1e-4, max_iter=100,
                 verbose=False, assume_centered=False, compute_objective=False,
                 random_state=None):
        self.L = L
        self.max_iter = max_iter
        self.verbose = verbose
        self.assume_centered = assume_centered
        self.eta = eta
        self.mu = mu
        self.rho = rho
        self.tol = tol
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
       # check_data_dimensions(X, layers=2)
        X = check_array(X, ensure_min_features=2,
                         ensure_min_samples=2, estimator=self)

        self.X_train = X
        if self.assume_centered:
            self.location_ = np.zeros((X.shape[0],  X.shape[1]))
        else:
            self.location_ = X.mean(0)

        emp_cov = empirical_covariance(
            		X, assume_centered=self.assume_centered)
        self.precision_,  self.hidden_, \
        self.observed_,  self.emp_cov, \
        self.n_iter_ = two_layers_fixed_links_GL(
                        emp_cov, self.L, eta=self.eta, mu=self.mu,
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

   
def par_max(emp_cov):
    A = np.copy(emp_cov)
    A.flat[::A.shape[0] + 1] = 0
    return np.max(np.abs(A))


def flgl_path(X_train, links=None, etas=[0.1], mus=[0.1],
              X_test=None, tol=1e-3, max_iter=200,
              update_rho=False, verbose=0, score='ebic',
              random_state=None):
    
    score_func = {'likelihood': log_likelihood,
                  'bic': BIC,
                  'ebic': partial(EBIC, n=X_test.shape[0]),
                  'ebicm': partial(EBIC_m, n=X_test.shape[0])}
    try:
        score_func = score_func[score]
    except KeyError:
        warnings.warn("The score type passed is not available, using log likelihood.")
        score_func = log_likelihood
    
    
    emp_cov = empirical_covariance(X_train)
    covariance_ = emp_cov.copy()
   
    covariances_ = list()
    precisions_ = list()
    hiddens_ =  list()
    scores_ = list()
    
    if X_test is not None:
        test_emp_cov = empirical_covariance(X_test)

    for eta in etas:
        for mu in mus:
            try:
                # Capture the errors, and move on
                cov_, prec_, hid_,_ = two_layers_fixed_links_GL(
                    emp_cov, links, mu, eta, max_iter=max_iter, 
                    random_state=random_state, return_n_iter=False)
                covariances_.append(cov_)
                precisions_.append(prec_)
                hiddens_.append(hid_)
                
                if X_test is not None:
                    this_score = score_func(test_emp_cov, prec_)
            except FloatingPointError:
                this_score = -np.inf
                covariances_.append(np.nan)
                precisions_.append(np.nan)
            if X_test is not None:
                if not np.isfinite(this_score):
                    this_score = -np.inf
                scores_.append(this_score)
            if verbose:
                if X_test is not None:
                    print('[graphical_lasso_path] eta: %.2e, mu: %.2e, score: %.2e'
                          % (eta, mu, this_score))
                else:
                    print('[graphical_lasso_path] eta: %.2e, mu: %.2e' % (eta, mu))
    if X_test is not None:
        return covariances_, precisions_, hiddens_, scores_
    return covariances_, precisions_, hiddens_


class TwoLayersFixedLinksCV():
    def __init__(self, links, etas=4, mus=4, rho=1., cv=None, tol=1e-4,
                 max_iter=100, n_jobs=1,
                 verbose=False, assume_centered=False,
                 update_rho=False, random_state=None):
        self.links = links
        self.etas = etas
        self.mus = mus
        self.rho = 1.
        self.cv = cv
        self.n_jobs = n_jobs
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.update_rho = update_rho
        self.assume_centered = assume_centered
        self.random_state = random_state

    def grid_scores(self):
        return self.grid_scores_

    def fit(self, X, y=None):
        """Fits the GraphLasso covariance model to X.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data from which to compute the covariance estimate
        y : (ignored)
        """
        # Covariance does not make sense for a single feature
        self.random_state = check_random_state(self.random_state)
        # check_data_dimensions(X, layers=2)
        X = check_array(X, ensure_min_features=2,
                         ensure_min_samples=2, estimator=self)

        self.X_train = X
        if self.assume_centered:
            self.location_ = np.zeros((X.shape[0],  X.shape[1]))
        else:
            self.location_ = X.mean(0)

        emp_cov = empirical_covariance(
                        X, assume_centered=self.assume_centered)
       
        X = check_array(X, ensure_min_features=2, estimator=self)
        cv = check_cv(self.cv, y, classifier=False)

        # List of (alpha, scores, covs)
        path = list()
        n_etas = self.etas
        inner_verbose = max(0, self.verbose - 1)

        if isinstance(n_etas, Sequence):
            etas = self.etas
        else:
            eta_1 = par_max(emp_cov)
            eta_0 = 1e-2 * eta_1
            etas = np.logspace(np.log10(eta_0), np.log10(eta_1),
                                 n_etas)[::-1]
        
        n_mus = self.mus
        inner_verbose = max(0, self.verbose - 1)

        if isinstance(n_mus, Sequence):
            mus = self.mus
        else:
            mu_1 = par_max(emp_cov) # not sure is the best strategy
            mu_0 = 1e-2 * mu_1
            mus = np.logspace(np.log10(mu_0), np.log10(mu_1),
                                 n_mus)[::-1]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)

        this_path = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )(delayed(flgl_path)(X[train], links=self.links, etas=etas, mus= mus,
                                 X_test=X[test], tol=self.tol,
                                        max_iter=int(.1 * self.max_iter),
                                        update_rho=self.update_rho,
                                        verbose=0, random_state=self.random_state)
              for train, test in cv.split(X, y))

        # Little danse to transform the list in what we need
        covs, precs, hidds, scores = zip(*this_path)
        covs = zip(*covs)
        precs = zip(*precs)
        hidds = zip(*hidds)
        scores = zip(*scores)
        combinations = list(product(etas, mus))
        path.extend(zip(combinations, scores, covs))
        path = sorted(path, key=operator.itemgetter(0), reverse=True)

        # Find the maximum (avoid using built in 'max' function to
        # have a fully-reproducible selection of the smallest alpha
        # in case of equality)
        best_score = -np.inf
        last_finite_idx = 0
        for index, (combination, scores, _) in enumerate(path):
            this_score = np.mean(scores)
            if this_score >= .1 / np.finfo(np.float64).eps:
                this_score = np.nan
            if np.isfinite(this_score):
                last_finite_idx = index
            if this_score >= best_score:
                best_score = this_score
                best_index = index

            
        path = list(zip(*path))
        grid_scores = list(path[1])
        parameters = list(path[0])
        # Finally, compute the score with alpha = 0
        best_eta, best_mu = combinations[best_index]
        self.eta_ = best_eta
        self.mu_ = best_mu
        self.cv_parameters_ = combinations

        # Finally fit the model with the selected alpha
        self.covariance_, self.precision_, self.hidden_, self.R_, self.n_iter_ = two_layers_fixed_links_GL(
            emp_cov, self.links, eta=best_eta, mu=best_mu, tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose, random_state=self.random_state, 
            compute_objective=True, return_n_iter=True)
        return self