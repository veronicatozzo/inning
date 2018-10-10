import warnings
import operator
import numpy as np

from functools import partial
from itertools import product
from scipy.linalg import pinvh
from collections import Sequence

from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import squared_norm, fast_logdet
from sklearn.covariance import empirical_covariance
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.preprocessing import normalize
from sklearn.model_selection import check_cv
from sklearn.utils import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

from regain.covariance.graph_lasso_ import graph_lasso

from network_inference.datasets import is_pos_def
from network_inference.utils import log_likelihood, BIC, EBIC, EBIC_m
from network_inference.utils import convergence, l1_od_norm



def par_max(emp_cov):
    A = np.copy(emp_cov)
    A.flat[::A.shape[0] + 1] = 0
    return np.max(np.abs(A))


def _penalized_nll(K, S=None, regularizer=None):
    res = - fast_logdet(K) + np.sum(K*S)
    res += np.linalg.norm(regularizer*K, 1)
    return res


def flgl_path(X_train, mask=None, lambdas=[0.1], mus=[0.1], 
              X_test=None, random_search=True,  tol=1e-3, max_iter=200,
              update_rho=False, verbose=0, score='ebic',
              random_state=None, save_all=False):
    
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
    observeds_ =  list()
    scores_ = list()
    
    if X_test is not None:
        test_emp_cov = empirical_covariance(X_test)

    if random_search:
        couples = zip(lambdas, mus)
    else:
        couples = product(lambdas, mus)
    
    for i, params in enumerate(couples):
        try:
            # Capture the errors, and move on
            prec_, cov_, _ = EM(
                emp_cov, mask, params[0], params[1], max_iter=max_iter, 
                return_n_iter=False)
            if save_all:
                covariances_.append(cov_)
                precisions_.append(prec_)
            
            h = mask.shape[1]
            observed = prec_[h:,h:] - prec_[h:, :h].dot(pinvh(prec_[:h, :h])).dot(prec_[:h, h:])
            if X_test is not None:
                this_score = score_func(test_emp_cov, observed)
                print("Iter: %d, Lambda %.4f, mu %.4f, score %.4f" %(i, params[0], params[1], this_score))
            observeds_.append(observed)
        except FloatingPointError:
            this_score = -np.inf
            if save_all:
                covariances_.append(np.nan)
                precisions_.append(np.nan)
                observeds_.append(np.nan)
        if X_test is not None:
            if not np.isfinite(this_score):
                this_score = -np.inf
            scores_.append(this_score)
        if verbose:
            if X_test is not None:
                print('[graphical_lasso_path] lambda_: %.2e, mu: %.2e, score: %.2e'
                      % (lambda_, mu, this_score))
            else:
                print('[graphical_lasso_path] lambda_: %.2e, mu: %.2e' % (lambda_, mu))
    if X_test is not None:
        return covariances_, precisions_, observeds_, scores_
    return covariances_, precisions_, observeds_


def EM(emp_cov, M, lambda_, mu_, assume_centered=False, tol=1e-3, max_iter=200, verbose=0, compute_objective=False,
       return_n_iter=False, penalize_latent=True):
    h = M.shape[1]
    o = emp_cov.shape[0]
    emp_cov_H = np.zeros((h, h))
    emp_cov_OH = np.zeros_like(M)
    
    K = np.random.randn(h+o, h+o)
    K = K.dot(K.T)
    K /= np.max(K)
     
    if penalize_latent:
        regularizer = np.ones((h+o, h+o))*lambda_
        regularizer -= np.diag(np.diag(regularizer))
    else:
        regularizer = np.zeros((h+o, h+o))
        regularizer[h:, h:] = lambda_*np.ones((o,o))
        regularizer[h:, h:] -=  np.diag(np.diag(regularizer[h:, h:]))
    regularizer[:h, h:] = mu_*M.T
    regularizer[h:, :h] = mu_*M
    penalized_nll = np.inf
    for iter_  in range(max_iter):
        
        #expectation step
        sigma = pinvh(K)
        sigma_o_inv = pinvh(sigma[h:, h:])
        sigma_ho = sigma[:h, h:]
        sigma_oh = sigma[h:, :h]
        emp_cov_H = (sigma[:h, :h] - sigma_ho.dot(sigma_o_inv).dot(sigma_oh)+
                     sigma_ho.dot(sigma_o_inv).dot(sigma[h:, h:]).dot(sigma_o_inv).dot(sigma_oh))
        emp_cov_OH = emp_cov.dot(sigma_o_inv).dot(sigma_oh)
        S = np.zeros_like(K)
        S[:h,:h] = emp_cov_H 
        S[:h, h:] =  emp_cov_OH.T
        S[h:, :h] = emp_cov_OH
        S[h:, h:] = emp_cov
        penalized_nll_old = penalized_nll
        penalized_nll = _penalized_nll(K, S, regularizer)
        
        # maximization step
        K, _ = graph_lasso(S, alpha=regularizer, return_n_iter=False)
        nll_diff = penalized_nll_old - penalized_nll
        if verbose:
            print("iter: %d, NLL: %.6f , NLL_diff: %.6f"%(iter_, penalized_nll, nll_diff))
        if  np.abs(nll_diff) < tol:
            break
    else:
        warnings.warn("The optimization of EM did not converged.")
    return K, S, penalized_nll_old


class EMLatentGraphLassoCV():
    """
    lambdas: the parameters to test
        If an integer n is passed then n parameters are selected in a (hopefully) suitable interval.
        If a list is passed then the parameters in the list are used to test the method.
        If a tuple of two elements is passed 10 elements between the elements of the tuple are tested.
    mus: the parameters to test
        If an integer n is passed then n parameters are selected in a (hopefully) suitable interval.
        If a list is passed then the parameters in the list are used to test the method.
        If a tuple of two elements is passed 10 elements between the elements of the tuple are tested.
    random_search: boolean, default True
        If True and lambdas and mus are either an integer or a tuple, the values are randomly selected, 
        not in a grid search. 
    """
    def __init__(self, mask, lambdas=4, mus=4, rho=1., score='ebic', cv=None, tol=1e-4,
                 max_iter=100, n_jobs=1, random_search=True, 
                 verbose=False, assume_centered=False,
                 update_rho=False, random_state=None, n_params=10, save_all=False):
        self.mask = mask
        self.lambdas = lambdas
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
        self.score = score
        self.random_search = random_search
        self.n_params = n_params
        self.save_all = save_all

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
        n_lambdas = self.lambdas
        inner_verbose = max(0, self.verbose - 1)

        if isinstance(n_lambdas, list):
            lambdas = self.lambdas
        else:
            lambda_1 = par_max(emp_cov)
            lambda_0 = 1e-2 * lambda_1
            if self.random_search:
                if isinstance(n_lambdas, tuple):
                     lambdas = np.array([self.random_state.uniform(n_lambdas[0], n_lambdas[1]) for i in range(self.n_params)])
                else:
                    lambdas = np.array([self.random_state.uniform(lambda_0, lambda_1) for i in range(n_lambdas)])
                print(lambdas)
            else:
                lambdas = np.logspace(np.log10(eta_0), np.log10(eta_1),
                                      n_lambdas)[::-1]
        
        n_mus = self.mus
        inner_verbose = max(0, self.verbose - 1)

        if isinstance(n_mus, list):
            mus = self.mus
        else:
            mu_1 = par_max(emp_cov) 
            mu_0 = 1e-2 * mu_1
            if self.random_search:
                if isinstance(n_mus, tuple):
                    mus = np.array([self.random_state.uniform(n_mus[0], n_mus[1]) for i in range(self.n_params)])
                else:
                    mus = np.array([self.random_state.uniform(mu_0, mu_1) for i in range(n_mus)])
            else:
                mus = np.logspace(np.log10(mu_0), np.log10(mu_1),
                                     n_mus)[::-1]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)

        this_path = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )(delayed(flgl_path)(X[train], mask=self.mask, lambdas=lambdas, mus=mus,
                                 X_test=X[test], random_search=self.random_search, tol=self.tol,
                                        max_iter=int(.1 * self.max_iter),
                                        update_rho=self.update_rho,
                                        verbose=0, random_state=self.random_state,
                                score=self.score, save_all=self.save_all)
              for train, test in cv.split(X, y))

        # Little danse to transform the list in what we need
        covs, precs, hidds, scores = zip(*this_path)
        if self.save_all:
            covs = zip(*covs)
            precs = zip(*precs)
            hidds = zip(*hidds)
        scores = zip(*scores)
        combinations = list(product(lambdas, mus))
        if self.save_all:
            path.extend(zip(combinations, scores, covs))
        else:
            path.extend(zip(combinations, scores))
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
        best_lambda, best_mu = combinations[best_index]
        self.lambda_ = best_lambda
        self.mu_ = best_mu
        self.cv_parameters_ = combinations

        # Finally fit the model with the selected alpha
        self.precision_, self.covariance_,  self.n_iter_ = EM(
            emp_cov, self.mask, lambda_=best_lambda, mu_=best_mu, tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            compute_objective=True, return_n_iter=True)
        return self