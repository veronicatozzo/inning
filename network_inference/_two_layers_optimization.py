from __future__ import division

import numpy as np
import warnings
import sys

from scipy import linalg
from functools import partial

from sklearn.covariance import GraphicalLasso, empirical_covariance
from sklearn.utils.extmath import fast_logdet , squared_norm
from sklearn.utils.validation import check_array, check_random_state
from sklearn.datasets import make_sparse_spd_matrix

from network_inference.utils import check_data_dimensions, convergence, \
                                    update_rho, l1_od_norm
from network_inference.utils import _scalar_product, update_rho, convergence
from network_inference.prox import prox_logdet, soft_thresholding_od, \
                                    soft_thresholding_sign
from network_inference.datasets import is_pos_def, is_pos_semi_def


def objective2LGL(emp_cov, K1, K2, A1, A2, R, alpha1, alpha2, tau, rho):
    """Objective function for time-varying graphical lasso."""
    K = [A1, A2]
    sample_sizes = [E.shape[0] for E in emp_cov]
    obj = np.sum(-(1/n) * log_likelihood(E, precision)
                 for E, precision, n in zip(emp_cov, K, sample_sizes))
    obj += alpha1 * l1_od_norm(K1)
    obj += alpha2 * l1_od_norm(K2)
    obj += tau * l1_od_norm(R)
    return obj


def _R_update(K1, K2, A1, A2, R, U1, U2, rho, tau, max_iter):
    gamma = 1e-7
    invK1 = np.linalg.pinv(K1)
    invK2 = np.linalg.pinv(K2)
    KAU2 = K2 - A2 + U2
    KAU1 = K1 - A1 + U1
    for iter_ in range(max_iter):
        R_old = R.copy()
        invK1R = invK1.dot(R)
        RinvK2 = R.dot(invK2)

        gradient = -2*rho*(invK1R.dot(KAU2 - R.T.dot(invK1R)) +
                           (KAU1 - RinvK2.dot(R.T)).dot(RinvK2))
        if(np.any(np.isnan(gradient))):
            print(KAU1)
            print(KAU2)
            print(KAU2 - R.T.dot(invK1R))
            print(KAU1 - RinvK2.dot(R.T))
            print(iter, R)
        R = R - gamma*gradient
        R = soft_thresholding_sign(R, tau*gamma)
        #print(np.linalg.norm(R_old - R)/np.linalg.norm(R))
        if np.linalg.norm(R_old - R)/np.linalg.norm(R) < 1e-4:
            #print(R)
            break
    else:
        warnings.warn("The update of the matrix R did not converge.")
    return R


def _K_update(K1, K2, A1, A2, R, U1, U2, rho, alpha1, alpha2, max_iter):
    gamma = 1e-7
    AU1 = U1 - A1
    AU2 = U2 - A2

    for iter_ in range(max_iter):
        K1_old = K1.copy()
        K2_old = K2.copy()



def two_layers_graphical_lasso(
        data_list, alpha1=0.01, alpha2=0.01, tau=0.01, mode='admm', rho=1.,
        tol=1e-3, rtol=1e-5, max_iter=100, verbose=False, return_n_iter=True,
        return_history=False, compute_objective=False, compute_emp_cov=False,
        random_state=None, update_rho=False):
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
    update_rho: bool, optional
        If True the value of the parameter rho is updated.
        Default False.

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
        # M = A1 + np.linalg.multi_dot((R, np.linalg.pinv(K2), R.T)) + U1
        # K1 = soft_thresholding_od(M, lamda=alpha1 / rho)
        # # update K2
        # M = A2 + np.linalg.multi_dot((R.T, np.linalg.pinv(K1), R)) + U2
        # K2 = soft_thresholding_od(M, lamda=alpha2 / rho)
        K1, K2 = _K_update(K1, K2, A1, A2, R, U1, U2, rho, alpha1, alpha2,
                           max_iter=500)
        # update A1
        M = K1 - np.linalg.multi_dot((R, np.linalg.pinv(K2), R.T)) - U1
        M += M.T
        M /= 2.
        A1 = prox_logdet(data_list[0] - (rho/n2) * M, lamda=n1 / rho)
        # update A2
        M = K2 - np.linalg.multi_dot((R.T, np.linalg.pinv(K1), R)) - U2
        M += M.T
        M /= 2.
        A2 = prox_logdet(data_list[1] - (rho/n2) * M, lamda=n2 / rho)

        # update R
        R = _R_update(K1, K2, A1, A2, R, U1, U2, rho, tau,
                      max_iter=1000)
        #print(R)
        # update residuals
        RK2R = np.linalg.multi_dot((R, np.linalg.pinv(K2), R.T))
        RK1R = np.linalg.multi_dot((R.T, np.linalg.pinv(K1), R))
        U1 += A1 - K1 + RK2R
        U2 += A2 - K2 + RK1R

        # diagnostics, reporting, termination checks
        obj = objective2LGL(emp_cov, K1, K2, A1, A2, R, alpha1, alpha1, tau, rho) \
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
        if update_rho:
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


#
#
# OPTIMIZATION OF TWO LAYERS WITH FIXED LINKS
#
#
def objective_H(H, R=None, T=None, K=None, U= None,_rho=1, _mu=1):
    if not is_pos_def(H):
        return np.inf
    return 0.5 * _rho * squared_norm(R - T + U + np.linalg.multi_dot((K.T, linalg.pinvh(H), K))) \
            + _mu * l1_od_norm(H)

def _choose_lambda(lamda, R, T, K, H, U,  _rho, _mu, prox, grad, gamma, delta=1e-4, eps=0.9, max_iter=500):
    """Choose lambda for backtracking.

    References
    ----------
    Salzo S. (2017). https://doi.org/10.1137/16M1073741

    """
    partial_f = partial(objective_H, R=R, T=T, K=K, U=U, _rho=_rho, _mu=_mu)
    fx = partial_f(H)

    y_minus_x = prox - H
    tolerance = _scalar_product(y_minus_x, grad)
    tolerance += delta / gamma * _scalar_product(y_minus_x, y_minus_x)
    #print("Tolerance:", tolerance)
    for i in range(max_iter):
        # line-search
        x1 = H + lamda * y_minus_x

        loss_diff = partial_f(x1) - fx
        #print("Loss diff:", loss_diff)
        if loss_diff <= lamda * tolerance:
              break
        lamda *= eps
    else:
        warnings.warn("Did not find lambda")
    return lamda, i + 1

def _choose_gamma(gamma, H, R, T, K, U, _rho, _mu, _lambda, grad,
                 eps=0.9, max_iter=500):
    """Choose gamma for backtracking.

    References
    ----------
    Salzo S. (2017). https://doi.org/10.1137/16M1073741

    """
    partial_f = partial(objective_H, R=R, T=T, K=K, U=U, _rho=_rho, _mu=_mu)
    fx = partial_f(H)
    for i in range(max_iter):
        prox = soft_thresholding_od(H - gamma * grad, _mu * gamma)
        if is_pos_def(prox):
            break
        gamma *= eps
    else:
        warnings.warn("Did not find gamma")
    return gamma, prox

def _upgrade_H(H, R, T, K, U, _rho, _mu, verbose=0, random_state=None):
    # H = make_sparse_spd_matrix(dim=K.shape[0], alpha=0.5, random_state=random_state)
    _lambda = 1
    gamma = 1
    obj = 1e+10
    for iter_ in range(2000):
        H_old = H.copy()
        Hinv = linalg.pinvh(H)
        gradient = - _rho * K.dot(R - T + U + np.linalg.multi_dot((K.T, Hinv, K))).dot(K.T).dot(Hinv).dot(Hinv)
        gamma, _ = _choose_gamma(gamma, H, R, T, K, U, _rho,_mu, _lambda, gradient)
        # print(gamma)
        Y = soft_thresholding_od(H - gamma * gradient, gamma * _mu)
        _lambda,_ = _choose_lambda(_lambda, R, T, K, H, U,_rho, _mu, Y, gradient, 1, max_iter=1000, delta=1e-2)

        H = H + _lambda * (Y - H)
        
        obj_old = obj
        obj = objective_H(H, R, T, K, U,_rho=_rho, _mu=_mu)
        obj_diff = obj_old - obj
        iter_diff =np.linalg.norm(H - H_old) 
        if verbose:
            print("Iter: %d, obj: %.5f, iter_diff: %.5f, obj_diff:%.10f"%(iter_, obj, iter_diff, obj_diff))
        if(obj_diff<1e-4): 
            break
    else:
        warnings.warn("Algorithm for H minimization did not converge")
    return H


def objectiveFLGL(emp_cov, K, R, T, H, U, mu, eta, rho):
    res = - fast_logdet(R) + np.sum(R * emp_cov)
    res += rho / 2. * squared_norm(R - T + U + np.linalg.multi_dot((K.T, linalg.pinvh(H), K)))
    res += mu * l1_od_norm(H)
    res += eta * l1_od_norm(T)
    return res
    

def two_layers_fixed_links_GL(X, K, mu=0.01, eta=0.01, rho=1., 
        tol=1e-3, rtol=1e-5, max_iter=100, verbose=False, return_n_iter=True,
        return_history=False, compute_objective=False, compute_emp_cov=False,
        random_state=None, update_rho=False):
    """
    Params
    ------
    X: data or empirical covariance matrix
    L: links between observed and hidden layer

    """
    random_state = check_random_state(random_state)
    if compute_emp_cov:
        n = X.shape[0] 
        emp_cov = empirical_covariance(X, assume_centered=False)
    else:
        emp_cov = X

    H = make_sparse_spd_matrix(K.shape[0], alpha=0.5, random_state=random_state)
    #H = np.eye(K.shape[0])
    T = emp_cov.copy()
    T = (T + T.T) / 2.
    R = T - np.linalg.multi_dot((K.T, linalg.pinvh(H), K))
    U = np.zeros((K.shape[1], K.shape[1]))
    
    checks = []
    for iteration_ in range(max_iter):
        R_old = R.copy()
        
        # R update
        Hinv = linalg.pinvh(H)
        M = T - U - K.T.dot(Hinv).dot(K)
        M = (M + M.T)/2
        R = prox_logdet(emp_cov - rho * M, 1. / rho)
        assert is_pos_def(R), "iter %d"%iteration_
        
        # T update
        M = R + U + K.T.dot(Hinv).dot(K)
        M = (M + M.T) / 2.
        T = soft_thresholding_od(M, eta / rho)
        assert is_pos_def(T, tol=1e-8), "teta iter %d"%iteration_
       
        # H update
        H = _upgrade_H(H, R, T, K, U, rho, mu, verbose=0)
        assert(is_pos_def(H))
        
        # U update
        KHK = np.linalg.multi_dot((K.T, linalg.pinvh(H), K))
        U += R - T + KHK

        # diagnostics, reporting, termination checks 
        obj = objectiveFLGL(emp_cov, K, R, T, H,U, mu, eta, rho) \
            if compute_objective else np.nan
        rnorm = np.linalg.norm(R - T + KHK)
        snorm = rho * np.linalg.norm(R - R_old)
        check = convergence(
            obj=obj, rnorm=rnorm, snorm=snorm,
            e_pri=(np.sqrt(R.size) * tol + rtol *
                   max(np.linalg.norm(R),
                       np.linalg.norm(T - KHK))),
            e_dual=(np.sqrt(R.size) * tol + rtol * rho *
                    np.linalg.norm(U))
        )

        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % check)

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break
        if update_rho:
            rho_new = update_rho(rho, rnorm, snorm, iteration=iteration_)
            # scaled dual variables should be also rescaled
            U *= rho / rho_new
            rho = rho_new
    else:
        warnings.warn("Objective did not converge.")

    return_list = [linalg.pinvh(T), T, H, R]
    if return_n_iter:
        return_list.append(iteration_)
    if return_history:
        return_list.append(checks)
    return return_list
