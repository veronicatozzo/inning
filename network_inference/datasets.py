import numpy as np

from itertools import product
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.utils import check_random_state


def generate_latent_network(n_obs=100, n_lat=10, n_samples=500,
                            sparsity_obs=0.3, sparsity_lat=0.7,
                            sparsityinter=0.3, random_state=None):

#     random_state = check_random_state(random_state)
#     glob =  np.zeros((n_obs+n_lat, n_obs+n_lat))
#     glob[n_lat:, n_lat:] = make_sparse_spd_matrix(dim=n_obs, alpha=1-sparsity_obs,
#                                    random_state=random_state)
#     glob[:n_lat, :n_lat] = make_sparse_spd_matrix(dim=n_lat, alpha=1-sparsity_lat,
#                                    random_state=random_state)
#     inter = np.zeros((n_obs, n_lat))
#     prod = np.array(list(product(np.arange(0, n_obs), np.arange(0, n_lat))))
#     np.random.shuffle(prod)
#     length = int(prod.shape[0]*(1-sparsityinter))
#     indices_r = [p[0] for p in prod[:length]]
#     indices_c = [p[1] for p in prod[:length]]
#     inter[indices_r, indices_c] = random_state.randn(length)
#     inter /= 1e-6
#     glob[n_lat:, :n_lat] = inter
#     glob[:n_lat, n_lat:] = inter.T
#     #sum_ = np.sum(glob[:n_lat, :n_lat], axis=0) + np.sum(inter, axis=0)
#     #glob[:n_lat, :n_lat] += np.diag(sum_)

#     T_obs = glob[n_lat:, n_lat:] - \
#             inter.dot(np.linalg.inv(glob[:n_lat, :n_lat])).dot(inter.T)
#     print(is_pos_semi_def(inter.dot(np.linalg.inv(glob[:n_lat, :n_lat])).dot(inter.T)))
#     samples = np.random.multivariate_normal(np.zeros(n_obs),
#                                             np.linalg.inv(T_obs), n_samples)
    A = make_sparse_spd_matrix(dim=n_obs + n_lat, alpha=sparsity_obs, random_state=random_state)

    T_true = A[n_lat:,n_lat:]
    K_true = A[n_lat:,:n_lat]
    H_true = A[0:n_lat,0:n_lat]
    per_cov = K_true*0.3
    T_obs = T_true - per_cov.dot(np.linalg.inv(H_true)).dot(per_cov.T)
    print(is_pos_semi_def(per_cov.dot(np.linalg.inv(H_true)).dot(per_cov.T)))
    samples = np.random.multivariate_normal(np.zeros(n_obs), np.linalg.inv(T_obs), n_samples)

    return T_obs, T_true, H_true, K_true, samples

def is_pos_def(x, tol=1e-15):
    """Check if x is positive definite."""
    eigs = np.linalg.eigvalsh(x)
    eigs[np.abs(eigs) < tol] = 0
    return np.all(eigs > 0)


def is_pos_semi_def(x, tol=1e-15):
    """Check if x is positive definite."""
    eigs = np.linalg.eigvalsh(x)
    eigs[np.abs(eigs) < tol] = 0
    return np.all(eigs >= 0)


def generate_two_layers_network(
        d1, n1, d2, n2, sparsity1=0.7, sparsity2=0.7, sparsityinter=0.3,
        random_state=None):
    """
    Generation of the network, covariance and samples for a 2 layers network.

    Parameters
    ----------

    d1: int
        Number of dimension for the first layer.

    n1: int
        Number of samples to generate for the first layer.

    d2: int
        Number of dimension for the second layer.

    n2: int
        Number of samples to generate for the second layer.

    sparsity1: float between 0 and 1, optional (default=0.95)
        The probability that a coefficient in the first layer is zero.
        Larger values enforce more sparsity.

    sparsity2: float between 0 and 1, optional (default=0.95)
        The probability that a coefficient in the second layer is zero.
        Larger values enforce more sparsity.

    sparsityinter: float between 0 and 1, optional (default=0.95)
        The probability that a inter link is zero.
        Larger values enforce more sparsity.

    Returns
    -------
    K1: numpy.array, size=(d1, d1)
        True precision matrix for the first layer.
    K2: numpy.array, size=(d2, d2)
        True precision matrix for the second layer.
    R: numpy.array, size=(d1, d2)
        Links inter layers.
    K1_obs: numpy.array, size=(d1, d1)
        Observed precision matrix for the first layer.
    K2_obs: numpy.array, size=(d2, d2)
        Observed precision matrix for the second layer.
    cov1: numpy.array, size=(d1, d1)
        Covariance matrix for the first layer.
    cov2: numpy.array, size=(d2, d2)
        Covariance matrix for the second layer.
    samples1: numpy.array, size=(n1, d1)
        Observation from multivariate normal distribution.
    samples2: numpy.array, size=(n2, d2)
        Observation from multivariate normal distribution.
    """
    random_state = check_random_state(random_state)
    # K1 = make_sparse_spd_matrix(dim=d1, alpha=sparsity1,
    #                             random_state=random_state)
    # K2 = make_sparse_spd_matrix(dim=d2, alpha=sparsity2,
    #                             random_state=random_state)

    K = make_sparse_spd_matrix(dim=d1+d2, alpha=sparsity1,
                               random_state=random_state)
    # R = np.zeros((d1, d2))
    # values = random_state.rand(d1, d2)
    # indices = np.where(random_state.rand(d1, d2) > sparsityinter)
    # R[indices] = values[indices]
    # print(R)
    # R *= 0.2
    # print(R)
    # print("---------------------------------------------------")
    K1 = K[0:d1, 0:d1]
    K2 = K[d1:, d1:]
    R = K[:d1, d1:]
    K1_obs = K1 - np.linalg.multi_dot((R, np.linalg.pinv(K2), R.T))
    K2_obs = K2 - np.linalg.multi_dot((R.T, np.linalg.pinv(K1), R))
    assert is_pos_def(K1_obs), ("The observed precision matrix of the first "
                                "layer is not pd")
    assert is_pos_def(K2_obs), ("The observed precision matrix of the second "
                                "layer is not pd")

    cov1 = np.linalg.inv(K1_obs)
    cov2 = np.linalg.inv(K2_obs)
    samples1 = random_state.multivariate_normal(np.zeros(d1), cov1, size=n1)
    samples2 = random_state.multivariate_normal(np.zeros(d2), cov2, size=n2)

    return K1, K2, R, K1_obs, K2_obs, cov1, cov2, samples1, samples2
