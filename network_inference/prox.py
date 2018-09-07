import numpy as np


def prox_logdet(a, lamda):
    """Time-varying latent variable graphical lasso prox."""
    es, Q = np.linalg.eigh(a)
    xi = (- es + np.sqrt(np.square(es) + 4. / lamda)) * lamda / 2.
    return np.linalg.multi_dot((Q, np.diag(xi), Q.T))


def soft_thresholding_sign(a, lamda):
    """Soft-thresholding."""
    return np.sign(a) * np.maximum(np.abs(a) - lamda, 0)


def soft_thresholding_od(a, lamda):
    """Off-diagonal soft-thresholding."""
    if a.ndim > 2:
        res = []
        for i, x in enumerate(a):
            st = soft_thresholding_sign(x, lamda[i])
            st.flat[::x.shape[0]+1] = np.diag(x)
            res.append(st)
        return np.array(res)
    soft = np.sign(a) * np.maximum(np.abs(a) - lamda, 0)
    soft.flat[::a.shape[1] + 1] = np.diag(a)
    return soft
