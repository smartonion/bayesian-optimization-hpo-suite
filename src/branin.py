import numpy as np

def branin(x, a=1.0,
           b=5.1 / (4.0 * np.pi**2),
           c=5.0 / np.pi,
           r=6.0,
           s=10.0,
           t=1.0 / (8.0 * np.pi)):
    """
    Standard Branin-Hoo function.

    Parameters
    ----------
    x : array_like
        Either shape (..., 2) where the last dimension is [x1, x2],
        or any 2-tuple/list (x1, x2).

    Returns
    -------
    f : ndarray or float
        Function value(s).
    """
    x = np.asarray(x)
    x1 = x[..., 0]
    x2 = x[..., 1]

    term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
    term2 = s * (1 - t) * np.cos(x1)
    y = term1 + term2 + s
    return y



def branin_modified(x, **kwargs):
    """
    Forrester et al. modified Branin:
    f_mod(x1, x2) = Branin(x1, x2) + 5 * x1
    """
    x = np.asarray(x)
    x1 = x[..., 0]
    return branin(x, **kwargs) + 5.0 * x1

