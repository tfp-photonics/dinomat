import jax.numpy as np


def atleast_nd(x, dims):
    if x.ndim >= dims:
        return x
    return np.reshape(x, x.shape + (dims - x.ndim) * (1,))


def find_roots(p):
    row0 = -p[..., 1:] / p[..., 0][..., None]
    rows_lower = np.eye(row0.shape[-1], dtype=row0.dtype)[:-1]
    rows_lower = np.resize(rows_lower, (*row0.shape[:2], *rows_lower.shape))
    mat = np.concatenate([row0, rows_lower], -2)
    roots = np.linalg.eigvals(mat)
    return roots


def filter_kz(x):
    """Filters out forward-propagating modes."""
    re, im = np.real(x), np.imag(x)
    x = np.where((im > 0) | ((im == 0) & (re > 0)), x, 0)
    nonzero = np.nonzero(x, size=x.size // 2)
    return x[nonzero].reshape((*x.shape[:-1], -1))
