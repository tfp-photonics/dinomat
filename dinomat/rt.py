import jax.numpy as np

from dinomat.fresnel import (
    get_fmat_te_4,
    get_fmat_te_6,
    get_fmat_tm_2,
    get_fmat_tm_4,
    get_fmat_tm_6,
)
from dinomat.kz import (
    get_kz_te_4,
    get_kz_te_6,
    get_kz_tm_2,
    get_kz_tm_4,
    get_kz_tm_6,
)
from dinomat.util import atleast_nd


def get_rt(k0, kx, x, thickness, epsSC=1.0, pol=1):
    x = np.reshape(x, (len(x), -1, 1, 1))
    x = x[0::2] + 1j * x[1::2]
    x = np.repeat(x[:, None], 3, axis=1)

    k0 = atleast_nd(k0, 3)
    kx = atleast_nd(kx, 3)

    if len(x) == 2:  # WSD
        if pol == 0:  # TE
            raise NotImplementedError("Only TM implemented for WSD")
        elif pol == 1:
            return get_rt_tm_2(k0, kx, *x, thickness, epsSC)
    elif len(x) == 3:  # 4th order
        if pol == 0:  # TE
            return get_rt_te_4(k0, kx, *x, thickness, epsSC)
        elif pol == 1:
            return get_rt_tm_4(k0, kx, *x, thickness, epsSC)
    elif len(x) == 4:  # 6th order
        if pol == 0:
            return get_rt_te_6(k0, kx, *x, thickness, epsSC)
        elif pol == 1:
            return get_rt_tm_6(k0, kx, *x, thickness, epsSC)


def get_rt_tm_2(k0, kx, eps, mu, thickness, epsSC):
    kz = get_kz_tm_2(k0, kx, eps, mu)
    fmat = get_fmat_tm_2(k0, kx, kz, eps, mu, thickness, epsSC)

    s0 = np.sqrt(epsSC**2 * k0**2 - kx**2)
    s1 = epsSC**2 * k0**2 * np.ones_like(kx)
    zeros = np.zeros((*kx.shape[:2], 2))
    src = np.concatenate([s0, s1, zeros], -1)

    rt = np.linalg.solve(fmat, -src)
    return rt[..., (0, -1)]


def get_rt_tm_4(k0, kx, eps, mu, gamma, thickness, epsSC):
    kz = get_kz_tm_4(k0, kx, eps, mu, gamma)
    fmat = get_fmat_tm_4(k0, kx, kz, eps, mu, gamma, thickness, epsSC)

    s0 = np.sqrt(epsSC**2 * k0**2 - kx**2)
    s1 = epsSC**2 * k0**2 * np.ones_like(kx)
    zeros = np.zeros((*kx.shape[:2], 4))
    src = np.concatenate([s0, s1, zeros], -1)

    rt = np.linalg.solve(fmat, -src)
    return rt[..., (0, -1)]


def get_rt_te_4(k0, kx, eps, mu, gamma, thickness, epsSC):
    kz = get_kz_te_4(k0, kx, eps, mu, gamma)
    fmat = get_fmat_te_4(k0, kx, kz, eps, mu, gamma, thickness, epsSC)

    I = np.ones_like(kx)
    kzi = np.sqrt((epsSC * k0) ** 2 - kx**2)
    zeros = np.zeros((*kx.shape[:2], 4))
    src = np.concatenate([-I, -kzi, zeros], -1)

    rt = np.linalg.solve(fmat, -src)
    return rt[..., (0, -1)]


def get_rt_te_6(k0, kx, eps, mu, gamma, tau, thickness, epsSC):
    kz = get_kz_te_6(k0, kx, eps, mu, gamma, tau)
    fmat = get_fmat_te_6(k0, kx, kz, eps, mu, gamma, tau, thickness, epsSC)

    I = np.ones_like(kx)
    kzi = np.sqrt((epsSC * k0) ** 2 - kx**2)
    zeros = np.zeros((*kx.shape[:2], 6))
    src = np.concatenate([-I, -kzi, zeros], -1)

    rt = np.linalg.solve(fmat, -src)
    return rt[..., (0, -1)]


def get_rt_tm_6(k0, kx, eps, mu, gamma, tau, thickness, epsSC):
    kz = get_kz_tm_6(k0, kx, eps, mu, gamma, tau)
    fmat = get_fmat_tm_6(k0, kx, kz, eps, mu, gamma, tau, thickness, epsSC)

    s0 = np.sqrt(epsSC**2 * k0**2 - kx**2)
    s1 = epsSC**2 * k0**2 * np.ones_like(kx)
    zeros = np.zeros((*kx.shape[:2], 6))
    src = np.concatenate([s0, s1, zeros], -1)

    rt = np.linalg.solve(fmat, -src)
    return rt[..., (0, -1)]
