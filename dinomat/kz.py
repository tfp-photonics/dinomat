import jax.numpy as np

from dinomat.util import filter_kz, find_roots


def get_kz_tm_2(k0, kx, eps, mu):
    A = np.ones_like(kx)
    B = -(k0**2 * mu[1] * eps[0] - eps[0] / eps[2] * kx**2)

    O = np.zeros_like(A)
    p = np.stack([A, O, B], -1)

    return filter_kz(find_roots(p))


def get_kz_te_4(k0, kx, eps, mu, gamma):
    k02 = k0**2
    kx2 = kx**2

    A = -k02 * gamma[1] * mu[0] * mu[2] * np.ones_like(kx)
    B = mu[2] - 2 * k02 * kx2 * gamma[1] * mu[0] * mu[2]
    C = kx2 * mu[0] - k02 * mu[0] * mu[2] * (kx2**2 * gamma[1] + eps[1])

    O = np.zeros_like(A)
    p = np.stack([A, O, B, O, C], -1)

    return filter_kz(find_roots(p))


def get_kz_tm_4(k0, kx, eps, mu, gamma):
    k02 = k0**2
    kx2 = kx**2

    A = -k02 * gamma[0] * eps[2] * mu[1] * np.ones_like(kx)
    B = eps[2] - k02 * kx2 * mu[1] * (gamma[0] * eps[0] + gamma[2] * eps[2])
    C = eps[0] * (kx2 - k02 * mu[1] * (kx2**2 * gamma[2] + eps[2]))

    O = np.zeros_like(A)
    p = np.stack([A, O, B, O, C], -1)

    return filter_kz(find_roots(p))


def get_kz_te_6(k0, kx, eps, mu, gamma, tau):
    k02 = k0**2
    kx2 = kx**2

    A = k02 * tau[0] * np.ones_like(kx)
    B = k02 * (gamma[1] + kx2 * (2 * tau[0] + tau[2]))
    C = k02 * kx2 * (2 * gamma[1] + kx2 * (tau[0] + 2 * tau[2])) - 1 / mu[0]
    D = k02 * (kx2**2 * gamma[1] + eps[1] + kx2**3 * tau[2]) - kx2 / mu[2]

    O = np.zeros_like(A)
    p = np.stack([A, O, B, O, C, O, D], -1)

    return filter_kz(find_roots(p))


def get_kz_tm_6(k0, kx, eps, mu, gamma, tau):
    k02 = k0**2
    kx2 = kx**2

    A = -k02 * eps[2] * mu[1] * tau[1] * np.ones_like(kx)
    B = -k02 * (
        gamma[0] * eps[2] * mu[1]
        + kx2 * (eps[0] * mu[1] * tau[1] + 2 * eps[2] * mu[1] * tau[1])
    )
    C = eps[2] - k02 * kx2 * (
        gamma[0] * eps[0] * mu[1]
        + gamma[2] * eps[2] * mu[1]
        + kx2 * (2 * eps[0] * mu[1] * tau[1] + eps[2] * mu[1] * tau[1])
    )
    D = kx2 * eps[0] - k02 * (
        kx2**2 * (gamma[2] * eps[0] * mu[1] + kx2 * eps[0] * mu[1] * tau[1])
        + eps[0] * eps[2] * mu[1]
    )

    O = np.zeros_like(A)
    p = np.stack([A, O, B, O, C, O, D], -1)

    return filter_kz(find_roots(p))
