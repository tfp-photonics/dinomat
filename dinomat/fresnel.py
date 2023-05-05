from functools import partial

import jax.numpy as np

_cat = partial(np.concatenate, axis=-1)


def get_fmat_tm_2(k0, kx, kz, eps, mu, thickness, epsSC):
    O = np.zeros_like(kx)
    I = np.ones_like(kx)
    k02 = k0**2
    kzi = np.sqrt((epsSC * k0) ** 2 - kx**2)

    fa = -eps[2] / eps[0] * kz
    fb = -eps[2] * k02 * I

    return np.stack(
        [
            _cat([-kzi, fa, -fa, O]),
            _cat([k02 * I, fb, fb, O]),
            _cat(
                [
                    O,
                    fa * np.exp(1j * kz * thickness),
                    -fa * np.exp(-1j * kz * thickness),
                    kzi,
                ]
            ),
            _cat(
                [
                    O,
                    fb * np.exp(1j * kz * thickness),
                    fb * np.exp(-1j * kz * thickness),
                    k02 * I,
                ]
            ),
        ],
        axis=-2,
    )


def get_fmat_te_4(k0, kx, kz, eps, mu, gamma, thickness, epsSC):
    k02 = k0**2
    kx2 = kx**2
    kz2 = kz**2
    kzi = np.sqrt((epsSC * k0) ** 2 - kx2)

    fb = -kz * (-k02 * (kx2 + kz2) * gamma[1] + 1 / mu[0])
    fc = (kx2 + kz2) * gamma[1]

    O = np.zeros_like(kx)
    I = np.ones_like(kx)
    return np.stack(
        [
            _cat([I, -I, -I, -I, -I, O]),
            _cat([-kzi, fb, -fb, O]),
            _cat([O, fc, fc, O]),
            _cat([O, -np.exp(1j * kz * thickness), -np.exp(-1j * kz * thickness), I]),
            _cat(
                [
                    O,
                    fb * np.exp(1j * kz * thickness),
                    -fb * np.exp(-1j * kz * thickness),
                    kzi,
                ]
            ),
            _cat(
                [
                    O,
                    fc * np.exp(1j * kz * thickness),
                    fc * np.exp(-1j * kz * thickness),
                    O,
                ]
            ),
        ],
        axis=-2,
    )


def get_fmat_tm_4(k0, kx, kz, eps, mu, gamma, thickness, epsSC):
    k02 = k0**2
    kx2 = kx**2
    kz2 = kz**2
    kzi = np.sqrt((epsSC * k0) ** 2 - kx2)
    kR = np.sqrt(kx2 + kzi**2)
    kzT = np.sqrt((epsSC * k0) ** 2 - kx2)
    kT = np.sqrt(kx2 + kzi**2)

    fa = -eps[2] * kz / eps[0]
    fb = (kx2 + eps[2] * kz2 / eps[0]) * (
        k02 * (kz2 * gamma[0] + kx2 * gamma[2]) - 1 / mu[1]
    )
    fc = -kz * gamma[0] * (kx2 + eps[2] * kz2 / eps[0])

    O = np.zeros_like(kx)
    return np.stack(
        [
            _cat([-kzi, fa, -fa, O]),
            _cat([kR**2, fb, fb, O]),
            _cat([O, fc, -fc, O]),
            _cat(
                [
                    O,
                    fa * np.exp(1j * kz * thickness),
                    -fa * np.exp(-1j * kz * thickness),
                    kzT,
                ],
            ),
            _cat(
                [
                    O,
                    fb * np.exp(1j * kz * thickness),
                    fb * np.exp(-1j * kz * thickness),
                    kT**2,
                ],
            ),
            _cat(
                [
                    O,
                    fc * np.exp(1j * kz * thickness),
                    -fc * np.exp(-1j * kz * thickness),
                    O,
                ]
            ),
        ],
        axis=-2,
    )


def get_fmat_te_6(k0, kx, kz, eps, mu, gamma, tau, thickness, epsSC):
    k02 = k0**2
    kx2 = kx**2
    kz2 = kz**2
    kzi = np.sqrt((epsSC * k0) ** 2 - kx2)

    fb = -kz * (
        1 / mu[0] - k02 * (kx2 + kz2) * (gamma[1] + kz2 * tau[0] + kx2 * tau[2])
    )
    fc = (kx2 + kz2) * (gamma[1] + kz2 * tau[0] + kx2 * tau[2])
    fd = kz * (kx2 + kz2) * tau[0]

    O = np.zeros_like(kx)
    I = np.ones_like(kx)
    return np.stack(
        [
            _cat([I, -I, -I, -I, -I, -I, -I, O]),
            _cat([-kzi, fb, -fb, O]),
            _cat([O, fc, fc, O]),
            _cat([O, fd, -fd, O]),
            _cat(
                [O, -np.exp(1j * kz * thickness), -np.exp(-1j * kz * thickness), I],
            ),
            _cat(
                [
                    O,
                    fb * np.exp(1j * kz * thickness),
                    -fb * np.exp(-1j * kz * thickness),
                    kzi,
                ],
            ),
            _cat(
                [
                    O,
                    fc * np.exp(1j * kz * thickness),
                    fc * np.exp(-1j * kz * thickness),
                    O,
                ]
            ),
            _cat(
                [
                    O,
                    fd * np.exp(1j * kz * thickness),
                    -fd * np.exp(-1j * kz * thickness),
                    O,
                ]
            ),
        ],
        axis=-2,
    )


def get_fmat_tm_6(k0, kx, kz, eps, mu, gamma, tau, thickness, epsSC):
    k02 = k0**2
    kx2 = kx**2
    kz2 = kz**2
    kzi = np.sqrt((epsSC * k0) ** 2 - kx2)
    kR = np.sqrt(kx2 + kzi**2)
    kzT = np.sqrt((epsSC * k0) ** 2 - kx2)
    kT = np.sqrt(kx2 + kzi**2)

    fa = -eps[2] * kz / eps[0]
    fb = (kx2 + eps[2] * kz2 / eps[0]) * (
        k02 * (gamma[0] * kz2 + gamma[2] * kx2 + tau[1] * (kx2 + kz2) ** 2) - 1 / mu[1]
    )
    fc = -kz * (gamma[0] + tau[1] * (kx2 + kz2)) * (kx2 + eps[2] * kz2 / eps[0])
    fd = tau[1] * (kx2 + kz2) * (kx2 + eps[2] * kz2 / eps[0])

    O = np.zeros_like(kx)
    return np.stack(
        [
            _cat([-kzi, fa, -fa, O]),
            _cat([kR**2, fb, fb, O]),
            _cat([O, fc, -fc, O]),
            _cat([O, fd, fd, O]),
            _cat(
                [
                    O,
                    fa * np.exp(1j * kz * thickness),
                    -fa * np.exp(-1j * kz * thickness),
                    kzT,
                ],
            ),
            _cat(
                [
                    O,
                    fb * np.exp(1j * kz * thickness),
                    fb * np.exp(-1j * kz * thickness),
                    kT**2,
                ],
            ),
            _cat(
                [
                    O,
                    fc * np.exp(1j * kz * thickness),
                    -fc * np.exp(-1j * kz * thickness),
                    O,
                ]
            ),
            _cat(
                [
                    O,
                    fd * np.exp(1j * kz * thickness),
                    fd * np.exp(-1j * kz * thickness),
                    O,
                ]
            ),
        ],
        axis=-2,
    )
