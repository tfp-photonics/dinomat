#!/usr/bin/env python3

import argparse
from functools import partial
from pathlib import Path

import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
from jax import jacfwd, jit, random
from jax.config import config
from scipy.io import loadmat
from scipy.optimize import least_squares
from tqdm import tqdm

from dinomat.rt import get_rt


@partial(jit, static_argnums=(5,))
def _residuals(x, rt_tgt, k0, kx, dslab, polarizations, epsSC=1.0):
    rt = np.stack([get_rt(k0, kx, x, dslab, epsSC, pol=p) for p in polarizations])
    res = (rt - rt_tgt).ravel()
    return np.concatenate([np.real(res), np.imag(res)])


def residuals(x, rt_tgt, k0, kx, dslab, polarizations):
    return onp.array(_residuals(x, rt_tgt, k0, kx, dslab, polarizations))


_jacobian = jit(jacfwd(_residuals), static_argnums=(5,))


def jacobian(x, rt_tgt, k0, kx, dslab, polarizations):
    return onp.array(_jacobian(x, rt_tgt, k0, kx, dslab, polarizations))


def run_trials(lsq_fun, num_trials, x0, bounds, args, rng_key):
    lb, ub = bounds
    best = {"cost": np.inf}
    for _ in range(num_trials):
        info = lsq_fun(x0.ravel(), bounds=(lb.ravel(), ub.ravel()), args=args)
        if info["cost"] < best["cost"]:
            best = info
        x0 = random.uniform(rng_key, shape=lb.shape, minval=lb, maxval=ub)
    return best


def main(args):
    config.update("jax_enable_x64", True)

    key = random.PRNGKey(args.seed)
    data = loadmat(args.rt_data)
    k0s = slice(None)
    kxs = slice(None)
    ntrials = args.ntrials

    match args.model:
        case "wsd":
            nparams = 4
        case "ssd_gamma":
            nparams = 6
        case "ssd_tau":
            nparams = 8

    polarizations = (1,)  # (0,) for TE, (0, 1) for both TE & TM

    rt_tgt = np.stack([data["R"], data["T"]], axis=-1)[(k0s, kxs, slice(None))][None]

    k0 = data["k0"].ravel()[k0s]
    kx = data["kx"][(k0s, kxs)]
    thickness = data["thickness"].item() / 1000

    param_names = [
        "eps_re",
        "eps_im",
        "mu_re",
        "mu_im",
        "gamma_re",
        "gamma_im",
        "tau_re",
        "tau_im",
    ][:nparams]

    lb = np.array([-1e1, 0, -1e1, -1e1, -1e-1, -1e-1, -1e-1, -1e-1])[:nparams]
    ub = np.array([1e1, 1e1, 1e1, 1e1, 1e-1, 1e-1, 1e-5, 1e-5])[:nparams]
    x0 = random.uniform(key, shape=lb.shape, minval=lb, maxval=ub) / 2

    lsq_fun = partial(
        least_squares,
        residuals,
        jac=jacobian,
        ftol=1e-9,
        xtol=1e-9,
        loss="soft_l1",
        x_scale="jac",
    )

    lb_ = lb
    ub_ = ub
    results = {k: [] for k in param_names}
    for idx in tqdm(range(len(k0))):
        info = run_trials(
            lsq_fun,
            ntrials,
            x0,
            (lb_, ub_),
            (rt_tgt[:, idx, :, None], k0[idx], kx[idx], thickness, polarizations),
            key,
        )

        x0 = info["x"]
        lb_ = x0 - np.abs(x0) / 10
        ub_ = x0 + np.abs(x0) / 10
        lb_ = np.where(lb_ < lb, lb, lb_)
        ub_ = np.where(ub_ > ub, ub, ub_)

        for n, p in enumerate(param_names):
            results[p].append(info["x"][n])

    fig, ax = plt.subplots(nparams // 2, 2, sharex=True)
    for axi, (k, v) in zip(ax.ravel(), results.items()):
        axi.plot(k0, v)
        axi.set_title(k)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rt_data", type=Path, required=True)
    parser.add_argument(
        "--model", type=str, default="wsd", choices=["wsd", "ssd_gamma", "ssd_tau"]
    )
    parser.add_argument("--ntrials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=8257024450357582)
    main(parser.parse_args())
