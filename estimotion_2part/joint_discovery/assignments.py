"""Persistent per-track static / moving / outlier responsibilities (spec 4.1, 4.2, 4.4-A).

The mixture is over *observation hypotheses* for a material track; a track's position is never an
affine blend of static and moving predictions. Each component's likelihood aggregates all of the
track's valid observations (missing observations simply contribute nothing). Static and moving
components both carry one fitted 3D anchor per track, so neither gains free parameters.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp

COMPONENTS = ("static", "moving", "outlier")


def student_t_logpdf(r, dof):
    """Log density of a whitened residual (unit scale; the scale term is added by the caller)."""
    return (gammaln((dof + 1) / 2) - gammaln(dof / 2) - 0.5 * np.log(dof * np.pi)
            - (dof + 1) / 2 * np.log1p(r * r / dof))


def observation_loglik(res, sigma_uv, sigma_z, noise):
    """Per-observation log-likelihood [M] from whitened residuals (NaN entries contribute 0).

    ``sigma_uv`` scalar px and ``sigma_z`` [M] metres convert densities back to measurement units
    so static / moving / outlier components are on the same scale."""
    r_uv, r_z = res["r_uv"], res["r_z"]
    ll = np.zeros(len(r_z))
    ok = np.isfinite(r_uv)
    ll += np.where(ok, student_t_logpdf(np.where(ok, r_uv, 0), noise.student_t_dof) - np.log(sigma_uv), 0).sum(1)
    okz = np.isfinite(r_z)
    ll += np.where(okz, student_t_logpdf(np.where(okz, r_z, 0), noise.student_t_dof) - np.log(np.where(okz, sigma_z, 1)), 0)
    return ll


def outlier_loglik(obs, noise, use_depth=True):
    ll = np.where(obs.uv_valid, -2 * np.log(noise.outlier_px_range), 0.0)
    if use_depth:
        dz = obs.depth_valid & obs.mask_valid & np.isfinite(obs.z)
        ll = ll + np.where(dz, -np.log(noise.outlier_depth_range_m), 0.0)
    return ll


def track_sum(obs, per_obs, mask=None):
    m = np.ones(obs.n_obs, bool) if mask is None else mask
    return np.bincount(obs.track[m], per_obs[m], minlength=obs.n_tracks)


def responsibilities(ll_static, ll_moving, ll_outlier, priors, eligible):
    """Posterior [N,3]; ineligible tracks (no depth anchor / too few obs) keep NaN = unknown."""
    L = np.stack([ll_static, ll_moving, ll_outlier], 1) + np.log(np.asarray(priors, float))[None]
    post = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    post[~eligible] = np.nan
    return post


def update_priors(post, eligible, floor=0.01):
    p = np.nanmean(post[eligible], 0) if eligible.any() else np.array([0.65, 0.3, 0.05])
    p = np.clip(p, floor, 1.0)
    return p / p.sum()


def mixture_nll(ll_static, ll_moving, ll_outlier, priors, eligible):
    L = np.stack([ll_static, ll_moving, ll_outlier], 1) + np.log(np.asarray(priors, float))[None]
    return -logsumexp(L[eligible], axis=1)


def labels(post, eligible, confidence):
    lab = np.full(len(post), "unknown", dtype=object)
    for k, name in enumerate(COMPONENTS):
        lab[eligible & (post[:, k] >= confidence)] = name
    return lab
