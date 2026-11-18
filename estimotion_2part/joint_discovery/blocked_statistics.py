"""Independent-block statistics for joint selection v2 (spec section 6).

Tracked points share images, surfaces and calibration, and adjacent frames share tracker state. None of them is an
independent replicate. Everything here resamples *whole* units that are declared independent before fitting:

* time blocks wider than the measured residual autocorrelation (``block_length_from_autocorrelation``);
* spatial surface clusters (voxels) for per-interval point sets;
* cameras, as a leave-one-out sensitivity (never as a two-sample bootstrap).

Duplicating tracks/pixels/pairs therefore cannot shrink an interval (spec tests 6/23).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


# --------------------------------------------------------------------------- temporal blocks
def autocorrelation(x, max_lag=None) -> np.ndarray:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        return np.array([1.0])
    x = x - x.mean()
    v = float(x @ x)
    if v <= 0:
        return np.array([1.0])
    L = min(max_lag or n // 2, n - 2)
    return np.array([float(x[: n - k] @ x[k:]) / v for k in range(L + 1)])


def block_length_from_autocorrelation(series_list, threshold=0.2, min_len=2, max_len=None) -> dict:
    """First lag at which the pooled autocorrelation of residual series drops below ``threshold``.

    ``series_list``: per-track (or per-camera) residual time series sampled on the model time grid. Returns the
    chosen block length with the pooled curve so the choice is auditable. Falls back to ``min_len`` only when
    nothing can be measured (recorded as such)."""
    curves = [autocorrelation(s) for s in series_list if np.isfinite(np.asarray(s, float)).sum() >= 6]
    if not curves:
        return dict(block_length=int(min_len), source="fallback_min_len (no series with >= 6 samples)",
                    pooled_acf=None, threshold=threshold)
    L = max(len(c) for c in curves)
    acc, cnt = np.zeros(L), np.zeros(L)
    for c in curves:
        acc[: len(c)] += c
        cnt[: len(c)] += 1
    pooled = acc / np.maximum(cnt, 1)
    below = np.flatnonzero(pooled < threshold)
    lag = int(below[0]) if len(below) else L
    bl = max(int(min_len), lag)
    if max_len is not None:
        bl = min(bl, int(max_len))
    return dict(block_length=bl, source="pooled residual autocorrelation", pooled_acf=pooled.round(4).tolist(),
                threshold=threshold, n_series=len(curves))


@dataclass
class BlockPartition:
    """Contiguous time blocks declared before any fitting. ``heldout`` marks blocks withheld from training."""
    T: int
    block_length: int
    blocks: list            # list[list[int]]
    heldout: list           # list[bool]
    reference_time_index: int
    source: str

    @property
    def heldout_times(self) -> list:
        return sorted(int(t) for b, h in zip(self.blocks, self.heldout) if h for t in b)

    @property
    def train_times(self) -> list:
        return sorted(int(t) for b, h in zip(self.blocks, self.heldout) if not h for t in b)

    def block_of(self) -> np.ndarray:
        out = -np.ones(self.T, int)
        for k, b in enumerate(self.blocks):
            out[b] = k
        return out

    def to_json(self) -> dict:
        d = asdict(self)
        d.update(heldout_times=self.heldout_times, train_times=self.train_times, n_blocks=len(self.blocks),
                 n_heldout_blocks=int(sum(self.heldout)))
        return d


def make_block_partition(T: int, block_length: int, ref: int, heldout_fraction=0.25, source="") -> BlockPartition:
    """Contiguous blocks of ``block_length``; every k-th *interior* block (never the one holding the reference, the
    first or the last block) is held out, so held-out blocks are separated by >= one training block."""
    bl = max(1, int(block_length))
    blocks = [list(range(s, min(s + bl, T))) for s in range(0, T, bl)]
    if len(blocks) > 1 and len(blocks[-1]) < max(1, bl // 2):      # merge a tiny tail
        blocks[-2] += blocks.pop()
    nb = len(blocks)
    ref_block = next(k for k, b in enumerate(blocks) if ref in b)
    interior = [k for k in range(1, nb - 1) if k != ref_block]
    n_ho = int(round(heldout_fraction * nb))
    held = [False] * nb
    if interior and n_ho > 0:
        # every other interior block at most: a held-out block is never adjacent to another held-out block
        cand = interior[::2] if len(interior[::2]) >= n_ho else interior
        pick = np.linspace(0, len(cand) - 1, min(n_ho, len(cand))).round().astype(int)
        for k in np.unique(pick):
            held[cand[k]] = True
    return BlockPartition(T=T, block_length=bl, blocks=blocks, heldout=held, reference_time_index=int(ref),
                          source=source)


def count_independent_blocks(times, block_length) -> int:
    """Number of distinct blocks of ``block_length`` touched by ``times`` -- six adjacent frames in two runs of three
    count as (at most) two blocks, never six (spec test 28)."""
    t = np.unique(np.asarray(times, int))
    if len(t) == 0:
        return 0
    n, last = 1, t[0]
    for x in t[1:]:
        if x - last >= block_length:
            n += 1
            last = x
    return n


# --------------------------------------------------------------------------- resampling
def block_bootstrap(per_block_values, n_boot=2000, seed=0, stat=np.sum, alpha=0.05) -> dict:
    """Bootstrap over independent blocks. ``per_block_values``: one scalar (or 1D array, summed) per block."""
    vals = [np.atleast_1d(np.asarray(v, float)) for v in per_block_values]
    vals = [v[np.isfinite(v)] for v in vals]
    vals = [v for v in vals if len(v)]
    nb = len(vals)
    if nb < 2:
        return dict(n_blocks=nb, estimate=float(stat(np.concatenate(vals))) if nb else None, ci=None,
                    status="INSUFFICIENT_INDEPENDENT_EVIDENCE")
    rng = np.random.default_rng(seed)
    est = float(stat(np.concatenate(vals)))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, nb, nb)
        boots[i] = stat(np.concatenate([vals[k] for k in pick]))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(n_blocks=nb, estimate=est, ci=[float(lo), float(hi)], boot_sd=float(boots.std(ddof=1)),
                p_positive=float(np.mean(boots > 0)), status="OK")


def spatial_clusters(points, voxel_m=0.05) -> np.ndarray:
    """Surface-cluster id per point (voxel hash). Points in one voxel share surface/calibration failure modes."""
    P = np.asarray(points, float)
    if len(P) == 0:
        return np.zeros(0, int)
    key = np.floor(P / voxel_m).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    return inv.ravel()


def cluster_jackknife(estimator, cluster_ids, n_min=4):
    """Delete-one-cluster jackknife covariance of a vector estimator(keep_mask) -> (cov, n_clusters).

    Duplicated points fall in the same cluster, so duplication cannot reduce the covariance."""
    ids = np.unique(cluster_ids)
    G = len(ids)
    if G < n_min:
        return None, G
    full = np.asarray(estimator(np.ones(len(cluster_ids), bool)), float)
    reps = []
    for g in ids:
        r = estimator(cluster_ids != g)
        if r is not None:
            reps.append(np.asarray(r, float))
    if len(reps) < n_min:
        return None, G
    R = np.stack(reps)
    m = R.mean(0)
    cov = (len(R) - 1) / len(R) * (R - m).T @ (R - m)
    return cov, G


def effective_n_from_clusters(cluster_ids, weights=None) -> float:
    """Kish effective number of clusters (weights summed per cluster)."""
    ids = np.asarray(cluster_ids)
    if len(ids) == 0:
        return 0.0
    w = np.ones(len(ids)) if weights is None else np.asarray(weights, float)
    s = np.bincount(np.unique(ids, return_inverse=True)[1].ravel(), w)
    return float(s.sum() ** 2 / max((s ** 2).sum(), 1e-30))
