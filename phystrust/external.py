"""External training data for the learned prior (widens the trained T/RH and dynamics range).

DALTON (Karmakar et al., 2024, arXiv:2407.14501; github.com/prasenjit52282/dalton-dataset, AGPL, non-commercial
research): 1 Hz indoor air quality from 30 sites in West Bengal, India. We use NDIR CO2 + T + RH from bedroom
monitors: June (31-38 C, 25-54 %RH) and October (27-30 C, 52-68 %RH), i.e. Chennai-like indoor climate with real
occupancy rises and decays.

The prior predicts the next-step change of log C from the last 60 s of [dlog C, T, RH], so the absolute level
(NDIR ppm vs MQ135 index) does not matter; only dynamics and T/RH do.
"""
import glob
import os

import numpy as np
import pandas as pd

DALTON_DIR = "data/external/dalton"


def dalton_segments(path=DALTON_DIR, min_len=900):
    """Contiguous valid 1 Hz runs as (logc, T, RH) arrays; gaps, invalid readings and sensor zeros split runs."""
    segs = []
    for f in sorted(glob.glob(os.path.join(path, "*.csv"))):
        d = pd.read_csv(f, usecols=["ts", "T", "H", "CO2", "Valid", "Valid_CO2"])
        ok = ((d["Valid"] == 1) & (d["Valid_CO2"] == 1) & (d["T"] > 0) & (d["H"] > 0) & (d["CO2"] > 250)).values
        # smooth the NDIR quantisation (1 ppm steps) like the node's fused estimate
        co2 = d["CO2"].rolling(5, center=True, min_periods=1).median().values
        edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a >= min_len:
                segs.append((np.log(co2[a:b]), d["T"].values[a:b], d["H"].values[a:b]))
    return segs


def external_xy(prior, segs, max_windows=60000, seed=0):
    """Strided training windows from the external segments, capped so our own node's data is not swamped."""
    total = sum(max(0, len(s[0]) - prior.need) for s in segs)
    if total == 0:
        return None, None
    stride = max(1, total // max_windows)
    X, y = [], []
    for logc, t, rh in segs:
        for k in range(prior.need - 1, len(logc) - 1, stride):
            s = slice(k - prior.need + 1, k + 1)
            X.append(prior.features(logc[s], t[s], rh[s]))
            y.append(logc[k + 1] - logc[k])
    X, y = np.array(X), np.array(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))[:max_windows]
    return X[idx], y[idx]
