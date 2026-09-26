"""Synthetic node: room mass balance + MQ135 physics + DHT11 lag/quantisation.
Gives ground truth for testing before hardware exists."""
import numpy as np
import pandas as pd

from .config import Config
from .sensor_physics import cf_datasheet, rs_to_adc


def _lag(x, tau):
    y = np.empty_like(x)
    y[0] = x[0]
    a = 1 - np.exp(-1.0 / tau)
    for k in range(1, len(x)):
        y[k] = y[k - 1] + a * (x[k] - y[k - 1])
    return y


def _bump(n, start, rise=60.0, fall=300.0, hold=300):
    k = np.arange(n) - start
    up = np.where(k >= 0, 1 - np.exp(-np.clip(k, 0, None) / rise), 0.0)
    down = np.where(k > hold, np.exp(-(k - hold) / fall), 1.0)
    return up * down


def simulate(duration_s=3600, people=(), thswing=(), swabs=(), seed=0, cfg=None,
             r0_true=None, t0=25.0, rh0=50.0, wander=(1.0, 5.0), b_true=None):
    """people: [(start, end, n)], thswing: [(start, dT, dRH)], swabs: [(start, dur, local_index)]."""
    cfg = cfg or Config()
    rng = np.random.default_rng(seed)
    n = duration_s
    k = np.arange(n)
    r0 = r0_true or cfg.r0_kohm
    b_exp = cfg.b if b_true is None else b_true  # true sensor exponent may differ from prior

    occ = np.zeros(n)
    for s, e, p in people:
        occ[s:e] += p
    T = t0 + wander[0] * np.sin(2 * np.pi * k / 5400 + rng.uniform(0, 6))
    RH = rh0 + wander[1] * np.sin(2 * np.pi * k / 7200 + rng.uniform(0, 6))
    for s, dT, dRH in thswing:
        b = _bump(n, s)
        T, RH = T + dT * b, RH + dRH * b

    # room mass balance, S = 5 index*m3/s per person, V = 30 m3
    C = np.empty(n)
    C[0] = cfg.c_bg
    e = np.exp(-cfg.lam)
    for i in range(1, n):
        sv = 5.0 * occ[i] / 30.0
        C[i] = cfg.c_bg + (C[i - 1] - cfg.c_bg) * e + sv / cfg.lam * (1 - e)

    local = np.zeros(n)
    fault = np.zeros(n, int)
    for s, dur, amp in swabs:
        kk = k - s
        prof = np.where((kk >= 0) & (kk < dur), 1 - np.exp(-np.clip(kk, 0, None) / 2.0), 0.0)
        tail = np.where(kk >= dur, (1 - np.exp(-dur / 2.0)) * np.exp(-np.clip(kk - dur, 0, None) / 12.0), 0.0)
        local += amp * (prof + tail)
        fault[(kk >= 0) & (kk < dur + 40)] = 1

    # true sensor CF deviates from datasheet (so refit matters)
    cf_true = cf_datasheet(T, RH) * np.exp(0.03 * (T - 25) / 5 - 0.02 * (RH - 50) / 20)
    rs = r0 * cf_true * ((C + local) / cfg.a) ** (1 / b_exp) * np.exp(rng.normal(0, 0.005, n))
    v = rs_to_adc(rs, cfg.rl_kohm)[:, None] + rng.normal(0, 0.7, (n, 10))
    v = np.clip(np.round(v), 0, cfg.adc_max)

    # DHT11: lag, offset, 1-unit quantisation, 2 s update
    Td = np.round(_lag(T, 10) + rng.uniform(-1, 1))
    RHd = np.round(_lag(RH, 6) + rng.uniform(-3, 3))
    Td, RHd = np.repeat(Td[::2], 2)[:n], np.repeat(RHd[::2], 2)[:n]

    return pd.DataFrame(dict(
        millis=k * 1000, adc_mean=v.mean(1), adc_std=v.std(1), T=Td, RH=RHd, dht_ok=1,
        c_true=C, c_local=C + local, T_true=T, RH_true=RH, fault=fault,
        fault_kind=np.where(fault == 1, "spike", ""),
    ))


def training_day(seed=1, cfg=None, b_true=None):
    """8 h with occupancy blocks and quiet periods for fitting."""
    n = 8 * 3600
    people = [(3600, 5400, 2), (9000, 12600, 1), (18000, 20000, 3), (25000, 26500, 1)]
    return simulate(n, people=people, seed=seed, cfg=cfg, wander=(4.0, 15.0), b_true=b_true)


def test_day(seed=7, cfg=None, b_true=None, **kw):
    """2 h scripted scenario; drift added later via faults.drift."""
    people = [(600, 2000, 2), (4300, 5200, 1)]
    thswing = [(2600, 5.0, 20.0)]
    swabs = [(1200, 10, 4000.0), (3800, 10, 4000.0)]
    return simulate(7200, people=people, thswing=thswing, swabs=swabs, seed=seed, cfg=cfg,
                    b_true=b_true, **kw)
