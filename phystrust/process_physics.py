"""L1b: single-zone well-mixed mass balance  dC/dt = S/V - lam (C - C_bg)."""
import numpy as np


def transition(c, lam, c_bg, dt, s_over_v=0.0):
    """Exact discretisation, source held constant over the step."""
    e = np.exp(-lam * dt)
    return c_bg + (c - c_bg) * e + (s_over_v / lam) * (1.0 - e)


def fit_decay(t_s, c, c_a, min_excess=5.0):
    """ASTM E741 concentration decay: ln(C - Ca) = ln(C0 - Ca) - lam t.
    Returns lam [1/s], C0, R^2."""
    t_s = np.asarray(t_s, float)
    c = np.asarray(c, float)
    m = (c - c_a) > min_excess
    if m.sum() < 10:
        raise ValueError("decay segment too short or too close to background")
    t, y = t_s[m] - t_s[m][0], np.log(c[m] - c_a)
    slope, icpt = np.polyfit(t, y, 1)
    yhat = slope * t + icpt
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return -slope, c_a + np.exp(icpt), r2


def max_rate(c_est, cfg, noise_rate=0.0):
    """Plausibility envelope on |dC/dt| for room-level change."""
    return cfg.rise_max + cfg.lam_max * abs(c_est - cfg.c_bg) + 3.0 * noise_rate
