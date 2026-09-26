"""Fault injection on logs (operates in Rs domain, re-quantised to ADC).
Each injector sets 'fault'=1 and 'fault_kind' on affected rows."""
import numpy as np

from .sensor_physics import adc_to_rs, rs_to_adc


def _label(out, mask, kind):
    prev = out["fault"].values if "fault" in out else np.zeros(len(out), int)
    prev_kind = out["fault_kind"].values if "fault_kind" in out else np.full(len(out), "", object)
    out["fault"] = np.where(mask, 1, prev)
    out["fault_kind"] = np.where(mask, kind, prev_kind)
    return out


def _apply_rs(df, mask, fn, rl, kind):
    out = df.copy()
    rs = adc_to_rs(out["adc_mean"].values, rl)
    rs[mask] = fn(rs[mask], np.flatnonzero(mask))
    out["adc_mean"] = np.round(rs_to_adc(rs, rl), 2)
    return _label(out, mask, kind)


def _mask(df, start, end=None):
    k = np.arange(len(df))
    return (k >= start) & (k < (len(df) if end is None else end))


def drift(df, start, rate_per_min=0.002, rl=10.0):
    """Multiplicative R0 drift (sensitivity loss => Rs increases)."""
    return _apply_rs(df, _mask(df, start),
                     lambda r, i: r * (1 + rate_per_min) ** ((i - start) / 60.0), rl, "drift")


def gain(df, start, end, g=0.8, rl=10.0):
    return _apply_rs(df, _mask(df, start, end), lambda r, i: r * g, rl, "gain")


def noise(df, start, end, factor=3.0, base_sd=0.005, seed=0, rl=10.0):
    rng = np.random.default_rng(seed)
    sd = base_sd * factor
    return _apply_rs(df, _mask(df, start, end),
                     lambda r, i: r * np.exp(rng.normal(0, sd, len(r))), rl, "noise")


def stuck(df, start, end):
    out = df.copy()
    m = _mask(df, start, end)
    out.loc[m, "adc_mean"] = out["adc_mean"].iloc[start]
    out.loc[m, "adc_std"] = 0.0
    return _label(out, m, "stuck")


def spike(df, start, dur=10, depth=0.4, tau_up=2.0, tau_down=12.0, tail=40, rl=10.0):
    """Swab-like sensor-local interference: Rs drops to `depth`, fast rise/recovery."""
    m = _mask(df, start, start + dur + tail)

    def fn(r, i):
        k = (i - start).astype(float)
        on_end = 1 - np.exp(-dur / tau_up)
        prof = np.where(k < dur, 1 - np.exp(-k / tau_up), on_end * np.exp(-(k - dur) / tau_down))
        return r * (1 - (1 - depth) * prof)

    return _apply_rs(df, m, fn, rl, "spike")
