"""L0 + L1a: MQ135 signal chain and T/RH cross-sensitivity (sensor physics).

ppm = a * (Rs / (R0 * CF(T, RH)))^b ;  Rs = RL * (ADC_max / ADC - 1)
"""
import numpy as np

# GeorgK/MQ135 (T < 20 C) and balk77 linearisation (T >= 20 C). Verify vs repo source.
CORA, CORB, CORC, CORD = 0.00035, 0.02718, 1.39538, 0.0018
CORE, CORF, CORG = -0.003333333, -0.001923077, 1.130128205


def adc_to_rs(adc, rl_kohm, adc_max=1023):
    adc = np.clip(np.asarray(adc, float), 0.5, adc_max - 0.5)
    return rl_kohm * (adc_max / adc - 1.0)


def rs_to_adc(rs, rl_kohm, adc_max=1023):
    return adc_max * rl_kohm / (rl_kohm + np.asarray(rs, float))


def cf_datasheet(t, rh):
    t = np.asarray(t, float)
    rh = np.asarray(rh, float)
    low = CORA * t * t - CORB * t + CORC - (rh - 33.0) * CORD
    high = CORE * t + CORF * rh + CORG
    return np.where(t < 20.0, low, high)


def cf_kalra(t, rh):
    """Kalra et al. 2016 scaling factor (regression on datasheet Fig. 4)."""
    return 1.6979 - 0.012 * np.asarray(t, float) - 0.00612 * np.asarray(rh, float)


def log_index(rs, r0, a, b, cf=1.0):
    """log of CO2-equivalent index."""
    return np.log(a) + b * (np.log(rs) - np.log(r0) - np.log(cf))


def r0_from_clean(rs, cf, a, b, c_ref):
    """Invert power law at known clean-air index."""
    return np.asarray(rs, float) / cf * (a / c_ref) ** (1.0 / b)


class RefitCF:
    """Abdullah-style grey-box refit: log Rs = b0 + b1 T + b2 RH + b3 T*RH on clean air.
    CF normalised to 1 at datasheet reference (20 C, 33 %RH)."""

    T_REF, RH_REF = 20.0, 33.0

    def __init__(self, beta=None, t_range=(15.0, 35.0), rh_range=(20.0, 85.0), resid_sd=None):
        self.beta = None if beta is None else np.asarray(beta, float)
        self.t_range = tuple(t_range)
        self.rh_range = tuple(rh_range)
        self.resid_sd = resid_sd

    @staticmethod
    def _X(t, rh):
        t = np.atleast_1d(np.asarray(t, float))
        rh = np.atleast_1d(np.asarray(rh, float))
        return np.column_stack([np.ones_like(t), t, rh, t * rh])

    def fit(self, t, rh, rs):
        X = self._X(t, rh)
        y = np.log(np.asarray(rs, float))
        self.beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.resid_sd = float(np.std(y - X @ self.beta))
        self.t_range = (float(np.percentile(t, 1)), float(np.percentile(t, 99)))
        self.rh_range = (float(np.percentile(rh, 1)), float(np.percentile(rh, 99)))
        return self

    def __call__(self, t, rh):
        ref = self._X(self.T_REF, self.RH_REF) @ self.beta
        out = np.exp(self._X(t, rh) @ self.beta - ref)
        return out if np.ndim(t) else float(out[0])

    def envelope_penalty(self, t, rh):
        """0 inside fitted T/RH range, grows linearly outside (per 5 C / 10 %RH)."""
        dt = max(self.t_range[0] - t, 0.0, t - self.t_range[1]) / 5.0
        drh = max(self.rh_range[0] - rh, 0.0, rh - self.rh_range[1]) / 10.0
        return dt + drh

    def to_dict(self):
        return dict(beta=self.beta.tolist(), t_range=self.t_range,
                    rh_range=self.rh_range, resid_sd=self.resid_sd)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
