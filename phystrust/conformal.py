"""L5: split-conformal on normalised residuals |target - fused| / sigma (log units)."""
import numpy as np


class Conformal:
    def __init__(self, alpha=0.1, q=1.645):
        self.alpha, self.q = alpha, q  # default = Gaussian 90% until calibrated

    def fit(self, scores):
        s = np.asarray(scores, float)
        s = s[np.isfinite(s)]
        n = len(s)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q = float(np.quantile(s, level, method="higher"))
        return self

    def interval(self, logc, sigma_log):
        return float(np.exp(logc - self.q * sigma_log)), float(np.exp(logc + self.q * sigma_log))
