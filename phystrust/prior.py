"""L2: learned prior (deep ensembles). Predicts next-step change in log index from fused
history + T/RH; ensemble spread = epistemic uncertainty; Mahalanobis distance = OOD score.

EnsemblePrior: 5 bootstrap MLPs on a 10-d summary vector.
LSTMPrior:     5 bootstrap LSTMs on a 12-step sequence (60 s @ 5 s) of [dlogC, T, RH]."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neural_network import MLPRegressor


def _shrunk_inv(Z):
    """Inverse of a Ledoit-Wolf shrinkage covariance. The lag features are nearly collinear, so the
    sample covariance is almost singular and its (pseudo-)inverse lets tiny shape changes dominate the
    Mahalanobis distance; shrinkage regularises those near-zero eigen-directions."""
    return np.linalg.inv(LedoitWolf().fit(Z).covariance_)


def _ood_summary(S):
    """Reduced OOD features: lag level, lag slope, T, RH, dT, dRH. The six lags are nearly collinear, so
    Mahalanobis on all of them amplifies tiny shape differences (false OOD on steeper-than-trained rises)."""
    S = np.atleast_2d(S)
    lags = S[:, :6]
    return np.column_stack([lags.mean(1), lags[:, -1] - lags[:, 0], S[:, 6:10]])


class EnsemblePrior:
    def __init__(self, n_members=5, hist_s=60, step_s=10, dT_win=30, seed=0):
        self.n_members, self.hist_s, self.step_s = n_members, hist_s, step_s
        self.dT_win, self.seed = dT_win, seed
        self.members, self.mu, self.sd, self.cov_inv = [], None, None, None

    @property
    def need(self):
        return max(self.hist_s, self.dT_win) + 1

    def features(self, logc, t, rh):
        """logc, t, rh: 1-D arrays with >= self.need samples, last = now."""
        logc, t, rh = np.asarray(logc), np.asarray(t), np.asarray(rh)
        cur = logc[-1]
        lags = logc[-1 - np.arange(self.step_s, self.hist_s + 1, self.step_s)] - cur
        w = self.dT_win
        return np.concatenate([lags, [t[-1], rh[-1], t[-1] - t[-1 - w], rh[-1] - rh[-1 - w]]])

    def dataset(self, logc, t, rh):
        X, y = [], []
        for k in range(self.need - 1, len(logc) - 1):
            s = slice(k - self.need + 1, k + 1)
            X.append(self.features(logc[s], t[s], rh[s]))
            y.append(logc[k + 1] - logc[k])
        return np.array(X), np.array(y)

    def _oz(self, S):
        S = np.atleast_2d(S)[:, :10]
        return ((S if len(self.mu) == 10 else _ood_summary(S)) - self.mu) / self.sd

    def d2(self, X):
        """Vectorised Mahalanobis D^2 of the OOD summary features (no network evaluation)."""
        Z = self._oz(X)
        return self._dist(Z)

    def _dist(self, Z):
        """OOD score of standardised summaries. Single Gaussian: Mahalanobis D^2. With a fitted mixture
        (multi-climate training data): negative log-likelihood above the training median, >= 0."""
        Z = np.atleast_2d(Z)
        if getattr(self, "gmm", None) is not None:
            return np.maximum(0.0, -self.gmm.score_samples(Z) - self.nll0)
        return np.einsum("ij,jk,ik->i", Z, self.cov_inv, Z)

    def fit_ood_mixture(self, X, X_hold, k=6):
        """Replace the single-Gaussian OOD reference by a Gaussian mixture. Needed when training data mixes
        climates (e.g. hot-dry June + humid October): one Gaussian then puts its mass between the clusters and
        calls typical inputs far. Threshold = 99.5 % of the training score."""
        from sklearn.mixture import GaussianMixture
        O = _ood_summary(np.atleast_2d(X)[:, :10])
        self.mu, self.sd = O.mean(0), O.std(0) + 1e-9
        Z = (O - self.mu) / self.sd
        # reg_covar 0.05 (standardised units): DHT11 whole-unit T/RH steps otherwise give needle components and an
        # unstable likelihood (tuned: held-out node data 7.5 % flagged; RH 85 %, 40 C, perfume-like spike >= 1.9x thr)
        self.gmm = GaussianMixture(k, covariance_type="full", reg_covar=0.05, random_state=0).fit(Z)
        self.nll0 = float(np.median(-self.gmm.score_samples(Z)))
        # 2x margin on the training 99.5 %: real node 09:30-20:00 OOD 8 % (day) / 18 % (unseen humid evening),
        # while RH 85 %, 40 C and cold-dry inputs stay above threshold (1.3x, 1.5x, 2.5x)
        self.ood_thr = 2.0 * float(np.quantile(self.d2(X), 0.995))
        self.ood_hold_rate = float(np.mean(self.d2(X_hold) > self.ood_thr))
        return self

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        O = _ood_summary(X)
        self.mu, self.sd = O.mean(0), O.std(0) + 1e-9
        self.cov_inv = _shrunk_inv((O - self.mu) / self.sd)
        self.x_mu, self.x_sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.x_mu) / self.x_sd
        self.members = []
        for i in range(self.n_members):
            idx = rng.integers(0, len(Z), len(Z))
            m = MLPRegressor(hidden_layer_sizes=(16, 16), max_iter=500, early_stopping=True,
                             random_state=self.seed + i)
            m.fit(Z[idx], y[idx])
            self.members.append(m)
        return self

    def predict(self, x, k=None):
        """Returns (delta_mean, delta_std, mahalanobis_d2). k = number of members to evaluate."""
        xs = getattr(self, "x_mu", None)
        z = (x - (self.mu if xs is None else xs)) / (self.sd if xs is None else self.x_sd)
        preds = np.array([m.predict(z[None])[0] for m in self.members[:k]])
        o = self._oz(x)[0]
        d2 = float(self._dist(o)[0])
        return float(preds.mean()), float(preds.std()), d2


try:  # module-level so trained members can be pickled into the bundle
    import torch.nn as _nn

    class LSTMNet(_nn.Module):
        """Sequence [dlogC, T, RH] x 12 -> LSTM(32) -> linear -> next-step change."""

        def __init__(self, h):
            super().__init__()
            self.lstm = _nn.LSTM(3, h, batch_first=True)
            self.head = _nn.Linear(h, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1]).squeeze(-1)
    class TCNNet(_nn.Module):
        """Causal temporal CNN: 3 dilated conv layers (d = 1, 2, 4), kernel 3 -> last step -> linear."""

        def __init__(self, h):
            super().__init__()
            layers, c = [], 3
            for dil in (1, 2, 4):
                layers += [_nn.ConstantPad1d((2 * dil, 0), 0.0), _nn.Conv1d(c, h, 3, dilation=dil), _nn.ReLU()]
                c = h
            self.net = _nn.Sequential(*layers)
            self.head = _nn.Linear(h, 1)

        def forward(self, x):                       # x: (B, L, 3)
            y = self.net(x.transpose(1, 2))         # (B, h, L)
            return self.head(y[:, :, -1]).squeeze(-1)

    class TransformerNet(_nn.Module):
        """Temporal Transformer encoder: linear embed + learned positions, 1 layer, 4 heads -> last token."""

        def __init__(self, h, seq_len=12):
            super().__init__()
            self.embed = _nn.Linear(3, h)
            self.pos = _nn.Parameter(_nn.init.normal_(_nn.Parameter(__import__("torch").zeros(1, seq_len, h)), std=0.02))
            layer = _nn.TransformerEncoderLayer(h, 4, 2 * h, dropout=0.0, batch_first=True)
            self.enc = _nn.TransformerEncoder(layer, 1)
            self.head = _nn.Linear(h, 1)

        def forward(self, x):
            y = self.enc(self.embed(x) + self.pos[:, :x.shape[1]])
            return self.head(y[:, -1]).squeeze(-1)
except ImportError:  # torch optional: MLP prior still works
    LSTMNet = TCNNet = TransformerNet = None


class LSTMPrior(EnsemblePrior):
    """Deep ensemble of small LSTMs. Same interface as EnsemblePrior:
    features() returns [10-d summary | flattened (seq_len x 3) sequence]."""

    def __init__(self, n_members=5, hist_s=60, step_s=10, dT_win=30, seq_stride=5,
                 hidden=32, epochs=8, lr=1e-3, batch=256, seed=0):
        super().__init__(n_members, hist_s, step_s, dT_win, seed)
        self.seq_stride, self.hidden, self.epochs, self.lr, self.batch = seq_stride, hidden, epochs, lr, batch
        self.seq_len = hist_s // seq_stride
        self.seq_mu = self.seq_sd = None
        self.y_sd = 1.0

    def features(self, logc, t, rh):
        logc, t, rh = np.asarray(logc), np.asarray(t), np.asarray(rh)
        summary = super().features(logc, t, rh)
        idx = -1 - np.arange(self.seq_len)[::-1] * self.seq_stride   # oldest -> newest
        seq = np.stack([logc[idx] - logc[-1], t[idx], rh[idx]], axis=1)
        return np.concatenate([summary, seq.ravel()])

    def _split(self, X):
        X = np.atleast_2d(X)
        return X[:, :10], X[:, 10:].reshape(len(X), self.seq_len, 3)

    def _net(self):
        return LSTMNet(self.hidden)

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        S, Q = self._split(X)
        O = _ood_summary(S)                                      # summary stats -> OOD
        self.mu, self.sd = O.mean(0), O.std(0) + 1e-9
        self.cov_inv = _shrunk_inv((O - self.mu) / self.sd)
        self.seq_mu, self.seq_sd = Q.reshape(-1, 3).mean(0), Q.reshape(-1, 3).std(0) + 1e-9
        Qn = torch.tensor((Q - self.seq_mu) / self.seq_sd, dtype=torch.float32)
        self.y_sd = float(y.std() + 1e-9)
        yt = torch.tensor(y / self.y_sd, dtype=torch.float32)
        self.members, self.train_log = [], []
        for i in range(self.n_members):
            net = self._net()
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            idx = rng.integers(0, len(Qn), len(Qn))               # bootstrap resample
            n_val = max(1, len(idx) // 10)
            tr, va = idx[n_val:], idx[:n_val]
            best, best_state = np.inf, None
            for ep in range(self.epochs):
                net.train()
                perm = tr[rng.permutation(len(tr))]
                for b in range(0, len(perm), self.batch):
                    j = perm[b:b + self.batch]
                    loss = torch.mean((net(Qn[j]) - yt[j]) ** 2)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    vl = float(torch.mean((net(Qn[va]) - yt[va]) ** 2))
                if vl < best:                                      # early stopping on val MSE
                    best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
            net.load_state_dict(best_state)
            net.eval()
            self.members.append(net)
            self.train_log.append(best)
        return self

    def predict(self, x, k=None):
        import torch
        S, Q = self._split(x)
        z = self._oz(S)[0]
        q = torch.tensor((Q - self.seq_mu) / self.seq_sd, dtype=torch.float32)
        with torch.no_grad():
            preds = np.array([float(m(q)[0]) for m in self.members[:k]]) * self.y_sd
        return float(preds.mean()), float(preds.std()), float(self._dist(z)[0])


class TCNPrior(LSTMPrior):
    """Same deep ensemble, temporal-CNN members."""

    def _net(self):
        return TCNNet(self.hidden)


class TransformerPrior(LSTMPrior):
    """Same deep ensemble, Transformer-encoder members."""

    def _net(self):
        return TransformerNet(self.hidden, self.seq_len)


PRIORS = {"mlp": EnsemblePrior, "lstm": LSTMPrior, "tcn": TCNPrior, "transformer": TransformerPrior}


def admission_test(prior_cls, X, y, val_frac=0.2, **kw):
    """Fit on the first (1 - val_frac) of the data, score one-step skill vs persistence on the rest.
    skill = 1 - MSE(model) / MSE(persistence); persistence predicts zero change.
    Returns (prior refit on all data or None if rejected, skill)."""
    n = len(y)
    cut = int(n * (1 - val_frac))
    p = prior_cls(**kw).fit(X[:cut], y[:cut])
    pred = np.array([p.predict(x)[0] for x in X[cut:]])
    skill = 1.0 - np.mean((pred - y[cut:]) ** 2) / max(np.mean(y[cut:] ** 2), 1e-12)
    if skill <= 0.0:
        return None, float(skill)
    # empirical OOD threshold: 99.5th percentile of Mahalanobis D^2 over ALL training data, so every
    # regime seen in training (e.g. occupancy rises) counts as in-distribution. Features are not Gaussian,
    # so the chi^2 value over-flags. Mahalanobis with 10 dims and >10^4 samples barely overfits in-sample.
    full = prior_cls(**kw).fit(X, y)
    # in-sample quantile under-states novelty tolerance; also take held-out D^2 of the 80 % model on the
    # last 20 % (time split), so a normal regime unseen in the first part does not count as OOD.
    full.ood_thr = float(max(np.quantile(full.d2(X), 0.995), np.quantile(p.d2(X[cut:]), 0.995)))
    return full, float(skill)
