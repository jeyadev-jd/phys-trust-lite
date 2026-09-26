"""Build calibration bundle from a clean-ish log; fit lambda from a decay test.

python -m phystrust.calibrate bundle --log data/room.csv --out models/bundle.joblib
python -m phystrust.calibrate decay  --log data/decay.csv --start 120 --end 1500
"""
import argparse

import numpy as np
import pandas as pd

from .config import Config
from .conformal import Conformal
from .pipeline import Bundle, run_frame
from .prior import PRIORS, admission_test
from .process_physics import fit_decay
from .sensor_physics import RefitCF, adc_to_rs, cf_datasheet, r0_from_clean


def quiet_mask(rs, t, rh, win=1200, tol=0.03):
    """Near-clean-air samples: datasheet-corrected Rs within tol of its rolling max
    (reducing gases lower Rs, so clean air = highest corrected Rs)."""
    y = pd.Series(np.log(rs) - np.log(cf_datasheet(t, rh)))
    top = y.rolling(win, center=True, min_periods=1).max()
    return (y >= top - tol).values


def build_bundle(df, cfg=None, clean_window=(0, 600), truth_col=None, split=0.7, external=None):
    cfg = cfg or Config()
    ok = np.isfinite(df["T"]) & np.isfinite(df["RH"])
    if "dht_ok" in df:
        ok &= df["dht_ok"] == 1
    df = df[ok].reset_index(drop=True)
    t, rh = df["T"].values, df["RH"].values
    rs = adc_to_rs(df["adc_mean"].values, cfg.rl_kohm, cfg.adc_max)

    # refit needs real T/RH variation, else keep datasheet CF
    q = quiet_mask(rs, t, rh)
    refit = None
    if np.ptp(t[q]) >= 3.0 and np.ptp(rh[q]) >= 10.0:
        refit = RefitCF().fit(t[q], rh[q], rs[q])
    cf_fn = refit if refit is not None else cf_datasheet
    w = slice(*clean_window)
    r0 = float(np.median(r0_from_clean(rs[w], cf_fn(t[w], rh[w]), cfg.a, cfg.b, cfg.c_ref)))
    b = Bundle(cfg, r0, refit)

    n_tr = int(len(df) * split)
    base = run_frame(df.iloc[:n_tr], b, "refit_phys")
    X, y = PRIORS[cfg.prior_kind]().dataset(np.log(base["fused"].values), t[:n_tr], rh[:n_tr])
    n_ext = 0
    if external:  # external windows first: the admission test validates on the tail = this node's own data
        from .external import external_xy
        Xe, ye = external_xy(PRIORS[cfg.prior_kind](), external)
        if Xe is not None:
            X, y, n_ext = np.vstack([Xe, X]), np.r_[ye, y], len(ye)
    # admission test: the learned prior must beat persistence on held-out data, else it is disabled
    b.prior, prior_skill = admission_test(PRIORS[cfg.prior_kind], X, y)
    if b.prior is not None and n_ext:   # multi-climate training set: mixture OOD reference
        # balance sources 1:1, else the external set (smooth T/RH) swamps this node's own quantised DHT11 steps
        cut = int(len(y) * 0.8)
        own = X[n_ext:cut]
        ext = X[:n_ext][np.random.default_rng(0).permutation(n_ext)[:len(own)]]
        b.prior.fit_ood_mixture(np.vstack([ext, own]), X[cut:])

    # held-out part split in two: (A) estimate systematic error, (B) fit conformal.
    # Calibration data must be clean (no injected faults).
    out = run_frame(df, b, "adaptive").iloc[n_tr:]
    if truth_col:
        target = np.log(df[truth_col].values[n_tr:])
    else:  # reference-free consensus target
        target = np.log(out["idx_phys"]).rolling(31, center=True, min_periods=1).median().values
    resid = target - np.log(out["fused"].values)
    # calibrate on sigma without the time-since-anchor drift term: calibration data position
    # relative to the anchor differs from deployment, and that term is added on top afterwards
    sig = out["sigma_log_nodrift"].values
    h = len(resid) // 2
    b.sigma_sys = float(np.sqrt(max(0.0, np.mean(resid[:h] ** 2) - np.median(sig[:h] ** 2))))
    sig_tot = np.sqrt(sig[h:] ** 2 + b.sigma_sys ** 2)
    unscaled_cov = float(np.mean(np.abs(resid[h:]) < 1.645 * sig_tot))
    b.conformal = Conformal(cfg.alpha).fit(np.abs(resid[h:]) / sig_tot)
    return b, dict(r0=r0, n_external=n_ext, n_quiet=int(q.sum()), n_rows=len(df), sigma_sys=b.sigma_sys,
                   prior_skill=prior_skill, prior_admitted=b.prior is not None,
                   unscaled_cov90=unscaled_cov,
                   cf="refit" if refit is not None else "datasheet (T/RH range too small)",
                   beta=refit.beta.tolist() if refit is not None else None, q=b.conformal.q)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pb = sub.add_parser("bundle")
    pb.add_argument("--log", required=True)
    pb.add_argument("--out", default="models/bundle.joblib")
    pb.add_argument("--config")
    pb.add_argument("--clean", nargs=2, type=int, default=[0, 600], help="clean-air sample range")
    pb.add_argument("--truth-col")
    pd_ = sub.add_parser("decay")
    pd_.add_argument("--log", required=True)
    pd_.add_argument("--bundle", default="models/bundle.joblib")
    pd_.add_argument("--start", type=int, required=True)
    pd_.add_argument("--end", type=int, required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.log)
    if a.cmd == "bundle":
        cfg = Config.load(a.config) if a.config else Config()
        b, info = build_bundle(df, cfg, tuple(a.clean), a.truth_col)
        b.save(a.out)
        print(info)
    else:
        b = Bundle.load(a.bundle)
        out = run_frame(df, b, "refit_phys").iloc[a.start:a.end]
        lam, c0, r2 = fit_decay(out["millis"].values / 1000, out["fused"].values, b.cfg.c_bg)
        print(f"lambda = {lam:.3e} 1/s = {lam * 3600:.2f} ACH, C0 = {c0:.0f}, R2 = {r2:.3f}")
        b.cfg.lam = float(lam)
        b.save(a.bundle)
        print(f"saved lambda into {a.bundle}")


if __name__ == "__main__":
    main()
