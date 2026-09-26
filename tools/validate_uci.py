"""Validation against real reference data: UCI Air Quality (De Vito et al., 2008).

A tin-oxide MOX sensor (PT08.S1, CO-targeted; same family as the MQ135) logged hourly for a year next to a
certified CO analyser (CO(GT), mg/m^3). The full PHYS-TRUST pipeline runs on the MOX signal and is scored
against the analyser.

Mapping onto the pipeline (maths unchanged):
  one step = one hour (millis = k*1000), Rs := 1 / sensor response (response rises with gas)
  field calibration (as used by existing low-cost networks), fitted on days 0-30:
      log CO = c0 + b*log Rs + g1*T + g2*RH + g3*T*RH
  -> power law exponent b, a = exp(c0 + g.x_ref), R0 = 1, CF(T,RH) = exp(-(g/b).(x - x_ref))
  raw channel      = static calibration without T/RH
  physics channel  = static calibration with T/RH  (= the standard field-calibration baseline)
  days 30-60: LSTM prior training + sigma_sys / conformal calibration; days 60+: test (~10 months)

python tools/validate_uci.py  -> results/uci/*
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phystrust import faults  # noqa: E402
from phystrust.config import Config  # noqa: E402
from phystrust.conformal import Conformal  # noqa: E402
from phystrust.evaluate import alarm_old, detection  # noqa: E402
from phystrust.pipeline import MODES, Bundle, run_frame  # noqa: E402
from phystrust.prior import LSTMPrior, admission_test  # noqa: E402
from phystrust.sensor_physics import RefitCF, rs_to_adc  # noqa: E402

OUT = "results/uci"
CAL_D, TRAIN_D = 30, 60


def load():
    d = pd.read_csv("data/uci/AirQualityUCI.csv", sep=";", decimal=",").dropna(how="all", axis=1).dropna(how="all")
    d = d.replace(-200, np.nan)
    d["time"] = pd.to_datetime(d["Date"] + " " + d["Time"], format="%d/%m/%Y %H.%M.%S")
    d = d.dropna(subset=["PT08.S1(CO)", "T", "RH"]).reset_index(drop=True)
    d["rs"] = 1e6 / d["PT08.S1(CO)"]
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load()
    day = (d["time"] - d["time"].iloc[0]).dt.total_seconds() / 86400
    cal, train = day < CAL_D, (day >= CAL_D) & (day < TRAIN_D)
    test = day >= TRAIN_D
    print(f"rows {len(d)}  calibration {cal.sum()}  train/val {train.sum()}  test {test.sum()}  "
          f"({d.time[test].iloc[0]:%Y-%m-%d} .. {d.time.iloc[-1]:%Y-%m-%d})")

    # ---- field calibration on days 0-30 (rows with reference)
    m = cal & d["CO(GT)"].notna() & (d["CO(GT)"] > 0)
    X = np.column_stack([np.ones(m.sum()), np.log(d.rs[m]), d["T"][m], d.RH[m], d["T"][m] * d.RH[m]])
    y = np.log(d["CO(GT)"][m])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    c0, b, g = coef[0], coef[1], coef[2:]
    resid = y - X @ coef
    print(f"field calibration: b = {b:.3f}, resid sd = {resid.std():.3f} (log), n = {m.sum()}")
    t_ref, rh_ref = RefitCF.T_REF, RefitCF.RH_REF
    x_ref = np.array([t_ref, rh_ref, t_ref * rh_ref])
    refit = RefitCF(beta=np.r_[0.0, -g / b], t_range=(float(d["T"][cal].quantile(.01)), float(d["T"][cal].quantile(.99))),
                    rh_range=(float(d.RH[cal].quantile(.01)), float(d.RH[cal].quantile(.99))))
    co_bg = float(d["CO(GT)"][cal].quantile(0.05))

    rl = float(d.rs.median())
    cfg = Config(a=float(np.exp(c0 + g @ x_ref)), b=float(b), sigma_b=0.0, c_ref=co_bg, c_bg=co_bg, r0_kohm=1.0,
                 rl_kohm=rl, lam=0.3, lam_max=2.0, rise_max=3.0, rate_win=3,           # per hour
                 q_c=0.6, q_ld=2e-4, r_raw=resid.std() ** 2 * 1.5, r_phys=resid.std() ** 2,
                 recover_s=6, lag_win=2, dtdt_thr=3.0, drhdt_thr=10.0, cf_dis_thr=1e9,
                 abc_enabled=False, prior_kind="lstm")
    df = pd.DataFrame(dict(millis=np.arange(len(d)) * 1000.0, adc_mean=rs_to_adc(d.rs, rl), adc_std=1.0,
                           T=d["T"], RH=d["RH"], dht_ok=1, fault=0, fault_kind=""))
    b_ = Bundle(cfg, 1.0, refit)

    # ---- tune process noise q_c on the validation period only (days 30-60), never on test
    ref_all = d["CO(GT)"].values
    vi = np.flatnonzero(train)
    vok = np.isfinite(ref_all[vi]) & (ref_all[vi] > 0)
    best = None
    for qc in (0.6, 1.5, 3.0, 6.0, 12.0):
        b_.cfg.q_c = qc
        o = run_frame(df.iloc[:vi[-1] + 1], b_, "refit_phys").iloc[vi]
        e = np.sqrt(np.mean((np.log(o["fused"].values[vok]) - np.log(ref_all[vi][vok])) ** 2))
        print(f"  q_c={qc:5.1f}  validation rmse_log={e:.3f}")
        best = (e, qc) if best is None or e < best[0] else best
    b_.cfg.q_c = cfg.q_c = best[1]
    print(f"chosen q_c = {best[1]} (validation)")

    # ---- LSTM prior on days 30-60 (fused physics-channel history)
    tr_idx = np.flatnonzero(train)
    base = run_frame(df.iloc[:tr_idx[-1] + 1], b_, "refit_phys").iloc[tr_idx]
    Xp, yp = LSTMPrior().dataset(np.log(base["fused"].values), df["T"].values[tr_idx], df["RH"].values[tr_idx])
    b_.prior, skill = admission_test(LSTMPrior, Xp, yp, epochs=20)
    print(f"LSTM prior admission (1-step skill vs persistence) = {skill:.3f} -> "
          f"{'passed' if b_.prior is not None else 'REJECTED'}")
    # stage 2: fused-level admission on the held-out end of the validation period (never the test)
    if b_.prior is not None:
        hv = tr_idx[int(len(tr_idx) * 0.8):]
        hok = np.isfinite(ref_all[hv]) & (ref_all[hv] > 0)

        def val_err(bundle):
            o = run_frame(df.iloc[:hv[-1] + 1], bundle, "adaptive").iloc[hv]
            return np.sqrt(np.mean((np.log(o["fused"].values[hok]) - np.log(ref_all[hv][hok])) ** 2))
        with_p = val_err(b_)
        keep = b_.prior
        b_.prior = None
        without_p = val_err(b_)
        b_.prior = keep if with_p < without_p else None
        print(f"fused-level admission: val rmse with prior {with_p:.3f} vs without {without_p:.3f} -> "
              f"{'prior ADMITTED' if b_.prior is not None else 'prior REJECTED (disabled)'}")

    # ---- sigma_sys / conformal on days 30-60 against the reference (half A / half B)
    out = run_frame(df.iloc[:tr_idx[-1] + 1], b_, "adaptive").iloc[tr_idx]
    truth = d["CO(GT)"].values[tr_idx]
    ok = np.isfinite(truth) & (truth > 0)
    r = (np.log(truth) - np.log(out["fused"].values))[ok]
    s0 = out["sigma_log_nodrift"].values[ok]
    h = len(r) // 2
    b_.sigma_sys = float(np.sqrt(max(0.0, np.mean(r[:h] ** 2) - np.median(s0[:h] ** 2))))
    b_.conformal = Conformal(0.1).fit(np.abs(r[h:]) / np.sqrt(s0[h:] ** 2 + b_.sigma_sys ** 2))
    b_.save(f"{OUT}/bundle_uci.joblib")
    print(f"sigma_sys = {b_.sigma_sys:.3f}, conformal q = {b_.conformal.q:.2f}")

    # ---- test: all ablation modes, clean
    rows, traces = [], {}
    ref = d["CO(GT)"].values
    for mode in MODES:
        o = run_frame(df, b_, mode)
        traces[mode] = o
        tm = test.values & np.isfinite(ref) & (ref > 0)
        e = np.log(o["fused"].values[tm]) - np.log(ref[tm])
        ins = (ref[tm] >= o["lo"].values[tm]) & (ref[tm] <= o["hi"].values[tm])
        lin = o["fused"].values[tm] - ref[tm]
        r2 = 1 - np.sum(lin ** 2) / np.sum((ref[tm] - ref[tm].mean()) ** 2)
        rows.append(dict(mode=mode, rmse_log=np.sqrt(np.mean(e ** 2)), bias_log=e.mean(),
                         mae_mg_m3=np.mean(np.abs(lin)), r2=r2, picp90=ins.mean(),
                         width_log=np.mean(np.log(o["hi"].values[tm] / o["lo"].values[tm]))))
    # direct static calibration (what existing low-cost networks do: fixed regression, no filter)
    tm = test.values & np.isfinite(ref) & (ref > 0)
    lr = np.log(d.rs.values)
    for name, cols in (("static_formula_noTRH", [lr]),
                       ("static_formula_TRH", [lr, d["T"].values, d["RH"].values, d["T"].values * d["RH"].values])):
        Xc = np.column_stack([np.ones(m.sum())] + [c[m.values] for c in cols])
        cc, *_ = np.linalg.lstsq(Xc, np.log(d["CO(GT)"].values[m.values]), rcond=None)
        Xt = np.column_stack([np.ones(tm.sum())] + [c[tm] for c in cols])
        pred = np.exp(Xt @ cc)
        e = np.log(pred) - np.log(ref[tm])
        rows.append(dict(mode=name, rmse_log=np.sqrt(np.mean(e ** 2)), bias_log=e.mean(),
                         mae_mg_m3=np.mean(np.abs(pred - ref[tm])),
                         r2=1 - np.sum((pred - ref[tm]) ** 2) / np.sum((ref[tm] - ref[tm].mean()) ** 2),
                         picp90=np.nan, width_log=np.nan))
    tab = pd.DataFrame(rows).set_index("mode").round(4)
    print(tab.to_string())
    tab.to_csv(f"{OUT}/ablation.csv")

    # ---- monthly drift view (static field calibration vs adaptive)
    mon = []
    for mode in ("refit_phys", "fixed", "adaptive"):
        o = traces[mode]
        f = pd.DataFrame(dict(month=d["time"].dt.to_period("M").astype(str), ref=ref, fused=o["fused"],
                              lo=o["lo"], hi=o["hi"], test=test))
        f = f[f.test & f.ref.gt(0)]
        g_ = f.groupby("month").apply(lambda x: pd.Series(dict(
            rmse_log=np.sqrt(np.mean((np.log(x.fused) - np.log(x.ref)) ** 2)),
            bias_log=np.mean(np.log(x.fused) - np.log(x.ref)),
            picp90=np.mean((x.ref >= x.lo) & (x.ref <= x.hi)), n=len(x))), include_groups=False)
        g_["mode"] = mode
        mon.append(g_.reset_index())
    mon = pd.concat(mon)
    mon.to_csv(f"{OUT}/monthly.csv", index=False)
    print(mon.pivot(index="month", columns="mode", values="rmse_log").round(3).to_string())

    # ---- fault injection on real data (test period): spike, stuck, noise
    fdf = df.copy()
    t0 = int(np.flatnonzero(test)[0])
    for k in range(t0 + 200, len(fdf) - 300, 700):
        fdf = faults.spike(fdf, k, dur=3, depth=0.3, tau_up=1.0, tau_down=2.0, tail=6, rl=rl)
        fdf = faults.stuck(fdf, k + 250, k + 260)
        fdf = faults.noise(fdf, k + 450, k + 470, factor=4.0, base_sd=0.05, rl=rl)
    fo = run_frame(fdf, b_, "adaptive")
    det = detection(fo.iloc[t0:], fdf.iloc[t0:], 0.3, alarm_old)
    det = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in det.items()}
    print("fault detection on real data (test period):", det)
    pd.DataFrame([det]).to_csv(f"{OUT}/detection.csv", index=False)

    o = traces["adaptive"].assign(time=d["time"], ref=ref)
    o.to_csv(f"{OUT}/adaptive_trace.csv", index=False)
    _plot(o, traces["refit_phys"], test, f"{OUT}/uci_validation.png")


def _plot(o, stat, test, path):
    import matplotlib.pyplot as plt
    w = o[test.values].iloc[2000:2000 + 24 * 21]           # three test weeks
    s = stat[test.values].iloc[2000:2000 + 24 * 21]
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax[0].fill_between(w.time, w.lo, w.hi, alpha=0.25, label="90 % band (adaptive)")
    ax[0].plot(w.time, w.ref, "k.", ms=3, label="reference analyser CO(GT)")
    ax[0].plot(w.time, s.fused, lw=0.8, label="static field calibration")
    ax[0].plot(w.time, w.fused, lw=1.4, label="PHYS-TRUST adaptive")
    ax[0].set_ylabel("CO mg/m³")
    ax[0].legend(fontsize=8)
    for ch in ("raw", "phys", "prior"):
        ax[1].plot(w.time, w[f"trust_{ch}"], label=ch)
    ax[1].set_ylabel("trust")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
