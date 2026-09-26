"""One-day hardware session: calibrate from a clean-air window, fit lambda, analyse demos.

python tools/session_report.py --clean 11:20 11:35 --decay 12:05 --swab 12:30 12:32 --hot 12:40 12:42
Times are HH:MM today (local). Writes models/bundle.joblib and results/session/*.
"""
import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phystrust.calibrate import build_bundle  # noqa: E402
from phystrust.pipeline import run_frame  # noqa: E402
from phystrust.process_physics import fit_decay  # noqa: E402

OUT = "results/session"


def ts(hhmm, day):
    h, m = map(int, hhmm.split(":"))
    return dt.datetime.combine(day, dt.time(h, m)).timestamp()


def idx(df, t):
    return int(np.searchsorted(df["unix"].values, t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="data/burnin.csv")
    ap.add_argument("--clean", nargs=2, required=True)
    ap.add_argument("--decay", help="time people left the closed room")
    ap.add_argument("--decay-min", type=int, default=20)
    ap.add_argument("--swab", nargs="*", default=[])
    ap.add_argument("--hot", nargs=2)
    ap.add_argument("--history-h", type=float, default=2.0, help="hours before clean window used for training")
    ap.add_argument("--date", help="YYYY-MM-DD, default today")
    ap.add_argument("--external", action="store_true", help="add DALTON external data (data/external/dalton) to prior training")
    ap.add_argument("--train-until", help="HH:MM: train/calibrate on data up to here instead of the clean-window end "
                                          "(only when no demo events happened before it)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    day = dt.date.fromisoformat(a.date) if a.date else dt.date.today()

    df = pd.read_csv(a.log)
    df = df[(df["dht_ok"] == 1) & (df["RH"] >= 1) & (df["T"] >= 1)].reset_index(drop=True)  # drop corrupt DHT reads
    c0, c1 = ts(a.clean[0], day), ts(a.clean[1], day)
    df = df[df["unix"] >= c0 - a.history_h * 3600].reset_index(drop=True)
    cw = (idx(df, c0), idx(df, c1))
    print(f"rows {len(df)}; clean window rows {cw} ({cw[1] - cw[0]} samples)")

    # ---- calibration (split 0.7 inside build_bundle: clean window must precede held-out part)
    # only data up to the end of the clean window: later demos (swab, hot water) must not
    # leak into sigma_sys / conformal calibration
    end = idx(df, ts(a.train_until, day)) if a.train_until else cw[1]
    ext = None
    if a.external:
        from phystrust.external import dalton_segments
        ext = dalton_segments()
        print(f"external: {len(ext)} DALTON segments, {sum(len(e[0]) for e in ext)} samples")
    b, info = build_bundle(df.iloc[:max(end, cw[1])].reset_index(drop=True), clean_window=cw, external=ext)
    b.save("models/bundle.joblib")
    print("calibration:", {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in info.items() if k != "beta"})

    # ---- lambda decay (ASTM E741) on physics-corrected estimate
    if a.decay:
        d0 = idx(df, ts(a.decay, day))
        d1 = d0 + a.decay_min * 60
        o = run_frame(df, b, "refit_phys").iloc[d0:d1]
        lam, cstart, r2 = fit_decay(o["millis"].values / 1000.0, o["fused"].values, b.cfg.c_bg)
        print(f"decay: lambda = {lam:.3e} 1/s = {lam * 3600:.2f} ACH, C0 = {cstart:.0f}, R2 = {r2:.3f}")
        if r2 > 0.5 and lam > 0:
            b.cfg.lam = float(lam)
            b.save("models/bundle.joblib")
            print("lambda saved into models/bundle.joblib")
        else:
            print("lambda NOT saved (poor fit); default kept")

    # ---- full adaptive run for demos
    out = run_frame(df, b, "adaptive")
    out["clock"] = pd.to_datetime(df["unix"], unit="s").dt.tz_localize("UTC").dt.tz_convert(
        dt.datetime.now().astimezone().tzinfo).dt.strftime("%H:%M:%S")
    out.to_csv(f"{OUT}/trace.csv", index=False)

    rows = []
    for t in a.swab:
        k = idx(df, ts(t, day))
        w = out.iloc[max(k - 30, 0):k + 120]
        low = np.flatnonzero(np.minimum(w.trust_raw, w.trust_phys).values < 0.3)
        rows.append(dict(event=f"swab {t}", raw_peak=w.idx_raw.max(), fused_peak=w.fused.max(),
                         raw_over_fused=w.idx_raw.max() / w.fused.max(),
                         min_trust_raw=w.trust_raw.min(), min_trust_phys=w.trust_phys.min(),
                         flagged=len(low) > 0, first_flag_s=(low[0] - 30) if len(low) else np.nan,
                         max_width_log=np.log(w.hi / w.lo).max()))
    if a.hot:
        h0, h1 = idx(df, ts(a.hot[0], day)), idx(df, ts(a.hot[1], day))
        w = out.iloc[h0:h1 + 120]
        base = out.iloc[max(h0 - 120, 0):h0]
        rows.append(dict(event=f"hot air {a.hot[0]}-{a.hot[1]}",
                         dT=w["T"].max() - base["T"].mean(), dRH=w["RH"].min() - base["RH"].mean(),
                         raw_shift_log=np.log(w.idx_raw.median() / base.idx_raw.median()),
                         phys_shift_log=np.log(w.idx_phys.median() / base.idx_phys.median()),
                         fused_shift_log=np.log(w.fused.median() / base.fused.median()),
                         min_trust_raw=w.trust_raw.min(), min_trust_phys=w.trust_phys.min(),
                         max_width_log=np.log(w.hi / w.lo).max()))
    if rows:
        ev = pd.DataFrame(rows)
        ev.to_csv(f"{OUT}/events.csv", index=False)
        print(ev.round(3).T.to_string())

    # ---- coverage vs reference-free consensus, after calibration window (optimistic, no reference)
    post = out.iloc[cw[1]:]
    cons = np.exp(np.log(post.idx_phys).rolling(31, center=True, min_periods=1).median())
    ins = (cons >= post.lo) & (cons <= post.hi)
    print(f"coverage vs consensus after clean window: {ins.mean():.3f} (n={len(post)}; "
          f"consensus target, not a reference instrument)")
    _plot(out, a, day, df)


def _plot(out, a, day, df):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    x = pd.to_datetime(df["unix"], unit="s", utc=True).dt.tz_convert(dt.datetime.now().astimezone().tzinfo).dt.tz_localize(None)  # local time
    ax[0].fill_between(x, out.lo, out.hi, alpha=0.3, label="90% band")
    ax[0].plot(x, out.idx_raw, lw=0.6, label="raw")
    ax[0].plot(x, out.idx_phys, lw=0.6, label="physics")
    ax[0].plot(x, out.fused, lw=1.5, label="fused")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("CO2-eq index")
    ax[0].legend(fontsize=8)
    for ch in ("raw", "phys", "prior"):
        ax[1].plot(x, out[f"trust_{ch}"], label=ch)
    ax[1].set_ylabel("trust")
    ax[1].legend(fontsize=8)
    ax[2].plot(x, df["T"], label="T")
    ax[2].plot(x, df["RH"], label="RH")
    ax[2].legend(fontsize=8)
    marks = [("clean", t) for t in a.clean] + [("swab", t) for t in a.swab]
    marks += [("decay", a.decay)] if a.decay else []
    marks += [("hot", t) for t in a.hot] if a.hot else []
    for lab, t in marks:
        tt = pd.Timestamp(ts(t, day), unit="s", tz="UTC").tz_convert(dt.datetime.now().astimezone().tzinfo).tz_localize(None)
        for axx in ax:
            axx.axvline(tt, color="k", lw=0.5, ls="--")
        ax[0].text(tt, ax[0].get_ylim()[1], lab, fontsize=7, rotation=90, va="top")
    fig.tight_layout()
    fig.savefig(f"{OUT}/session.png", dpi=120)
    print(f"plot: {OUT}/session.png")


if __name__ == "__main__":
    main()
