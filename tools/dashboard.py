"""PHYS-TRUST Lite live dashboard.  streamlit run tools/dashboard.py

Sources:
  Live sensor   follows data/burnin.csv as the logger writes it (logger keeps COM8)
  CSV replay    replays a recorded log (replay start in hours)
  Simulator     synthetic day with injected faults (no hardware needed)
  Serial        reads the Arduino directly (stop the logger first: COM8 has one owner)
"""
import csv as csvmod
import datetime as dt
import os
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)
import viz  # noqa: E402
from phystrust import MODES, PhysTrust  # noqa: E402
from phystrust.evaluate import inject_suite  # noqa: E402
from phystrust.llm_explain import EventDetector, explain, summarize_row  # noqa: E402
from phystrust.pipeline import Bundle  # noqa: E402
from phystrust.reference import weather_now  # noqa: E402
from phystrust.simulate import test_day  # noqa: E402

TZ = "Asia/Kolkata"
BURNIN_START = pd.Timestamp("2026-09-25 06:22")
EVENTS_CSV = "data/events.csv"
DRAW_EVERY_S = 1.5  # screen refresh (1-2 s)

st.set_page_config(page_title="PHYS-TRUST Lite", page_icon="🌫️", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.2rem}
.badge{display:inline-block;padding:.35rem .8rem;border-radius:.5rem;font-weight:600;color:#fff;margin-right:.4rem}
.ok{background:#2e7d32}.warn{background:#ef6c00}.bad{background:#c62828}.info{background:#546e7a}
</style>""", unsafe_allow_html=True)

st.title("PHYS-TRUST Lite")
st.caption("Trust-adaptive, physics-informed air-quality estimation on one low-cost node "
           "(MQ135 + DHT11 + Arduino Uno) with an LSTM deep-ensemble prior. "
           "Index = relative CO₂-equivalent, not certified ppm.")

with st.sidebar:
    st.header("Data source")
    src = st.radio("Source", ["Live sensor", "CSV replay", "Simulator", "Serial"])
    csv = st.text_input("Log file", "data/burnin.csv")
    port = "COM8"
    if src == "Serial":
        try:
            from serial.tools import list_ports
            found = [p.device for p in list_ports.comports()]
        except Exception:
            found = []
        port = st.selectbox("Serial port", found or ["COM8"], help="Detected ports; the Arduino (CH340) was COM8")
        st.caption("Serial reads the board directly. The logger normally owns this port; "
                   "use **Live sensor** for the same real-time data without stopping the logger.")
    start_h = st.number_input("Replay start (hours into log)", 0.0, 1000.0, 3.0, 0.5) \
        if src == "CSV replay" else 0.0
    speed = st.slider("Replay speed (samples per frame)", 1, 120, 30) if src in ("CSV replay", "Simulator") else 1
    st.header("Model")
    bpath = st.text_input("Calibration bundle", "models/bundle.joblib")
    mode = st.selectbox("Estimator mode", list(MODES), index=list(MODES).index("adaptive"),
                        help="adaptive = full system; others are ablation baselines")
    window = st.slider("History shown (min)", 5, 60, 30) * 60
    use_llm = st.checkbox("Explain events with gemma3:4b (local Ollama)", False)
    if st.button("⟲ Re-zero now (window open, clean air)"):
        st.session_state["rezero"] = True
    run = st.toggle("▶ Run", False)

bundle = Bundle.load(bpath) if os.path.exists(bpath) else None
prior_name = type(bundle.prior).__name__ if bundle is not None and bundle.prior is not None else "none"
prior_label = {"LSTMPrior": "5-LSTM ensemble", "EnsemblePrior": "5-MLP ensemble", "TCNPrior": "5-TCN ensemble",
               "TransformerPrior": "5-Transformer ensemble"}.get(prior_name, prior_name)


# ------------------------------------------------------------------ helpers
def to_local(u):
    return pd.Timestamp(u, unit="s", tz="UTC").tz_convert(TZ).tz_localize(None)


def xaxis(d):
    if "unix" in d and d["unix"].notna().all():
        return list(pd.to_datetime(d["unix"], unit="s", utc=True).dt.tz_convert(TZ).dt.tz_localize(None))
    return (d["millis"] / 1000).tolist()


def load_events():
    if not os.path.exists(EVENTS_CSV):
        return []
    e = pd.read_csv(EVENTS_CSV)
    return [(to_local(u), lab) for u, lab in zip(e["unix"], e["label"])]


def log_event(unix, label):
    new = not os.path.exists(EVENTS_CSV)
    with open(EVENTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csvmod.writer(f)
        if new:
            w.writerow(["unix", "label"])
        w.writerow([unix, label])


def status(row):
    t = min(row["trust_raw"], row["trust_phys"])
    if t < viz.ALARM_THR:
        return "bad", "⚠ SENSOR FAULT — raw reading rejected"
    if row["trust_phys"] < viz.PHYS_LOW:
        return "warn", "Physics trust low (T/RH change)"
    return "ok", "Sensor trusted"


def aq_band(c):
    return ("Good", "ok") if c < 800 else (("Moderate", "warn") if c < 1200 else ("Poor", "bad"))


def serial_source():
    """Direct serial read with clear status messages instead of crashes: board missing, port busy
    (logger owns it), disconnect mid-stream (auto-retry), garbled lines (skipped)."""
    import serial
    while True:
        try:
            with serial.Serial(port, 115200, timeout=2) as ser:
                yield {"_status": ("info", f"Connected to {port}. The Uno resets on connect; data in ~2 s.")}
                while True:
                    p = ser.readline().decode(errors="ignore").strip().split(",")
                    if len(p) != 7 or not p[0].isdigit():
                        continue                           # header, partial or garbled line
                    try:
                        yield dict(millis=float(p[0]), adc_mean=float(p[1]), adc_std=float(p[2]),
                                   T=float(p[3]), RH=float(p[4]), dht_ok=int(float(p[5])), unix=time.time())
                    except ValueError:
                        continue
        except serial.SerialException as e:
            msg = str(e)
            if "PermissionError" in msg or "Access is denied" in msg:
                yield {"_status": ("error", f"{port} is busy: the logger owns it. Use **Live sensor** mode "
                                            "(same real-time data), or stop the logger to use Serial.")}
                return
            yield {"_status": ("warning", f"{port} not available (board unplugged?). Retrying every 3 s… "
                                          f"[{msg.split(':')[0]}]")}
            time.sleep(3)


def simulator_source():
    """Synthetic 2-h scenario with injected faults, generated with the LOADED calibration's physics
    (R0, curve a/b, background, load resistor) and Chennai-like indoor T/RH, so indices are on the
    same scale as the real node instead of an unrelated default sensor."""
    cfg = bundle.cfg if bundle is not None else None
    r0 = bundle.r0 if bundle is not None else None
    rl = bundle.cfg.rl_kohm if bundle is not None else 10.0
    sim = test_day(seed=int(time.time()) % 1000, cfg=cfg, r0_true=r0, t0=30.0, rh0=62.0)
    yield from inject_suite(sim, rl).to_dict("records")
    yield {"_status": ("info", "Simulated 2-hour scenario finished (switch Run off/on for a new one).")}


def source():
    if src == "Serial":
        yield from serial_source()
    elif src == "Simulator":
        yield from simulator_source()
    elif src == "CSV replay":
        df = pd.read_csv(csv)
        if "unix" in df:
            df = df[df["unix"] >= df["unix"].iloc[0] + start_h * 3600]
        yield from df.to_dict("records")
    else:  # Live sensor: warm up on recent history, then follow the file as the logger appends
        df = pd.read_csv(csv)
        for r in df.tail(window).to_dict("records"):
            r["_warm"] = True
            yield r
        cols = list(df.columns)
        with open(csv, encoding="utf-8") as f:
            f.seek(0, 2)
            while True:
                pos = f.tell()
                line = f.readline()
                if not line.endswith("\n"):  # nothing new, or logger mid-write: retry later
                    f.seek(pos)
                    time.sleep(0.5)
                    idle = time.time() - os.path.getmtime(csv)
                    if idle > 10 and int(idle) % 5 == 0:
                        yield {"_status": ("warning", f"No new sensor data for {idle / 60:.1f} min — check the "
                                                      "Arduino USB cable (logger keeps retrying).")}
                    continue
                p = line.strip().split(",")
                if len(p) == len(cols):
                    yield {c: float(v) for c, v in zip(cols, p)}


# ------------------------------------------------------------------ layout
if bundle is None:
    st.warning(f"No calibration bundle at `{bpath}`: using datasheet priors, no learned prior.")
head = st.empty()
cards = st.empty()
st.markdown("##### Estimate, 90 % band, sensor channels · red = sensor-fault alarm · dotted = logged events")
hero_ph = st.empty()
g1, g2 = st.columns([3, 2])
with g1:
    st.markdown("##### Trust per source (alarm threshold 30 %)")
    gauge_ph = st.empty()
with g2:
    st.markdown("##### Uncertainty breakdown (share of σ²_total)")
    unc_ph = st.empty()
st.markdown("##### Status timeline · green OK · amber physics trust low · red sensor-fault alarm")
strip_ph = st.empty()

t_trust, t_cross, t_events, t_valid, t_model, t_results = st.tabs(
    ["Trust over time", "Cross-sensitivity", "Event log", "✅ Validation", "🧠 Model & method", "📊 Results"])
with t_trust:
    trust_ph = st.empty()
with t_cross:
    st.caption("If the T/RH correction worked, the physics-corrected cloud would be flat against T and RH.")
    cross_ph = st.empty()
with t_events:
    ev_ph = st.empty()
    llm_ph = st.empty()

with t_valid:
    st.subheader("Live: DHT11 vs outdoor weather (Open-Meteo, Chennai)")
    st.caption("Indoor node vs outdoor air: expect an offset; the check is whether the node tracks the weather "
               "and stays physically plausible. Refreshed every 15 min.")
    live_val_ph = st.empty()
    st.subheader("Past data: DHT11 vs Open-Meteo, hourly")
    if os.path.exists("results/weather/stats.csv"):
        ws = pd.read_csv("results/weather/stats.csv").T.rename(columns={0: "value"})
        c1_, c2_ = st.columns([1, 2])
        c1_.dataframe(ws.round(3))
        if os.path.exists("results/weather/weather.png"):
            c2_.image("results/weather/weather.png")
    st.subheader("Reference-dataset validation: UCI Air Quality (MOX sensor vs certified CO analyser, 10 months)")
    if os.path.exists("results/uci/ablation.csv"):
        st.dataframe(pd.read_csv("results/uci/ablation.csv").round(3), hide_index=True)
    if os.path.exists("results/uci/uci_validation.png"):
        st.image("results/uci/uci_validation.png", caption="Three test weeks: analyser (dots), static calibration, "
                                                           "PHYS-TRUST adaptive with 90 % band, trust")

with t_model:
    st.subheader("Pipeline")
    st.markdown(f"""
| Layer | What it does | Code |
|---|---|---|
| L0 Signal | ADC → sensor resistance Rs = RL·(1023/ADC − 1), fault flags | `sensor_physics.py` |
| L1a Sensor physics | Power law C = a·(Rs/(R0·CF(T,RH)))^b, temperature/humidity correction | `sensor_physics.py` |
| L1b Process physics | Room mass balance dC/dt = S/V − λ(C − C_bg), plausibility envelope | `process_physics.py` |
| **L2 Learned prior — deep ensemble** | **{prior_label}**: 12-step sequence (60 s) of [Δlog C, T, RH] → next-step change; ensemble spread = epistemic uncertainty; Mahalanobis OOD score | `prior.py` |
| L3 Trust estimator | Innovation (NIS), rate plausibility, T/RH envelope, sensor lag, OOD → trust ∈ [0,1] per source | `trust.py` |
| L4 Fusion | Extended Kalman filter, state [C, drift]; noise R = R₀ / trust | `fusion.py` |
| L5 Uncertainty | σ²_total = q²(EKF + coefficient b + systematic) + drift·t; split-conformal 90 % band | `conformal.py` |
""")
    st.markdown("**Core equation:** R_s = R_s,0 / max(τ_s, ε) — the Kalman filter weights each source by its trust.")
    if bundle is not None:
        k = st.columns(5)
        k[0].metric("R0", f"{bundle.r0:.0f} kΩ")
        k[1].metric("σ_sys", f"{bundle.sigma_sys:.3f}")
        k[2].metric("Conformal q", f"{bundle.conformal.q:.2f}")
        k[3].metric("λ", f"{bundle.cfg.lam * 3600:.2f} /h")
        k[4].metric("Learned prior", prior_label)

with t_results:
    st.subheader("Synthetic benchmark (frozen; true sensor exponent swept ±σ, 5 test seeds)")
    r = st.columns(4)
    r[0].metric("Error vs fixed-weight fusion", "−9 %", "0.078 vs 0.085 RMSE log, p = 0.0008")
    r[1].metric("90 % band, clean / local faults", "100 % / 100 %")
    r[2].metric("Sensor-fault detection F1", "0.79", "latency 1–4 s")
    r[3].metric("Attribution to sensor", "20 / 20")
    if os.path.exists("results/prior_compare.csv"):
        pc = pd.read_csv("results/prior_compare.csv").groupby("prior").mean(numeric_only=True)
        st.markdown("**Learned prior: MLP vs LSTM ensemble (5 seeds)**")
        st.dataframe(pc[["rmse_log", "prior_1step_rmse", "picp_clean", "ms_per_step"]].round(4))
    if os.path.exists("results/adaptive_trace.png"):
        st.image("results/adaptive_trace.png", caption="Synthetic test day: fused estimate, band, trust")
    st.subheader("Real hardware session (2026-09-25)")
    r = st.columns(3)
    r[0].metric("Perfume interference", "flagged at onset", "fused held ≈ 917 while raw → 10⁶")
    r[1].metric("Hot air +8.5 °C / −18 %RH", "≈ 3 % corrected", "datasheet correction too weak", "inverse")
    r[2].metric("Undisturbed coverage", "62 %", "target 90 %", "inverse")
    if os.path.exists("results/session/session.png"):
        st.image("results/session/session.png", caption="Real node: index, trust, T/RH with event markers")
    st.info("Limits: drift unobservable within a day on one node (ABC mitigates); noise faults poorly "
            "detected; short burn-in; no reference instrument, so no absolute ppm claim.")

# ------------------------------------------------------------------ run loop
if not run:
    head.info("Choose a source in the sidebar and switch **▶ Run** on. Live sensor follows the running logger.")
else:
    pt = PhysTrust(bundle, mode)
    buf = deque(maxlen=window)
    det = EventDetector(thr=viz.ALARM_THR)
    notes = deque(maxlen=10)
    events = load_events()
    last_draw, drawn = 0.0, 0
    for i, s in enumerate(source()):
        if "_status" in s:                      # source-level message (connection, end of data)
            kind, msg = s["_status"]
            getattr(head, kind)(msg)
            if kind == "error":
                break
            continue
        warm = bool(s.pop("_warm", False))
        row = pt.step(s)
        row["unix"] = s.get("unix", np.nan)
        buf.append(row)
        if st.session_state.get("rezero") and not warm:
            pt.rezero()
            st.session_state["rezero"] = False
            if np.isfinite(row["unix"]):
                log_event(row["unix"], "re-zero")
                events.append((to_local(row["unix"]), "re-zero"))
        if not warm and use_llm:
            for ev in det(row):
                notes.appendleft(f"`{to_local(row['unix']).strftime('%H:%M:%S') if np.isfinite(row['unix']) else ''}` "
                                 f"{ev}  \n> 🤖 {explain(ev, summarize_row(row))}")
        if warm or (src in ("CSV replay", "Simulator") and i % speed):
            continue
        if time.time() - last_draw < (DRAW_EVERY_S if src != "Live sensor" else 0.9):
            continue
        last_draw, drawn = time.time(), drawn + 1

        d = pd.DataFrame(buf)
        xb, band = xaxis(d), d[["lo", "hi"]]
        if "unix" in d and d["unix"].notna().all():  # break lines at logger gaps (> 5 s)
            gap = d["unix"].diff() > 5
            if gap.any():
                pad = d[gap].copy()
                pad["unix"] -= 0.5
                for c in d.columns:
                    if c != "unix" and pd.api.types.is_float_dtype(d[c]):
                        pad[c] = np.nan
                d = pd.concat([d, pad]).sort_values("unix").reset_index(drop=True)
        x = xaxis(d)
        dn = d.dropna(subset=["fused"]).reset_index(drop=True)
        xn = xaxis(dn)

        # header + 4 metric cards
        cls, txt = status(row)
        aq, aqc = aq_band(row["fused"])
        scls = {"TRUSTED": "ok", "UNCERTAIN": "warn", "CALIBRATION_NEEDED": "warn", "OOD": "warn",
                "PHYSICS_MISMATCH": "bad", "SENSOR_FAULT": "bad"}.get(row.get("state"), "info")
        alert = " · 🔔 human alert" if row.get("alert_human") else ""
        head.markdown(f'<span class="badge {scls}">State: {row.get("state", "?")}</span>'
                      f'<span class="badge {cls}">{txt}</span><span class="badge {aqc}">Air quality: {aq} '
                      f'(relative)</span><span class="badge info">{src} · {mode} · prior: {prior_label}</span>'
                      f'<br><small>Action: <b>{row.get("action", "")}</b> — {row.get("action_note", "")}{alert} · '
                      f'P(OOD) {row.get("p_ood", float("nan")):.2f} · calibration confidence '
                      f'{row.get("calib_confidence", float("nan")):.2f} · compute tier {row.get("compute_tier", "")} · '
                      f'calibration {row.get("calib_version", "")}</small>', unsafe_allow_html=True)
        with cards.container():
            k = st.columns(4)
            back = dn["fused"].iloc[-61] if len(dn) > 61 else dn["fused"].iloc[0]
            k[0].metric("Fused index", f"{row['fused']:.0f}", f"{row['fused'] - back:+.0f} vs 1 min ago",
                        delta_color="inverse")
            half = (row["hi"] - row["lo"]) / 2
            k[1].metric("90 % band", f"± {half:.0f}", f"± {100 * half / row['fused']:.0f} %", delta_color="off")
            now_local = to_local(row["unix"]) if np.isfinite(row["unix"]) else None
            bh = (now_local - BURNIN_START).total_seconds() / 3600 if now_local is not None else float("nan")
            k[2].metric("Burn-in", f"{bh:.1f} h" if np.isfinite(bh) else "–", "since 25 Sep 06:22", delta_color="off")
            k[3].metric("Since last re-anchor", f"{row['t_since_anchor'] / 60:.0f} min",
                        f"ABC events {row['abc_events']}", delta_color="off")

        rng = (x[0], x[-1])
        ev_in = [(t, lab) for t, lab in events if isinstance(rng[0], pd.Timestamp) and rng[0] <= t <= rng[1]]
        hero_ph.plotly_chart(viz.hero(x, d, xb, band, ev_in), use_container_width=True,
                             key=f"h{drawn}", config=viz.NOBAR)
        gauge_ph.plotly_chart(viz.gauges(row), use_container_width=True, key=f"g{drawn}", config=viz.NOBAR)
        unc_ph.plotly_chart(viz.uncertainty_bar(row), use_container_width=True, key=f"u{drawn}", config=viz.NOBAR)
        strip_ph.plotly_chart(viz.alarm_strip(xn, dn), use_container_width=True, key=f"s{drawn}", config=viz.NOBAR)
        trust_ph.plotly_chart(viz.trust_area(xn, dn), use_container_width=True, key=f"t{drawn}", config=viz.NOBAR)
        cross_ph.plotly_chart(viz.cross_sensitivity(dn), use_container_width=True, key=f"c{drawn}", config=viz.NOBAR)
        tab = viz.event_table(xn, dn)
        if len(tab):
            ev_ph.dataframe(tab, hide_index=True, use_container_width=True)
        else:
            ev_ph.info("No trust events in the shown history.")
        wn = weather_now() if src in ("Live sensor", "Serial") else None
        if wn is not None:
            with live_val_ph.container():
                v = st.columns(4)
                v[0].metric("DHT11 temperature", f"{row['T']:.1f} °C")
                v[1].metric("Outdoor (Open-Meteo)", f"{wn['T_ref']:.1f} °C", f"node − outdoor {row['T'] - wn['T_ref']:+.1f} °C",
                            delta_color="off")
                v[2].metric("DHT11 humidity", f"{row['RH']:.0f} %")
                v[3].metric("Outdoor humidity", f"{wn['RH_ref']:.0f} %", f"node − outdoor {row['RH'] - wn['RH_ref']:+.0f} %",
                            delta_color="off")
                st.caption(f"Open-Meteo time {wn['time']:%H:%M}")
        elif src not in ("Live sensor", "Serial"):
            live_val_ph.info("Live weather comparison runs in Live sensor / Serial mode.")
        if use_llm:
            llm_ph.markdown("\n\n".join(notes) if notes else "_no LLM notes yet_")
        if src in ("CSV replay", "Simulator"):
            time.sleep(0.02)
