"""PHYS-TRUST Lite API for the React dashboard.

    python -m uvicorn server.app:app --port 8600      (from the project root)

WebSocket /ws streams estimator rows from one source:
    live     follows data/burnin.csv as the logger appends (logger keeps COM8)
    replay   replays a recorded log from `start_h` hours in, `speed` samples per frame
    sim      synthetic 2-h day with injected faults, generated with the loaded calibration's physics
    serial   reads the Arduino directly (stop the logger first: COM8 has one owner)
Client -> server messages: {"cmd": "rezero"}, {"cmd": "event", "label": "..."}.
REST: /api/meta (bundle + stored results), /api/weather (Open-Meteo now), /api/events.
The built frontend (web/dist) is served at /.
"""
import asyncio
import csv as csvmod
import math
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from phystrust import MODES, PhysTrust  # noqa: E402
from phystrust.evaluate import inject_suite  # noqa: E402
from phystrust.llm_explain import explain  # noqa: E402
from phystrust.pipeline import Bundle  # noqa: E402
from phystrust.reference import weather_now  # noqa: E402
from phystrust.simulate import test_day  # noqa: E402

LOG_CSV, EVENTS_CSV, BUNDLE = "data/burnin.csv", "data/events.csv", "models/bundle.joblib"
PRIOR_LABEL = {"LSTMPrior": "5-LSTM ensemble", "EnsemblePrior": "5-MLP ensemble", "TCNPrior": "5-TCN ensemble",
               "TransformerPrior": "5-Transformer ensemble"}
KEEP = ("millis", "unix", "T", "RH", "adc", "idx_raw", "idx_phys", "raw_c", "phys_c", "idx_prior", "fused", "lo", "hi", "drift",
        "drift_anchor", "trust_raw", "trust_phys", "trust_prior", "adc_fault", "abc_events", "t_since_anchor", "var_ekf", "var_b",
        "var_sys", "var_drift", "q_conf", "state", "action", "action_note", "alert_human", "compute_tier",
        "p_ood", "p_ood_smooth", "ood_cause", "calib_confidence", "calib_version", "node_id", "fault")

app = FastAPI(title="PHYS-TRUST Lite")


@app.middleware("http")
async def no_cache_html(request, call_next):
    """index.html must never be cached: asset names change on every frontend build."""
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.endswith(".html"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp
_bundle = {"mtime": None, "b": None}


def bundle():
    """Reload the calibration bundle when the file changes (recalibration while the server runs)."""
    if not os.path.exists(BUNDLE):
        return None
    m = os.path.getmtime(BUNDLE)
    if m != _bundle["mtime"]:
        _bundle.update(mtime=m, b=Bundle.load(BUNDLE))
    return _bundle["b"]


def clean(row):
    out = {}
    for k in KEEP:
        v = row.get(k)
        if isinstance(v, (np.floating, float)):
            v = float(v)
            v = round(v, 5) if math.isfinite(v) else None
        elif isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.bool_):
            v = bool(v)
        if v is not None:
            out[k] = v
    return out


def read_csv_safe(path):
    return pd.read_csv(path) if os.path.exists(path) else None


# ------------------------------------------------------------------ sources (blocking generators)
def live_source(stop, warm_n):
    df = pd.read_csv(LOG_CSV)
    for r in df.tail(warm_n).to_dict("records"):
        r["_warm"] = True
        yield r
    yield {"_warm_done": True}
    cols = list(df.columns)
    with open(LOG_CSV, encoding="utf-8") as f:
        f.seek(0, 2)
        last_warn = 0.0
        while not stop.is_set():
            pos = f.tell()
            line = f.readline()
            if not line.endswith("\n"):          # nothing new, or logger mid-write
                f.seek(pos)
                time.sleep(0.5)
                idle = time.time() - os.path.getmtime(LOG_CSV)
                if idle > 10 and time.time() - last_warn > 5:
                    last_warn = time.time()
                    yield {"_status": ("warning", f"No new sensor data for {idle / 60:.1f} min. Check the "
                                                  "Arduino USB cable; the logger keeps retrying.")}
                continue
            p = line.strip().split(",")
            if len(p) == len(cols):
                try:
                    yield {c: float(v) for c, v in zip(cols, p)}
                except ValueError:
                    continue


def replay_source(stop, start_h):
    df = pd.read_csv(LOG_CSV)
    if "unix" in df:
        df = df[df["unix"] >= df["unix"].iloc[0] + start_h * 3600]
    yield from df.to_dict("records")
    yield {"_status": ("info", "End of recorded log.")}


def sim_source(stop):
    b = bundle()
    cfg = b.cfg if b is not None else None
    # centre the synthetic room on the calibration log's T/RH, so a quiet simulated room is in-distribution
    t0, rh0 = 30.0, 62.0
    if os.path.exists(LOG_CSV):
        d = pd.read_csv(LOG_CSV).tail(6 * 3600)            # recent conditions, like the current calibration
        t0, rh0 = float(d["T"].median()), float(d["RH"].median())
    sim = test_day(seed=int(time.time()) % 1000, cfg=cfg, r0_true=b.r0 if b else None, t0=t0, rh0=rh0)
    yield from inject_suite(sim, b.cfg.rl_kohm if b else 10.0).to_dict("records")
    yield {"_status": ("info", "Simulated 2-hour scenario finished. Restart for a new one.")}


def serial_source(stop, port):
    import serial
    while not stop.is_set():
        try:
            with serial.Serial(port, 115200, timeout=2) as ser:
                yield {"_status": ("info", f"Connected to {port}. The Uno resets on connect; data in about 2 s.")}
                while not stop.is_set():
                    p = ser.readline().decode(errors="ignore").strip().split(",")
                    if len(p) != 7 or not p[0].isdigit():
                        continue
                    try:
                        yield dict(millis=float(p[0]), adc_mean=float(p[1]), adc_std=float(p[2]), T=float(p[3]),
                                   RH=float(p[4]), dht_ok=int(float(p[5])), unix=time.time())
                    except ValueError:
                        continue
        except serial.SerialException as e:
            msg = str(e)
            if "PermissionError" in msg or "Access is denied" in msg:
                yield {"_status": ("error", f"{port} is busy: the logger owns it. Use Live mode for the same "
                                            "data, or stop the logger.")}
                return
            yield {"_status": ("warning", f"{port} not available (board unplugged?). Retrying every 3 s.")}
            time.sleep(3)


def log_event(unix, label):
    new = not os.path.exists(EVENTS_CSV)
    with open(EVENTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csvmod.writer(f)
        if new:
            w.writerow(["unix", "label"])
        w.writerow([unix, label])


# ------------------------------------------------------------------ websocket
@app.websocket("/ws")
async def stream(ws: WebSocket):
    await ws.accept()
    q = ws.query_params
    src, mode = q.get("src", "live"), q.get("mode", "adaptive")
    def num(key, default, lo, hi, cast=int):
        try:
            return min(hi, max(lo, cast(q.get(key, default))))
        except (TypeError, ValueError):
            return default
    speed = num("speed", 20, 1, 600)
    window = num("window", 1800, 60, 4 * 3600)
    start_h = num("start_h", 3.0, 0.0, 10000.0, float)
    mode = mode if mode in MODES else "adaptive"
    stop, flags = threading.Event(), {"rezero": False, "events": []}
    loop = asyncio.get_running_loop()
    out: asyncio.Queue = asyncio.Queue(maxsize=50)

    def put(msg):
        fut = asyncio.run_coroutine_threadsafe(out.put(msg), loop)
        try:
            fut.result(timeout=30)
        except Exception:
            stop.set()

    def worker():
        try:
            b = bundle()
            pt = PhysTrust(b, mode)
            gen = {"live": lambda: live_source(stop, window),
                   "replay": lambda: replay_source(stop, start_h),
                   "sim": lambda: sim_source(stop),
                   "serial": lambda: serial_source(stop, q.get("port", "COM8"))}.get(src, lambda: live_source(stop, window))()
            batch, warm_rows = [], []
            # re-apply logged re-zeros when streaming history (else a reconnect would undo them)
            rz = []
            if src in ("live", "replay") and os.path.exists(EVENTS_CSV):
                e = pd.read_csv(EVENTS_CSV)
                rz = sorted(float(u) for u, lab in zip(e["unix"], e["label"]) if lab == "re-zero")
            paced = src in ("replay", "sim")
            for s in gen:
                if stop.is_set():
                    break
                if "_warm_done" in s:            # show history even before new samples arrive
                    if warm_rows:
                        put({"type": "rows", "rows": warm_rows})
                        warm_rows = []
                    continue
                if "_status" in s:
                    kind, msg = s["_status"]
                    put({"type": "status", "kind": kind, "msg": msg})
                    if kind == "error":
                        break
                    continue
                warm = bool(s.pop("_warm", False))
                u = s.get("unix", np.nan)
                while rz and np.isfinite(u) and u >= rz[0]:
                    if u - rz[0] < 5:                  # logged re-zero falls here: replay it
                        pt.rezero()
                    rz.pop(0)
                row = pt.step(s)
                row["unix"] = u
                row["fault"] = int(s.get("fault", 0) or 0)
                if flags["rezero"] and not warm:
                    pt.rezero()
                    flags["rezero"] = False
                    if np.isfinite(row["unix"]):
                        log_event(row["unix"], "re-zero")
                    put({"type": "event", "unix": row["unix"], "label": "re-zero"})
                while flags["events"] and not warm and np.isfinite(row["unix"]):
                    lab = flags["events"].pop(0)
                    log_event(row["unix"], lab)
                    put({"type": "event", "unix": row["unix"], "label": lab})
                # channels on the fused scale: the filter models each sensor channel as C * drift^b
                if np.isfinite(row.get("drift", np.nan)) and row["drift"] > 0:
                    k = row["drift"] ** (-pt.cfg.b)
                    row["raw_c"], row["phys_c"] = row["idx_raw"] * k, row["idx_phys"] * k
                r = clean(row)
                if warm:
                    warm_rows.append(r)
                    continue
                if warm_rows:
                    put({"type": "rows", "rows": warm_rows})
                    warm_rows = []
                batch.append(r)
                if not paced or len(batch) >= speed:
                    put({"type": "rows", "rows": batch})
                    batch = []
                    if paced:
                        time.sleep(0.25)
            if batch:
                put({"type": "rows", "rows": batch})
        except Exception as e:  # report instead of silently closing
            put({"type": "status", "kind": "error", "msg": f"{type(e).__name__}: {e}"})
        put({"type": "done"})

    async def reader():
        while True:
            m = await ws.receive_json()
            if m.get("cmd") == "rezero":
                flags["rezero"] = True
            elif m.get("cmd") == "event" and m.get("label"):
                flags["events"].append(str(m["label"])[:60])

    b = bundle()
    await ws.send_json({"type": "hello", "src": src, "mode": mode,
                        "prior": PRIOR_LABEL.get(type(b.prior).__name__, "none") if b and b.prior else "none"})
    threading.Thread(target=worker, daemon=True).start()
    rtask = asyncio.create_task(reader())
    try:
        while True:
            msg = await out.get()
            await ws.send_json(msg)
            if msg["type"] == "done":
                break
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        stop.set()
        rtask.cancel()


# ------------------------------------------------------------------ REST
@app.get("/api/meta")
def meta():
    b = bundle()
    info = None
    if b is not None:
        info = dict(r0=b.r0, sigma_sys=float(b.sigma_sys), q=float(b.conformal.q), lam_per_h=b.cfg.lam * 3600,
                    prior=PRIOR_LABEL.get(type(b.prior).__name__, "none") if b.prior else "none",
                    version=getattr(b, "version", ""), ood_thr=float(getattr(b.prior, "ood_thr", 0) or 0))

    def table(path, **kw):
        d = read_csv_safe(path)
        return None if d is None else d.round(4).replace({np.nan: None}).to_dict("records")

    pc = read_csv_safe("results/prior_compare.csv")
    if pc is not None:
        pc = pc.groupby("prior").mean(numeric_only=True).reset_index().round(4)
        pc = pc[[c for c in ("prior", "rmse_log", "prior_1step_rmse", "picp_clean", "ms_per_step") if c in pc]]
        pc = pc.to_dict("records")
    ws = read_csv_safe("results/weather/stats.csv")
    images = {k: p for k, p in dict(uci="results/uci/uci_validation.png", weather="results/weather/weather.png",
                                     session="results/session/session.png", trace="results/adaptive_trace.png").items()
              if os.path.exists(p)}
    return dict(bundle=info, modes=list(MODES), uci=table("results/uci/ablation.csv"), prior_compare=pc,
                weather_stats=None if ws is None else ws.round(3).iloc[0].replace({np.nan: None}).to_dict(),
                images=images)


@app.get("/api/weather")
def weather():
    w = weather_now()
    if w is None:
        return None
    return dict(T_ref=float(w["T_ref"]), RH_ref=float(w["RH_ref"]), time=str(w["time"]))


@app.post("/api/explain")
def explain_row(body: dict):
    """Plain-language explanation of the current state by the local LLM (gemma3:4b via Ollama).
    Commentary only: never feeds back into the estimator."""
    keys = ("state", "ood_cause", "action_note", "T", "RH", "idx_raw", "idx_phys", "fused", "lo", "hi",
            "trust_raw", "trust_phys", "trust_prior", "p_ood_smooth", "calib_confidence", "drift")
    row = {k: body.get(k) for k in keys if body.get(k) is not None}
    causes = {"distance": "current inputs (recent signal shape, temperature, humidity) differ statistically from the "
                          "calibration/training data",
              "envelope": "temperature or humidity is outside the calibrated range",
              "disagreement": "the five neural-network ensemble members disagree",
              "novelty": "the physics-corrected signal changed suddenly",
              "none": "no out-of-distribution cause"}
    row["ood_cause"] = causes.get(row.get("ood_cause"), row.get("ood_cause", "n/a"))
    event = (f"Operating state {row.get('state')}. Detected cause: {row['ood_cause']}. "
             f"System note: {row.get('action_note', '')} Explain only this cause; do not invent others.")
    return {"text": explain(event, row, timeout=60)}


@app.get("/api/events")
def events():
    d = read_csv_safe(EVENTS_CSV)
    return [] if d is None else d.to_dict("records")


@app.get("/results/{path:path}")
def result_file(path: str):
    full = os.path.normpath(os.path.join(ROOT, "results", path))
    if not full.startswith(os.path.join(ROOT, "results") + os.sep) or not os.path.isfile(full):
        raise HTTPException(404)
    return FileResponse(full)


DIST = os.path.join(ROOT, "web", "dist")
if os.path.isdir(DIST):
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
