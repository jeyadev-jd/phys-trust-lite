import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Area, CartesianGrid, ComposedChart, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";

const SOURCES = [
  ["live", "Live sensor"], ["replay", "Replay"], ["sim", "Simulator"], ["serial", "Serial"],
];
const STATE_COLOR = {
  TRUSTED: "var(--good)", UNCERTAIN: "var(--warning)", CALIBRATION_NEEDED: "var(--warning)",
  OOD: "var(--serious)", PHYSICS_MISMATCH: "var(--critical)", SENSOR_FAULT: "var(--critical)",
};
const STATE_LABEL = {
  TRUSTED: "Trusted", UNCERTAIN: "Uncertain", CALIBRATION_NEEDED: "Calibration needed",
  OOD: "Out of distribution", PHYSICS_MISMATCH: "Physics mismatch", SENSOR_FAULT: "Sensor fault",
};
const ALARM = 0.3;
const MAX_POINTS = 700;

// ------------------------------------------------------------------ formatting
const fmt = (v, d = 0) => (v == null || !isFinite(v) ? "–" : Number(v).toLocaleString("en-IN", {
  minimumFractionDigits: d, maximumFractionDigits: d,
}));
const fmtIdx = (v) => (v == null ? "–" : v >= 1e5 ? v.toExponential(2) : fmt(v));
const tKey = (r) => (r.unix != null ? r.unix : r.millis / 1000);
const tLabel = (t, abs) => {
  if (abs) return new Date(t * 1000).toLocaleTimeString("en-GB", { hour12: false });
  const s = Math.round(t), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return `${h}:${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
};
const aqBand = (c) => (c < 800 ? ["Good", "var(--good)"] : c < 1200 ? ["Moderate", "var(--warning)"] : ["Poor", "var(--critical)"]);

function downsample(rows, n = MAX_POINTS) {
  if (rows.length <= n) return rows;
  const step = rows.length / n, out = [];
  for (let i = 0; i < n; i++) out.push(rows[Math.floor(i * step)]);
  out.push(rows[rows.length - 1]);
  return out;
}

function spans(rows, pred) {
  const out = [];
  let start = null, prev = null;
  for (const r of rows) {
    const on = pred(r);
    if (on && start == null) start = tKey(r);
    if (!on && start != null) { out.push([start, tKey(prev)]); start = null; }
    prev = r;
  }
  if (start != null && prev) out.push([start, tKey(prev)]);
  return out;
}

// ------------------------------------------------------------------ connection
function useStream(cfg, running) {
  const [rows, setRows] = useState([]);
  const [notice, setNotice] = useState(null);
  const [events, setEvents] = useState([]);
  const [hello, setHello] = useState(null);
  const [connected, setConnected] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const ws = useRef(null);

  useEffect(() => { setRows([]); setEvents([]); setNotice(null); }, [cfg.src]);

  useEffect(() => {
    if (!running) return undefined;
    if (attempt === 0) { setRows([]); setHello(null); }
    setNotice(null);
    let closedByUs = false, retry = null;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const qs = new URLSearchParams({
      src: cfg.src, mode: cfg.mode, speed: cfg.speed, start_h: cfg.startH, window: cfg.window, port: cfg.port,
    });
    const sock = new WebSocket(`${proto}://${location.host}/ws?${qs}`);
    ws.current = sock;
    sock.onopen = () => setConnected(true);
    sock.onclose = () => {
      setConnected(false);
      if (closedByUs || cfg.src === "replay" || cfg.src === "sim") return;
      setNotice({ kind: "warning", msg: "Connection to the PHYS-TRUST server lost. Reconnecting…" });
      retry = setTimeout(() => setAttempt((a) => a + 1), 3000);
    };
    let lastMsg = Date.now();
    const beat = (cfg.src === "live" || cfg.src === "serial") ? setInterval(() => {
      if (Date.now() - lastMsg > 15000 && sock.readyState <= 1) sock.close();   // -> onclose -> reconnect
    }, 5000) : null;
    sock.onmessage = (e) => {
      lastMsg = Date.now();
      const m = JSON.parse(e.data);
      if (m.type === "rows") {
        setRows((old) => {
          const next = old.concat(m.rows);
          return next.length > cfg.window ? next.slice(next.length - cfg.window) : next;
        });
        setNotice((n) => (n && n.kind === "warning" ? null : n));
      } else if (m.type === "status") setNotice({ kind: m.kind, msg: m.msg });
      else if (m.type === "event") setEvents((ev) => ev.concat({ unix: m.unix, label: m.label }));
      else if (m.type === "hello") setHello(m);
      else if (m.type === "done") closedByUs = true;
    };
    return () => { closedByUs = true; clearTimeout(retry); clearInterval(beat); sock.close(); };
  }, [running, attempt, cfg.src, cfg.mode, cfg.speed, cfg.startH, cfg.window, cfg.port]);

  useEffect(() => { if (!running) setAttempt(0); }, [running]);
  useEffect(() => {
    const wake = () => {
      if (document.visibilityState === "visible" && running && ws.current && ws.current.readyState !== 1) setAttempt((a) => a + 1);
    };
    document.addEventListener("visibilitychange", wake);
    return () => document.removeEventListener("visibilitychange", wake);
  }, [running]);

  const send = (obj) => ws.current && ws.current.readyState === 1 && ws.current.send(JSON.stringify(obj));
  return { rows, notice, events, hello, connected, send };
}

// ------------------------------------------------------------------ small parts
function Tip({ active, payload, label, abs, items }) {
  if (!active || !payload || !payload.length) return null;
  const r = payload[0].payload;
  return (
    <div className="tt">
      <div className="t">{tLabel(label, abs)}</div>
      {items.map(([k, name, f]) => (r[k] != null ? (
        <div key={k}><span>{name}</span><b>{f ? f(r[k]) : fmtIdx(r[k])}</b></div>
      ) : null))}
    </div>
  );
}

function LinkBadge({ running, connected, lastRow, src }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const id = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(id); }, []);
  const live = src === "live" || src === "serial";
  const age = lastRow?.unix != null ? now / 1000 - lastRow.unix : null;
  let label, color;
  if (!running) { label = "Stopped"; color = "var(--neutral)"; }
  else if (!connected) { label = "Disconnected"; color = "var(--critical)"; }
  else if (live && age != null && age > 10) { label = `No new data · ${Math.round(age)} s`; color = "var(--warning)"; }
  else { label = live ? `Live${age != null ? ` · ${Math.max(0, Math.round(age))} s ago` : ""}` : "Playing"; color = "var(--good)"; }
  return <span className="link"><i style={{ background: color }} />{label}</span>;
}

function Readout({ label, value, unit, sub, children }) {
  return (
    <div className="readout">
      <div className="lab">{label}</div>
      <div className="val">{children || <>{value}{unit && <small>{unit}</small>}</>}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

function Meter({ name, v, alarm = true }) {
  const low = alarm && v != null && v < ALARM;
  return (
    <div className="meter">
      <span className="name">{name}</span>
      <div className="track">
        <div className="fill" style={{ width: `${Math.max(0, Math.min(1, v ?? 0)) * 100}%`, background: low ? "var(--critical)" : "var(--ink-2)" }} />
        <div className="thr" style={{ left: `${ALARM * 100}%` }} title="alarm threshold 0.30" />
      </div>
      <span className="v" style={{ color: low ? "var(--critical)" : undefined }}>{v == null ? "–" : v.toFixed(2)}</span>
    </div>
  );
}

function Uncertainty({ r }) {
  if (!r) return <div className="empty">No data</div>;
  const q2 = (r.q_conf ?? 1) ** 2;
  const parts = [
    ["Filter (EKF)", q2 * (r.var_ekf ?? 0), "var(--s-fused)"],
    ["Curve exponent b", q2 * (r.var_b ?? 0), "var(--s-raw)"],
    ["Systematic / trust", q2 * (r.var_sys ?? 0), "var(--s-phys)"],
    ["Drift since anchor", r.var_drift ?? 0, "var(--s-prior)"],
  ];
  const tot = parts.reduce((a, p) => a + p[1], 0) || 1;
  return (
    <>
      <div className="stack" role="img" aria-label="Share of total variance by source">
        {parts.map(([n, v, c]) => (
          <div key={n} title={`${n}: ${((100 * v) / tot).toFixed(0)} %`} style={{ flexGrow: v / tot, background: c }} />
        ))}
      </div>
      <div className="ulist">
        {parts.map(([n, v, c]) => (
          <React.Fragment key={n}>
            <span><i style={{ background: c }} />{n}</span>
            <span>{((100 * v) / tot).toFixed(0)} %</span>
          </React.Fragment>
        ))}
        <span>σ total (log units)</span><span>{Math.sqrt(tot).toFixed(3)}</span>
      </div>
    </>
  );
}

function StateStrip({ rows, abs }) {
  if (!rows.length) return <div className="strip" />;
  const segs = [];
  for (const r of rows) {
    const last = segs[segs.length - 1];
    if (last && last.s === r.state) last.n++;
    else segs.push({ s: r.state, n: 1 });
  }
  const used = [...new Set(rows.map((r) => r.state))];
  return (
    <>
      <div className="strip" role="img" aria-label="State over time">
        {segs.map((g, i) => (
          <div key={i} title={STATE_LABEL[g.s] || g.s} style={{ flexGrow: g.n, background: STATE_COLOR[g.s] || "var(--neutral)" }} />
        ))}
      </div>
      <div className="strip-axis"><span>{tLabel(tKey(rows[0]), abs)}</span><span>{tLabel(tKey(rows[rows.length - 1]), abs)}</span></div>
      <div className="legend" style={{ marginTop: 8 }}>
        {used.map((s) => <span key={s}><i className="box" style={{ background: STATE_COLOR[s] }} />{STATE_LABEL[s] || s}</span>)}
      </div>
    </>
  );
}

// ------------------------------------------------------------------ charts
function logTicks([lo, hi]) {
  const out = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= lo && v <= hi) out.push(v); }
  return out;
}

function MainChart({ rows, events, abs }) {
  const data = useMemo(() => downsample(rows).map((r) => ({
    ...r, t: tKey(r), band: r.lo != null && r.hi != null ? [r.lo, r.hi] : null,
    raw: r.raw_c ?? r.idx_raw, phys: r.phys_c ?? r.idx_phys,   // channels on the fused (drift-corrected) scale
  })), [rows]);
  // y-range: robust (1st-99th percentile, so one spike cannot squash the plot), snapped to 1-2-5 steps and
  // sticky: it only changes when the data leave it or use less than a third of it, instead of every second
  const range = useRef(null);
  const domain = useMemo(() => {
    const v = data.flatMap((d) => [d.lo, d.hi, d.fused, d.raw, d.phys]).filter((x) => x > 0).sort((a, b) => a - b);
    if (!v.length) return [100, 10000];
    const q = (p) => v[Math.min(v.length - 1, Math.floor(p * v.length))];
    const lo = q(0.01) * 0.9, hi = q(0.99) * 1.1;
    const snap = (x, up) => {
      const e = 10 ** Math.floor(Math.log10(x));
      const steps = [1, 2, 5, 10].map((m) => m * e);
      return up ? steps.find((st) => st >= x) : [...steps].reverse().find((st) => st <= x);
    };
    const cur = range.current;
    const fits = cur && lo >= cur[0] && hi <= cur[1] && Math.log(hi / lo) > Math.log(cur[1] / cur[0]) / 3;
    if (!fits) range.current = [snap(lo, false), snap(hi, true)];
    return range.current;
  }, [data]);
  const faults = useMemo(() => spans(data, (r) => r.state === "SENSOR_FAULT"), [data]);
  if (!data.length) return <div className="empty" style={{ height: 340 }}>Waiting for data…</div>;
  const t0 = data[0].t, t1 = data[data.length - 1].t;
  return (
    <ResponsiveContainer width="100%" height={340}>
      <ComposedChart data={data} margin={{ top: 8, right: 36, bottom: 0, left: 8 }}>
        <CartesianGrid stroke="var(--rule)" vertical={false} />
        <XAxis dataKey="t" type="number" domain={[t0, t1]} tickFormatter={(t) => tLabel(t, abs)} stroke="var(--rule-strong)" minTickGap={60} />
        <YAxis scale="log" domain={domain} ticks={logTicks(domain)} allowDataOverflow tickFormatter={fmtIdx} stroke="var(--rule-strong)" width={64}
          label={{ value: "index (log)", angle: -90, position: "insideLeft", offset: 4 }} />
        {faults.map(([a, b], i) => (
          <ReferenceArea key={i} x1={a} x2={Math.max(b, a + 1)} fill="var(--critical)" fillOpacity={0.12} stroke="none" />
        ))}
        {events.filter((e) => e.unix >= t0 && e.unix <= t1).map((e, i) => (
          <ReferenceLine key={i} x={e.unix} stroke="var(--ink-3)" strokeDasharray="3 3"
            label={{ value: e.label, position: "insideTopLeft", fontSize: 11 }} />
        ))}
        <Area dataKey="band" stroke="none" fill="var(--band)" isAnimationActive={false} />
        <Line dataKey="raw" stroke="var(--s-raw)" strokeWidth={1.2} dot={false} isAnimationActive={false} strokeOpacity={0.8} />
        <Line dataKey="phys" stroke="var(--s-phys)" strokeWidth={1.2} dot={false} isAnimationActive={false} strokeOpacity={0.8} />
        <Line dataKey="fused" stroke="var(--s-fused)" strokeWidth={2} dot={false} isAnimationActive={false} />
        <Tooltip content={<Tip abs={abs} items={[["fused", "Fused"], ["lo", "Band low"], ["hi", "Band high"], ["raw", "Raw channel"],
          ["phys", "Physics channel"], ["state", "State", (s) => STATE_LABEL[s] || s]]} />} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function TrustChart({ rows, abs }) {
  const data = useMemo(() => downsample(rows).map((r) => ({ ...r, t: tKey(r) })), [rows]);
  if (!data.length) return <div className="empty">Waiting for data…</div>;
  return (
    <>
      <div className="legend" style={{ marginBottom: 8 }}>
        <span><i style={{ background: "var(--s-raw)" }} />Raw channel</span>
        <span><i style={{ background: "var(--s-phys)" }} />Physics channel</span>
        <span><i style={{ background: "var(--s-prior)" }} />Learned prior</span>
        <span><i style={{ background: "var(--ink-3)" }} />Alarm threshold 0.30</span>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data} margin={{ top: 8, right: 36, bottom: 0, left: 8 }}>
          <CartesianGrid stroke="var(--rule)" vertical={false} />
          <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(t) => tLabel(t, abs)} stroke="var(--rule-strong)" minTickGap={60} />
          <YAxis domain={[0, 1]} stroke="var(--rule-strong)" width={40} />
          <ReferenceLine y={ALARM} stroke="var(--ink-3)" strokeDasharray="4 3" />
          <Line dataKey="trust_raw" stroke="var(--s-raw)" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line dataKey="trust_phys" stroke="var(--s-phys)" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line dataKey="trust_prior" stroke="var(--s-prior)" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Tooltip content={<Tip abs={abs} items={[["trust_raw", "Raw", (v) => v.toFixed(2)], ["trust_phys", "Physics", (v) => v.toFixed(2)],
            ["trust_prior", "Prior", (v) => v.toFixed(2)]]} />} />
        </LineChart>
      </ResponsiveContainer>
    </>
  );
}

function CrossChart({ rows, x, xLabel }) {
  const data = useMemo(() => downsample(rows, 500).filter((r) => r.idx_raw > 0 && r.idx_phys > 0 && r[x] != null), [rows, x]);
  const raw = data.map((r) => ({ x: r[x], y: Math.log10(r.idx_raw) }));
  const phys = data.map((r) => ({ x: r[x], y: Math.log10(r.idx_phys) }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <ScatterChart margin={{ top: 8, right: 16, bottom: 16, left: 8 }}>
        <CartesianGrid stroke="var(--rule)" />
        <XAxis dataKey="x" type="number" domain={["dataMin - 0.5", "dataMax + 0.5"]} stroke="var(--rule-strong)" tickFormatter={(v) => v.toFixed(0)}
          label={{ value: xLabel, position: "insideBottom", offset: -10 }} />
        <YAxis dataKey="y" type="number" domain={["auto", "auto"]} stroke="var(--rule-strong)" width={48} tickFormatter={(v) => v.toFixed(2)}
          label={{ value: "log₁₀ index", angle: -90, position: "insideLeft" }} />
        <ZAxis range={[14, 14]} />
        <Scatter data={raw} fill="var(--s-raw)" fillOpacity={0.5} isAnimationActive={false} />
        <Scatter data={phys} fill="var(--s-phys)" fillOpacity={0.5} isAnimationActive={false} />
        <Tooltip cursor={false} formatter={(v) => Number(v).toFixed(2)} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

// ------------------------------------------------------------------ tabs
function EventsTab({ rows, events, abs, onEvent, live }) {
  const [label, setLabel] = useState("");
  const segs = useMemo(() => {
    const out = [];
    let cur = null;
    for (const r of rows) {
      if (r.state !== "TRUSTED") {
        if (cur && cur.state === r.state) { cur.end = tKey(r); cur.n++; }
        else { cur = { state: r.state, start: tKey(r), end: tKey(r), n: 1, cause: r.ood_cause, note: r.action_note, minRaw: r.trust_raw, minPhys: r.trust_phys }; out.push(cur); }
        cur.minRaw = Math.min(cur.minRaw, r.trust_raw ?? 1); cur.minPhys = Math.min(cur.minPhys, r.trust_phys ?? 1);
      } else cur = null;
    }
    return out.filter((s) => s.n >= 3).reverse().slice(0, 40);
  }, [rows]);
  return (
    <>
      {live && (
        <div className="ctl" style={{ marginBottom: 14 }}>
          <label>Mark event</label>
          <input type="text" value={label} placeholder="e.g. window opened" onChange={(e) => setLabel(e.target.value)} style={{ width: 240 }} />
          <button className="btn" disabled={!label.trim()} onClick={() => { onEvent(label.trim()); setLabel(""); }}>Add marker</button>
        </div>
      )}
      <h3>Non-trusted periods (newest first)</h3>
      {segs.length ? (
        <table className="data">
          <thead><tr><th>Start</th><th className="n">Duration</th><th>State</th><th>Cause</th><th className="n">Min trust raw</th><th className="n">Min trust phys</th><th>Action</th></tr></thead>
          <tbody>
            {segs.map((s, i) => (
              <tr key={i}>
                <td className="mono">{tLabel(s.start, abs)}</td>
                <td className="n">{Math.round(s.end - s.start + 1)} s</td>
                <td><span className="pill" style={{ fontSize: 13 }}><i style={{ background: STATE_COLOR[s.state] }} />{STATE_LABEL[s.state]}</span></td>
                <td>{s.state === "OOD" ? s.cause : "–"}</td>
                <td className="n">{s.minRaw.toFixed(2)}</td>
                <td className="n">{s.minPhys.toFixed(2)}</td>
                <td style={{ color: "var(--ink-2)" }}>{s.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <p className="note">No non-trusted periods in the shown history.</p>}
      <h3>Logged markers</h3>
      {events.length ? (
        <table className="data">
          <thead><tr><th>Time</th><th>Label</th></tr></thead>
          <tbody>{events.slice().reverse().slice(0, 30).map((e, i) => (
            <tr key={i}><td className="mono">{tLabel(e.unix, true)}</td><td>{e.label}</td></tr>
          ))}</tbody>
        </table>
      ) : <p className="note">No markers yet.</p>}
    </>
  );
}

function Table({ rows, cols }) {
  if (!rows || !rows.length) return <p className="note">Not available.</p>;
  const keys = cols || Object.keys(rows[0]);
  return (
    <table className="data">
      <thead><tr>{keys.map((k) => <th key={k} className={typeof rows[0][k] === "number" ? "n" : ""}>{k.replace(/_/g, " ")}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => (
        <tr key={i}>{keys.map((k) => (
          <td key={k} className={typeof r[k] === "number" ? "n" : ""}>{typeof r[k] === "number" ? Number(r[k].toFixed(4)) : r[k] ?? "–"}</td>
        ))}</tr>
      ))}</tbody>
    </table>
  );
}

function ValidationTab({ meta, last, live }) {
  const [w, setW] = useState(null);
  useEffect(() => {
    if (!live) return undefined;
    const get = () => fetch("/api/weather").then((r) => r.json()).then(setW).catch(() => {});
    get();
    const id = setInterval(get, 15 * 60 * 1000);
    return () => clearInterval(id);
  }, [live]);
  const ws = meta?.weather_stats;
  return (
    <>
      <h3>Live: node vs outdoor weather (Open-Meteo, Chennai)</h3>
      <p className="note">The node is indoors, so an offset is expected. The check is that it tracks the weather and stays physically plausible.</p>
      {live && w && last ? (
        <div className="facts">
          <div className="fact"><span>Node temperature</span><b>{fmt(last.T, 1)} °C</b></div>
          <div className="fact"><span>Outdoor temperature</span><b>{fmt(w.T_ref, 1)} °C</b><em>node − outdoor {(last.T - w.T_ref >= 0 ? "+" : "") + fmt(last.T - w.T_ref, 1)} °C</em></div>
          <div className="fact"><span>Node humidity</span><b>{fmt(last.RH)} %</b></div>
          <div className="fact"><span>Outdoor humidity</span><b>{fmt(w.RH_ref)} %</b><em>node − outdoor {(last.RH - w.RH_ref >= 0 ? "+" : "") + fmt(last.RH - w.RH_ref)} %</em></div>
        </div>
      ) : <p className="note">{live ? "Waiting for weather and sensor data…" : "Runs in Live sensor or Serial mode."}</p>}
      <h3>Past data: DHT11 vs Open-Meteo, hourly</h3>
      {ws ? (
        <div className="facts">
          {Object.entries(ws).map(([k, v]) => <div className="fact" key={k}><span>{k.replace(/_/g, " ")}</span><b>{v == null ? "–" : typeof v === "number" ? Number(v.toFixed(3)) : v}</b></div>)}
        </div>
      ) : <p className="note">Not available.</p>}
      {meta?.images?.weather && <img className="fig" src={`/${meta.images.weather}`} alt="DHT11 against Open-Meteo hourly weather" />}
      <h3>Reference dataset: UCI Air Quality (MOX sensor vs certified CO analyser, 10 months)</h3>
      <Table rows={meta?.uci} />
      {meta?.images?.uci && <img className="fig" style={{ marginTop: 12 }} src={`/${meta.images.uci}`} alt="UCI validation: analyser, static calibration and adaptive estimate with 90% band" />}
    </>
  );
}

function ModelTab({ meta }) {
  const b = meta?.bundle;
  const layers = [
    ["L0 Signal", "ADC to sensor resistance Rs = RL·(1023/ADC − 1); fault flags", "sensor_physics.py"],
    ["L1a Sensor physics", "Power law C = a·(Rs/(R0·CF(T,RH)))^b with temperature/humidity correction", "sensor_physics.py"],
    ["L1b Process physics", "Room mass balance dC/dt = S/V − λ(C − C_bg); plausibility envelope", "process_physics.py"],
    ["L2 Learned prior", `${b?.prior || "Deep ensemble"}: 60-s sequence of [Δlog C, T, RH] to next-step change; spread = epistemic uncertainty; Mahalanobis OOD`, "prior.py"],
    ["L3 Trust", "Innovation (NIS), rate plausibility, T/RH envelope, sensor lag, OOD → trust in [0, 1] per source", "trust.py"],
    ["L4 Fusion", "Extended Kalman filter, state [C, log drift]; measurement noise R = R0 / trust", "fusion.py"],
    ["L5 Uncertainty", "σ²_total = q²(σ²_EKF + σ²_b + σ²_sys/trust) + k_d·t; split-conformal 90 % band", "conformal.py"],
    ["States", "TRUSTED, UNCERTAIN, CALIBRATION_NEEDED, OOD, PHYSICS_MISMATCH, SENSOR_FAULT with actions", "states.py"],
  ];
  return (
    <>
      <h3>Pipeline</h3>
      <table className="data">
        <thead><tr><th>Layer</th><th>Function</th><th>Module</th></tr></thead>
        <tbody>{layers.map((l) => <tr key={l[0]}><td style={{ whiteSpace: "nowrap", fontWeight: 500 }}>{l[0]}</td><td>{l[1]}</td><td className="mono">{l[2]}</td></tr>)}</tbody>
      </table>
      <h3>Core equation</h3>
      <div className="eq">R_s = R_s,0 / max(τ_s, ε) — the Kalman filter weights each source by its trust</div>
      <h3>Loaded calibration</h3>
      {b ? (
        <div className="facts">
          <div className="fact"><span>Version</span><b style={{ fontSize: 13 }}>{b.version}</b></div>
          <div className="fact"><span>R0</span><b>{fmt(b.r0, 1)} kΩ</b></div>
          <div className="fact"><span>σ systematic</span><b>{b.sigma_sys.toFixed(3)}</b></div>
          <div className="fact"><span>Conformal q</span><b>{b.q.toFixed(2)}</b></div>
          <div className="fact"><span>Air exchange λ</span><b>{b.lam_per_h.toFixed(2)} /h</b></div>
          <div className="fact"><span>OOD threshold D²</span><b>{b.ood_thr.toFixed(1)}</b></div>
          <div className="fact"><span>Learned prior</span><b style={{ fontSize: 13 }}>{b.prior}</b></div>
        </div>
      ) : <p className="note">No calibration bundle loaded; datasheet priors in use.</p>}
    </>
  );
}

function ResultsTab({ meta }) {
  return (
    <>
      <h3>Synthetic benchmark (true sensor exponent swept ±σ, 5 test seeds)</h3>
      <div className="facts">
        <div className="fact"><span>Error vs fixed-weight fusion</span><b>−9 %</b><em>RMSE log 0.078 vs 0.085, p = 0.0008</em></div>
        <div className="fact"><span>90 % band coverage</span><b>100 % / 100 %</b><em>clean / local faults</em></div>
        <div className="fact"><span>Sensor-fault detection F1</span><b>0.79</b><em>latency 1–4 s</em></div>
        <div className="fact"><span>False OOD, clean</span><b>3.4 %</b><em>outside T/RH swing (was 5.0 %)</em></div>
      </div>
      <h3>Learned prior comparison (5 seeds)</h3>
      <Table rows={meta?.prior_compare} />
      {meta?.images?.trace && <img className="fig" style={{ marginTop: 12 }} src={`/${meta.images.trace}`} alt="Synthetic test day: fused estimate, band and trust" />}
      <h3>Real hardware sessions (25–26 Sep 2026)</h3>
      <div className="facts">
        <div className="fact"><span>Perfume interference</span><b>Flagged at onset</b><em>fused held ≈ 917 while raw → 10⁶</em></div>
        <div className="fact"><span>Hot air +8.5 °C / −18 %RH</span><b>≈ 3 % corrected</b><em>datasheet correction too weak</em></div>
        <div className="fact"><span>90 % band coverage, 26 Sep</span><b>91 %</b><em>after pin fix + recalibration (was 65 %)</em></div>
        <div className="fact"><span>False OOD, quiet periods</span><b>0–2 %</b><em>per half-hour, real log</em></div>
      </div>
      {meta?.images?.session && <img className="fig" src={`/${meta.images.session}`} alt="Real node session: index, trust, T/RH with event markers" />}
      <p className="note" style={{ marginTop: 12 }}>Limits: drift is unobservable within a day on one node (ABC re-anchoring mitigates); noise faults are poorly detected; short burn-in; no reference instrument, so no absolute ppm claim.</p>
    </>
  );
}

// ------------------------------------------------------------------ app
const saved = (() => { try { return JSON.parse(localStorage.getItem("pt-settings")) || {}; } catch { return {}; } })();
// deep links, e.g. #tab=valid&src=sim (for sharing a view or capturing screenshots)
const hashArgs = Object.fromEntries(new URLSearchParams(location.hash.slice(1)));

export default function App() {
  const [src, setSrc] = useState(hashArgs.src || saved.src || "live");
  const [mode, setMode] = useState(saved.mode || "adaptive");
  const [speed, setSpeed] = useState(saved.speed || 20);
  const [startH, setStartH] = useState(saved.startH ?? 3);
  const [windowMin, setWindowMin] = useState(saved.windowMin || 30);
  const [running, setRunning] = useState(saved.running ?? true);
  const [explainText, setExplainText] = useState(null);
  useEffect(() => {
    try { localStorage.setItem("pt-settings", JSON.stringify({ src, mode, speed, startH, windowMin, running })); } catch { /* storage blocked */ }
  }, [src, mode, speed, startH, windowMin, running]);
  const [tab, setTab] = useState(hashArgs.tab || "trust");
  const [meta, setMeta] = useState(null);
  const [fileEvents, setFileEvents] = useState([]);

  useEffect(() => {
    fetch("/api/meta").then((r) => r.json()).then(setMeta).catch(() => {});
    fetch("/api/events").then((r) => r.json()).then(setFileEvents).catch(() => {});
  }, []);

  const cfg = { src, mode, speed, startH, window: windowMin * 60, port: "COM8" };
  const { rows, notice, events, hello, connected, send } = useStream(cfg, running);
  const last = rows[rows.length - 1];
  const explainNow = () => {
    if (!last) return;
    setExplainText("Asking gemma3:4b (local)…");
    fetch("/api/explain", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(last) })
      .then((r) => r.json()).then((j) => setExplainText(j.text)).catch(() => setExplainText("LLM unavailable."));
  };
  const abs = last ? last.unix != null : true;
  const allEvents = useMemo(() => fileEvents.concat(events), [fileEvents, events]);
  const live = src === "live" || src === "serial";

  const oneMinAgo = rows.length > 61 ? rows[rows.length - 61] : rows[0];
  const [aq, aqc] = last ? aqBand(last.fused) : ["–", "var(--neutral)"];
  const half = last ? (last.hi - last.lo) / 2 : null;

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <b>PHYS-TRUST Lite</b>
          <span>{last?.node_id || "node-A"} · MQ135 + DHT11 · {hello?.prior || meta?.bundle?.prior || "–"}</span>
          <LinkBadge running={running} connected={connected} lastRow={last} src={src} />
        </div>
        <div className="ctl">
          <label>Source</label>
          <div className="seg">
            {SOURCES.map(([k, l]) => (
              <button key={k} aria-pressed={src === k} onClick={() => { if (k !== src) { setSrc(k); setRunning(false); } }}>{l}</button>
            ))}
          </div>
        </div>
        {src === "replay" && (
          <div className="ctl"><label>Start h</label><input type="number" min={0} step={0.5} value={startH} onChange={(e) => setStartH(+e.target.value)} /></div>
        )}
        {(src === "replay" || src === "sim") && (
          <div className="ctl"><label>Speed</label>
            <select value={speed} onChange={(e) => setSpeed(+e.target.value)}>{[5, 20, 60, 120].map((s) => <option key={s} value={s}>{s}×</option>)}</select>
          </div>
        )}
        <div className="ctl"><label>Mode</label>
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            {(meta?.modes || ["adaptive"]).map((m) => <option key={m}>{m}</option>)}
          </select>
        </div>
        <div className="ctl"><label>History</label>
          <select value={windowMin} onChange={(e) => setWindowMin(+e.target.value)}>{[5, 15, 30, 60].map((m) => <option key={m} value={m}>{m} min</option>)}</select>
        </div>
        <button className="btn" disabled={!running || !live} onClick={() => send({ cmd: "rezero" })}
          title="Use only with the window open and clean air">Re-zero</button>
        <button className="btn primary" onClick={() => setRunning((r) => !r)}>{running ? "Stop" : "Start"}</button>
      </header>

      <main className={`page${hashArgs.only === "tab" ? " only-tab" : ""}`}>
        {notice && <div className={`notice ${notice.kind}`}>{notice.msg}</div>}
        {!running && <div className="notice info">Choose a source and press Start. Live sensor follows the running logger without taking the serial port.</div>}

        <section className="panel statusline">
          <div className="state">
            <span className="sq" style={{ background: STATE_COLOR[last?.state] || "var(--neutral)" }} />
            <b>{last ? (STATE_LABEL[last.state] || last.state).toUpperCase() : "NO DATA"}</b>
          </div>
          <div className="action">
            {last ? <><b>{last.action?.replace(/_/g, " ")}</b> — {last.action_note}{last.alert_human && <span className="alert">Human alert</span>}</> : "Estimator idle."}
            {last && <button className="btn link-btn" onClick={explainNow}>Explain · gemma3:4b (Ollama, local)</button>}
            {explainText && <div className="explain">{explainText}</div>}
          </div>
          <div className="chips">
            <div className="chip"><span>P(OOD)</span><b>{last ? last.p_ood_smooth?.toFixed(2) : "–"}</b></div>
            <div className="chip"><span>Calib. confidence</span><b>{last ? last.calib_confidence?.toFixed(2) : "–"}</b></div>
            <div className="chip"><span>Compute tier</span><b>{last ? last.compute_tier : "–"}</b></div>
          </div>
        </section>

        <section className="panel readouts">
          <Readout label="Fused index" value={last ? fmtIdx(last.fused) : "–"}
            sub={last && oneMinAgo ? `${Math.round(last.fused - oneMinAgo.fused) === 0 ? "±" : last.fused > oneMinAgo.fused ? "+" : "−"}${fmtIdx(Math.abs(last.fused - oneMinAgo.fused))} vs 1 min ago` : "relative CO₂-equivalent"} />
          <Readout label="90 % band" value={half != null ? `± ${fmtIdx(half)}` : "–"} sub={half != null ? `± ${fmt((100 * half) / last.fused)} % of estimate` : "conformal interval"} />
          <Readout label="Air quality (relative)" sub="bands at 800 / 1200">
            <span className="pill"><i style={{ background: aqc }} />{aq}</span>
          </Readout>
          <Readout label="Temperature" value={last ? fmt(last.T, 1) : "–"} unit="°C" sub={last ? `humidity ${fmt(last.RH)} % · DHT11` : "DHT11"} />
          <Readout label="Since re-anchor" value={last ? fmt(last.t_since_anchor / 60) : "–"} unit="min" sub={last ? `ABC events ${last.abc_events} · drift ${last.drift?.toFixed(3)}` : "–"} />
        </section>

        <section className="panel">
          <div className="panel-h">
            <h2>Estimate with 90 % band</h2>
            <div className="legend">
              <span><i style={{ background: "var(--s-fused)" }} />Fused</span>
              <span><i className="box" style={{ background: "var(--band)" }} />90 % band</span>
              <span><i style={{ background: "var(--s-raw)" }} />Raw channel</span>
              <span><i style={{ background: "var(--s-phys)" }} />Physics channel</span>
              <span><i className="box" style={{ background: "var(--critical)", opacity: 0.25 }} />Sensor fault</span>
            </div>
          </div>
          <div className="panel-b"><MainChart rows={rows} events={allEvents} abs={abs} /></div>
          <div className="panel-b" style={{ borderTop: "1px solid var(--rule)" }}>
            <div style={{ fontSize: 12, color: "var(--ink-3)", marginBottom: 6 }}>STATE TIMELINE</div>
            <StateStrip rows={rows} abs={abs} />
          </div>
        </section>

        <div className="grid2">
          <section className="panel">
            <div className="panel-h"><h2>Channel health</h2><small>1 = no fault evidence, not accuracy · tick = alarm 0.30</small></div>
            <div className="panel-b">
              <Meter name="Raw channel" v={last?.trust_raw} />
              <Meter name="Physics channel" v={last?.trust_phys} />
              <Meter name="Learned prior" v={last?.trust_prior} alarm={false} />
              <div className="scale"><span /><div><span>0</span><span>0.5</span><span>1</span></div><span /></div>
              <p className="note" style={{ margin: "8px 0 0" }}>The alarm applies to the sensor channels. A low prior weight means the network is down-weighted on unfamiliar input, not a fault.</p>
            </div>
          </section>
          <section className="panel">
            <div className="panel-h"><h2>Uncertainty breakdown</h2><small>share of σ²_total</small></div>
            <div className="panel-b"><Uncertainty r={last} /></div>
          </section>
        </div>

        <section className="panel" style={{ marginBottom: 14 }}>
          <div className="panel-h"><h2>Accuracy (validated)</h2><small>health and trust above are fault evidence, not accuracy</small></div>
          <div className="facts" style={{ margin: 0, borderTop: 0, borderLeft: 0 }}>
            <div className="fact"><span>90 % band coverage, this node</span><b>0.89</b><em>target 0.90 · vs consensus curve, 26 Sep</em></div>
            <div className="fact"><span>R² vs certified analyser</span><b>0.62</b><em>UCI Air Quality, 10 months, MOX sensor</em></div>
            <div className="fact"><span>Learned-prior skill</span><b>0.41</b><em>vs persistence, held-out node data</em></div>
            <div className="fact"><span>Fault detection</span><b>97 % / 94 %</b><em>spike / stuck, synthetic injection</em></div>
            <div className="fact"><span>Absolute ppm</span><b>not claimed</b><em>no reference instrument on this node</em></div>
          </div>
        </section>

        <section className="panel tabs-panel">
          <nav className="tabs" role="tablist">
            {[["trust", "Channel health over time"], ["cross", "Cross-sensitivity"], ["events", "Events"], ["valid", "Validation"],
              ["model", "Model & method"], ["results", "Results"]].map(([k, l]) => (
              <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)}>{l}</button>
            ))}
          </nav>
          <div className="panel-b">
            {tab === "trust" && <TrustChart rows={rows} abs={abs} />}
            {tab === "cross" && (
              <>
                <p className="note">If the temperature/humidity correction worked, the physics channel would be flat against T and RH while the raw channel slopes.</p>
                <div className="legend" style={{ marginBottom: 6 }}>
                  <span><i className="box" style={{ background: "var(--s-raw)" }} />Raw channel</span>
                  <span><i className="box" style={{ background: "var(--s-phys)" }} />Physics channel</span>
                </div>
                <div className="grid2" style={{ gridTemplateColumns: "1fr 1fr", margin: 0 }}>
                  <CrossChart rows={rows} x="T" xLabel="Temperature (°C)" />
                  <CrossChart rows={rows} x="RH" xLabel="Relative humidity (%)" />
                </div>
              </>
            )}
            {tab === "events" && <EventsTab rows={rows} events={allEvents} abs={abs} live={running && live} onEvent={(l) => send({ cmd: "event", label: l })} />}
            {tab === "valid" && <ValidationTab meta={meta} last={last} live={live && running} />}
            {tab === "model" && <ModelTab meta={meta} />}
            {tab === "results" && <ResultsTab meta={meta} />}
          </div>
        </section>
        <p className="foot">Index is a relative CO₂-equivalent from one uncalibrated low-cost node, not certified ppm. Calibration {last?.calib_version || meta?.bundle?.version || "–"}.</p>
      </main>
    </>
  );
}
