"""Optional: local LLM (Ollama gemma3:4b) narrates trust events in plain language.
Never in the estimation loop: output is commentary only, numbers come from the filter."""
import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"

PROMPT = """You explain a single-node air-quality estimator (MQ135 + DHT11) to a viewer.
Channels: raw sensor, physics-corrected (T/RH), learned prior. Trust in [0,1].
Index is a relative CO2-equivalent, NOT true ppm. Be factual, 2 sentences max.
Likely causes: fast rise+fast recovery = sensor-local interference (swab/breath);
T/RH change with raw distrust = cross-sensitivity; prior distrust = unfamiliar conditions/drift.

Event: {event}
State: {state}
Explanation:"""


def explain(event, state, model=MODEL, url=OLLAMA_URL, timeout=30):
    body = json.dumps({
        "model": model,
        "prompt": PROMPT.format(event=event, state=json.dumps(state, default=float)),
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 80},
    }).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["response"].strip()
    except Exception as e:  # LLM optional; never break the pipeline
        return f"(LLM unavailable: {e})"


def summarize_row(row):
    keys = ("T", "RH", "idx_raw", "idx_phys", "fused", "lo", "hi",
            "trust_raw", "trust_phys", "trust_prior", "rate_ratio", "drift")
    return {k: round(float(row[k]), 3) for k in keys if k in row}


class EventDetector:
    """Emits an event string when any trust crosses 0.5."""

    def __init__(self, thr=0.5):
        self.thr, self.prev = thr, {}

    def __call__(self, row):
        events = []
        for ch in ("raw", "phys", "prior"):
            v = row[f"trust_{ch}"]
            p = self.prev.get(ch, 1.0)
            if p >= self.thr > v:
                events.append(f"{ch} trust dropped to {v:.2f}")
            elif p < self.thr <= v:
                events.append(f"{ch} trust recovered to {v:.2f}")
            self.prev[ch] = v
        return events
