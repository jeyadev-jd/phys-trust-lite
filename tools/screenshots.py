"""Capture dashboard screenshots for the README (headless Chrome over the DevTools protocol, real time).

    python tools/screenshots.py            (dashboard server must be running on :8600)

Writes docs/screenshots/*.png. Views are selected with the dashboard's #src / #tab / #only deep links.
"""
import asyncio
import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.request

import websockets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "screenshots")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE = "http://localhost:8600/"
PORT = 9333
SHOTS = [  # (file, hash, wait seconds, width, height or None = full page)
    ("dashboard_live.png", "src=live&tab=trust", 12, 1440, None),
    ("dashboard_simulator.png", "src=sim&tab=trust", 25, 1440, None),
    ("tab_channel_health.png", "src=live&tab=trust&only=tab", 10, 1440, None),
    ("tab_cross_sensitivity.png", "src=live&tab=cross&only=tab", 10, 1440, None),
    ("tab_events.png", "src=live&tab=events&only=tab", 10, 1440, None),
    ("tab_validation.png", "src=live&tab=valid&only=tab", 14, 1440, None),
    ("tab_model.png", "src=live&tab=model&only=tab", 8, 1440, None),
    ("tab_results.png", "src=live&tab=results&only=tab", 8, 1440, None),
    ("dashboard_mobile.png", "src=live&tab=trust", 12, 390, 1600),
]


async def shoot(ws_url, url, wait, w, h, path):
    async with websockets.connect(ws_url, max_size=None) as ws:
        n = 0

        async def cmd(method, **params):
            nonlocal n
            n += 1
            await ws.send(json.dumps({"id": n, "method": method, "params": params}))
            while True:
                m = json.loads(await ws.recv())
                if m.get("id") == n:
                    return m.get("result", {})

        await cmd("Emulation.setDeviceMetricsOverride", width=w, height=h or 1000, deviceScaleFactor=1,
                  mobile=w < 600)
        await cmd("Page.enable")
        await cmd("Page.navigate", url=url)
        await asyncio.sleep(wait)                       # real time: live data must stream in
        if h is None:
            m = await cmd("Runtime.evaluate", expression="document.documentElement.scrollHeight", returnByValue=True)
            h = int(m["result"]["value"])
            await cmd("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=False)
            await asyncio.sleep(1.5)
        r = await cmd("Page.captureScreenshot", format="png")
        open(path, "wb").write(base64.b64decode(r["data"]))


def main():
    os.makedirs(OUT, exist_ok=True)
    prof = tempfile.mkdtemp(prefix="pt-shots-")
    proc = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                             f"--remote-debugging-port={PORT}", f"--user-data-dir={prof}", "about:blank"])
    try:
        for _ in range(40):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version", timeout=1)
                break
            except OSError:
                time.sleep(0.25)
        for name, h, wait, w, ht in SHOTS:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/json/new?about:blank", method="PUT")
            tab = json.loads(urllib.request.urlopen(req).read())
            asyncio.run(shoot(tab["webSocketDebuggerUrl"], f"{BASE}#{h}", wait, w, ht, os.path.join(OUT, name)))
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/close/{tab['id']}")
            print("saved", name)
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
