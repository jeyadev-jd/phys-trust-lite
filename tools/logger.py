"""Serial -> CSV logger.  python tools/logger.py --port COM3 --out data/burnin.csv"""
import argparse
import sys
import time

import serial

HEADER = "millis,adc_mean,adc_std,T,RH,dht_ok,mq7_adc"
STALL_S = 30  # seconds without a valid line before the port is reopened


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    show = sys.stdout is not None  # None under pythonw (no console)
    with open(a.out, "a", buffering=1) as f:
        if f.tell() == 0:
            f.write(HEADER + ",unix\n")
        while True:  # survive USB unplug / board reset: reconnect instead of exiting
            try:
                with serial.Serial(a.port, a.baud, timeout=2) as ser:
                    last_ok = time.time()
                    while True:
                        # watchdog: port open but silent (USB glitch, hung board) -> reopen, which also resets the Uno
                        if time.time() - last_ok > STALL_S:
                            break
                        line = ser.readline().decode(errors="ignore").strip()
                        if not line or line.startswith("millis") or line.count(",") != 6:
                            continue
                        try:  # drop garbled lines (partial reads merged after a glitch)
                            [float(v) for v in line.split(",")]
                        except ValueError:
                            continue
                        last_ok = time.time()
                        f.write(f"{line},{time.time():.3f}\n")
                        if show:
                            sys.stdout.write("\r" + line + "   ")
            except serial.SerialException:
                time.sleep(5)


if __name__ == "__main__":
    main()
