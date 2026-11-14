"""HC-SR04 distance sensor for the Raspberry Pi: reading, baseline and the too-close rule.

Wiring (BCM numbering):
  VCC  -> 5V   (pin 2)
  GND  -> GND  (pin 6)
  TRIG -> GPIO23 (pin 16)
  ECHO -> R1 -> GPIO24 (pin 18) -> R2 -> GND      (voltage divider: ECHO is 5 V, the Pi needs 3.3 V max)
          R1 = R2 = 1.2k ohm  -> about 2.5 V (safe)      R1 = 1.2k, R2 = 2.4k -> about 3.3 V (same as 1k + 2k)

Commands (add --simulate to try the logic on a laptop without a sensor):
  python ultrasonic.py read        # live distance
  python ultrasonic.py baseline    # sit normally; saves your usual distance
  python ultrasonic.py watch       # distance + too-close alert after --hold seconds

Manual calibration: compare the reading with a ruler at a few distances. If it is off by a
constant, put that in .env as ULTRASONIC_OFFSET_CM (true = reading + offset, e.g. -1.5).
Other optional settings in .env: ULTRASONIC_TRIG, ULTRASONIC_ECHO.
If every reading says "no echo": check the 5 V supply and the divider on ECHO.
"""
import argparse
import datetime
import json
import os
import random
import statistics
import time

from config import load_env

DEFAULT_TRIG, DEFAULT_ECHO = 23, 24
BASELINE_PATH = "distance_baseline.json"


class UltrasonicSensor:
    """Median-filtered distance in cm.

    read_raw_cm: function returning one raw distance in cm, or None for no echo.
    offset_cm: manual correction added to the result.
    """

    def __init__(self, read_raw_cm, offset_cm=0.0, samples=7, interval=0.06, close=None):
        self._read_raw = read_raw_cm
        self.offset_cm, self.samples, self.interval, self._close = offset_cm, samples, interval, close

    @classmethod
    def from_gpio(cls, trig=DEFAULT_TRIG, echo=DEFAULT_ECHO, max_distance_m=2.0, **kw):
        try:
            from gpiozero import DistanceSensor
        except ImportError:
            raise SystemExit("gpiozero is missing. On Raspberry Pi OS it is usually preinstalled "
                             "(a venv needs --system-site-packages); otherwise pip install gpiozero. "
                             "To test the logic on a laptop, add --simulate.")
        dev = DistanceSensor(echo=echo, trigger=trig, max_distance=max_distance_m, queue_len=1)
        limit_cm = max_distance_m * 100.0

        def raw():
            cm = dev.distance * 100.0  # gpiozero reports metres, and the maximum when there is no echo
            return None if cm >= limit_cm - 0.5 else cm

        return cls(raw, close=dev.close, **kw)

    def read_cm(self):
        """Median of several readings (removes spikes) plus the manual offset. None if mostly no echo."""
        values = []
        for i in range(self.samples):
            v = self._read_raw()
            if v is not None:
                values.append(v)
            if i < self.samples - 1 and self.interval:
                time.sleep(self.interval)
        if len(values) < (self.samples + 1) // 2:
            return None
        return round(statistics.median(values) + self.offset_cm, 1)

    def close(self):
        if self._close:
            self._close()


class SimulatedEcho:
    """Fake sensor for laptops: about 60 cm, then leans in to about 35 cm after drop_after seconds,
    with noise, occasional outliers and missing echoes."""

    def __init__(self, drop_after=10.0):
        self.t0, self.drop_after = time.time(), drop_after

    def __call__(self):
        r = random.random()
        if r < 0.05:
            return None
        if r < 0.08:
            return random.uniform(5, 150)
        base = 35.0 if time.time() - self.t0 > self.drop_after else 60.0
        return base + random.gauss(0, 0.8)


def measure_baseline(sensor, seconds=8.0, interval=0.3):
    """Median distance (and spread) while the user sits normally."""
    end, values = time.time() + seconds, []
    while time.time() < end:
        v = sensor.read_cm()
        if v is not None:
            values.append(v)
        time.sleep(interval)
    if len(values) < 5:
        raise RuntimeError("Not enough valid readings. Check the wiring and that you face the sensor.")
    return float(statistics.median(values)), float(statistics.pstdev(values))


def distance_label(cm, baseline_cm, margin_cm=15.0):
    """'tooclose' when closer than baseline minus margin, else 'ok'. None when there is no reading."""
    if cm is None:
        return None
    return "tooclose" if cm < baseline_cm - margin_cm else "ok"


class TooCloseMonitor:
    """Alerts after `hold` seconds of continuous too-close readings. Unknown readings are ignored."""

    def __init__(self, hold=8.0):
        self.hold, self.since = hold, None

    def update(self, label, t=None):
        t = time.time() if t is None else t
        if label == "tooclose":
            if self.since is None:
                self.since = t
        elif label == "ok":
            self.since = None
        bad_for = 0.0 if self.since is None else t - self.since
        return {"alert": self.since is not None and bad_for >= self.hold, "bad_for": bad_for}


# ---------------------------------------------------------------- command line

def make_sensor(args):
    load_env()
    offset = float(os.environ.get("ULTRASONIC_OFFSET_CM", "0") or 0)
    if args.simulate:
        return UltrasonicSensor(SimulatedEcho(getattr(args, "sim_drop_after", 10.0)),
                                offset_cm=offset, interval=0.005)
    return UltrasonicSensor.from_gpio(int(os.environ.get("ULTRASONIC_TRIG", DEFAULT_TRIG)),
                                      int(os.environ.get("ULTRASONIC_ECHO", DEFAULT_ECHO)),
                                      offset_cm=offset)


def cmd_read(args):
    sensor = make_sensor(args)
    try:
        while True:
            cm = sensor.read_cm()
            print(f"{cm:6.1f} cm" if cm is not None else "no echo", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        sensor.close()


def cmd_baseline(args):
    sensor = make_sensor(args)
    try:
        print("Sit the way you normally do, facing the screen. Starting in 3 s...")
        time.sleep(3)
        baseline, std = measure_baseline(sensor, args.seconds)
    finally:
        sensor.close()
    with open(args.baseline, "w") as f:
        json.dump({"baseline_cm": baseline, "std_cm": std,
                   "created": datetime.datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    print(f"Baseline distance: {baseline:.1f} cm (spread {std:.1f} cm). Saved {args.baseline}")
    if std > 3:
        print("Warning: you moved a lot. Sit still and run it again for a cleaner baseline.")


def cmd_watch(args):
    if not os.path.exists(args.baseline):
        raise SystemExit(f"No {args.baseline}. Run: python ultrasonic.py baseline")
    with open(args.baseline) as f:
        baseline = json.load(f)["baseline_cm"]
    sensor, monitor = make_sensor(args), TooCloseMonitor(args.hold)
    print(f"Baseline {baseline:.1f} cm. Too close = below {baseline - args.margin:.1f} cm for {args.hold:g} s.")
    end = None if args.duration is None else time.time() + args.duration
    was_alert = False
    try:
        while end is None or time.time() < end:
            cm = sensor.read_cm()
            label = distance_label(cm, baseline, args.margin)
            state = monitor.update(label)
            shown = f"{cm:6.1f} cm" if cm is not None else "no echo"
            print(f"\r{shown} | {label or '-':8s} | too close for {state['bad_for']:4.1f}s"
                  f"{'  ALERT' if state['alert'] else '       '}", end="", flush=True)
            if state["alert"] != was_alert:
                print(f"\n{'ALERT: too close to the screen' if state['alert'] else 'Back to a good distance'}")
                was_alert = state["alert"]
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    finally:
        sensor.close()
        print()


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--simulate", action="store_true", help="fake sensor, to test the logic on a laptop")
    common.add_argument("--baseline", default=BASELINE_PATH, help="user distance baseline file")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("read", parents=[common]).set_defaults(fn=cmd_read)

    p = sub.add_parser("baseline", parents=[common])
    p.add_argument("--seconds", type=float, default=8.0)
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("watch", parents=[common])
    p.add_argument("--margin", type=float, default=15.0, help="cm closer than the baseline that counts as too close")
    p.add_argument("--hold", type=float, default=8.0, help="seconds before the alert")
    p.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    p.add_argument("--sim-drop-after", type=float, default=10.0, help="--simulate: seconds before leaning in")
    p.set_defaults(fn=cmd_watch)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()