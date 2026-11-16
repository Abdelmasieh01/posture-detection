"""HC-SR04 distance sensor on the Raspberry Pi (uses the gpiozero library).

Wiring (BCM numbers):
  VCC  -> 5V   (pin 2)
  GND  -> GND  (pin 6)
  TRIG -> GPIO23 (pin 16)
  ECHO -> R1 -> GPIO24 (pin 18) -> R2 -> GND    (a voltage divider: ECHO gives 5 V, the Pi only accepts 3.3 V)
          R1 = R2 = 1.2k ohm gives about 2.5 V (safe).  R1 = 1.2k and R2 = 2.4k gives about 3.3 V.

  python ultrasonic.py read        # shows the distance (Ctrl+C to stop)
  python ultrasonic.py baseline    # sit normally; saves your usual distance

Manual calibration: compare the reading with a ruler. If it is always off by the same amount,
set ULTRASONIC_OFFSET_CM in .env (true = reading + offset, for example -1.5).
Pins can be changed in .env too: ULTRASONIC_TRIG, ULTRASONIC_ECHO.
"""
import datetime
import json
import os
import statistics
import sys
import time

from config import load_env

BASELINE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "distance_baseline.json")
MAX_DISTANCE_M = 2.0


def open_sensor():
    """gpiozero measures all the time in the background and gives the median of the last 5
    measurements, which removes random spikes.
    partial=True is important: without it, reading the distance waits forever when no echo
    ever arrives (for example a sensor that is not connected) and the whole program freezes."""
    from gpiozero import DistanceSensor
    from gpiozero.exc import BadPinFactory
    load_env()
    try:
        return DistanceSensor(echo=int(os.environ.get("ULTRASONIC_ECHO", 24)),
                              trigger=int(os.environ.get("ULTRASONIC_TRIG", 23)),
                              max_distance=MAX_DISTANCE_M, queue_len=5, partial=True)
    except BadPinFactory:
        raise RuntimeError("gpiozero cannot find a GPIO library. On the Pi run: sudo apt install python3-lgpio "
                           "and let the venv use it (see 'Raspberry Pi setup' in README.md)")


def read_cm(sensor):
    """Distance in cm (plus the manual offset), or None when there is no usable reading."""
    cm = sensor.distance * 100
    # 0 = nothing measured yet, the maximum = nothing heard, below 2 cm = under the sensor's minimum range
    if cm < 2 or cm >= MAX_DISTANCE_M * 100 - 0.5:
        return None
    return round(cm + float(os.environ.get("ULTRASONIC_OFFSET_CM", 0)), 1)


def load_baseline():
    """The usual distance of the user in cm, or None if baseline was never run."""
    if not os.path.exists(BASELINE_FILE):
        return None
    with open(BASELINE_FILE) as f:
        return json.load(f)["baseline_cm"]


def run_read():
    sensor = open_sensor()
    try:
        while True:
            cm = read_cm(sensor)
            print(f"{cm} cm" if cm is not None else "no echo")
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass


def run_baseline():
    sensor = open_sensor()
    print("Sit the way you normally do, facing the screen. Starting in 3 s...")
    time.sleep(3)
    values = []
    for _ in range(25):  # about 8 seconds
        cm = read_cm(sensor)
        if cm is not None:
            values.append(cm)
        time.sleep(0.3)
    if len(values) < 10:
        raise SystemExit("Not enough readings. Check the wiring and that you face the sensor.")
    baseline = statistics.median(values)
    spread = statistics.pstdev(values)
    with open(BASELINE_FILE, "w") as f:
        json.dump({"baseline_cm": baseline, "created": datetime.datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    print(f"Baseline distance: {baseline:.1f} cm (spread {spread:.1f} cm). Saved.")
    if spread > 3:
        print("You moved a lot. Sit still and run it again for a cleaner baseline.")


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["read"]:
            run_read()
        elif sys.argv[1:] == ["baseline"]:
            run_baseline()
        else:
            print("Usage: python ultrasonic.py read   or   python ultrasonic.py baseline")
    except RuntimeError as error:
        sys.exit(str(error))