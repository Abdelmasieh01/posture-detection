"""Records how YOU sit upright, so the head check knows your normal head position.

  python calibrate.py              # camera from .env
  python calibrate.py 1            # camera number 1 (or an rtsp:// address)

Sit as you would with good posture, head over your shoulders, looking at the screen.
Saves calibration.json. Press q to cancel.
"""
import sys
import time

import cv2

from camera import open_camera
from config import camera_source, mask_url
from posture_core import FEATURE_NAMES, MIN_VISIBILITY, PoseExtractor, draw_overlay
from posture_monitor import CALIBRATION_FILE, save_calibration

TITLE = "Calibration (q to cancel)"


def put_text(frame, text, y, color):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def calibrate(cap, extractor, out=CALIBRATION_FILE, seconds=10, delay=3, window=True, on_progress=None):
    """Waits `delay` seconds (get ready), then records for `seconds` seconds.
    window=False: no picture window (the Pi has no screen).
    on_progress("get_ready") and on_progress("recording") are called when those phases start.
    Returns the saved data, or None if cancelled or too few good readings."""
    say = on_progress or (lambda status: None)
    say("get_ready")
    start = time.time()
    samples, recording, last_sample, features = [], False, 0, None
    while time.time() - start < delay + seconds:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        if not recording and now - start >= delay:
            recording = True
            say("recording")
        if now - last_sample >= (0 if window else 0.2):  # a window shows every picture, without one 5 per second is enough
            last_sample = now
            frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
            features = extractor.extract(frame)
            if recording and features is not None and features.visibility >= MIN_VISIBILITY:
                samples.append(dict(features.values))
        if window:
            if features is not None:
                draw_overlay(frame, features)
            if recording:
                put_text(frame, f"Sit upright, look at the screen. {len(samples)} readings", frame.shape[0] - 20, (0, 255, 0))
            else:
                put_text(frame, "Calibration: get ready...", frame.shape[0] - 20, (0, 255, 255))
            cv2.imshow(TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyWindow(TITLE)
                print("Calibration cancelled.")
                return None
    if window:
        cv2.destroyWindow(TITLE)

    if len(samples) < 10:
        print(f"Only {len(samples)} good readings. Make sure your side is clearly visible and try again.")
        return None
    data = save_calibration(out, samples)
    print(f"Saved calibration from {len(samples)} readings:")
    for name in FEATURE_NAMES:
        print(f"  {name:16s} {data['baseline'][name]:8.2f}")
    return data


if __name__ == "__main__":
    source = camera_source(sys.argv[1] if len(sys.argv) > 1 else None)
    cap = open_camera(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {mask_url(source)}. Check .env or the camera number.")
    extractor = PoseExtractor("lite")
    result = calibrate(cap, extractor)
    cap.release()
    cv2.destroyAllWindows()
    if result is None:
        raise SystemExit("Calibration not saved.")