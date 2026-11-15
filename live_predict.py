"""Live view of what agent.py decides: posture label, head check and the alert rule, in a window.

  python live_predict.py           # camera from .env
  python live_predict.py 1         # camera number 1 (or an rtsp:// address)

The first time (no calibration.json) it calibrates for you. Keys: c = calibrate again, q = quit.
A red border = alert (bad posture for about HOLD seconds).
"""
import sys
import time

import cv2

from calibrate import calibrate
from camera import open_camera
from config import camera_source, mask_url
from posture_core import PoseExtractor, draw_overlay
from posture_monitor import CALIBRATION_FILE, GOOD_LABEL, MODEL_FILE, Alarm, HeadCheck, find_posture, load_model

INTERVAL = 1      # seconds between checks
HOLD = 8          # bad checks before an alert
NECK_MARGIN = 12  # degrees the head may lean forward compared to the calibration


def put_text(frame, text, y, color, scale=0.6):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


source = camera_source(sys.argv[1] if len(sys.argv) > 1 else None)
cap = open_camera(source)
if not cap.isOpened():
    raise SystemExit(f"Cannot open {mask_url(source)}. Check .env or the camera number.")
model = load_model(MODEL_FILE)
pose = PoseExtractor("lite")

head = HeadCheck.load(CALIBRATION_FILE, NECK_MARGIN)
if head is None and calibrate(cap, pose):
    head = HeadCheck.load(CALIBRATION_FILE, NECK_MARGIN)

alarm = Alarm(HOLD, GOOD_LABEL)
features, label, probs, alert, last_check = None, "unknown", [], False, 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
    height, width = frame.shape[:2]

    if time.time() - last_check >= INTERVAL:
        last_check = time.time()
        features = pose.extract(frame)
        label, confidence, probs = find_posture(features, model, head)
        alert = alarm.update(label)

    if features is not None:
        draw_overlay(frame, features)
    put_text(frame, label, height - 80, (0, 255, 0) if label == GOOD_LABEL else (0, 165, 255), 0.7)
    put_text(frame, " | ".join(f"{name[:4]} {p:.2f}" for name, p in probs), height - 50, (255, 255, 255))
    if head is None:
        put_text(frame, "head check OFF (press c to calibrate)", height - 110, (0, 255, 255))
    elif features is not None:
        put_text(frame, f"head {head.delta(features.values):+.0f} deg vs calibration (limit +{NECK_MARGIN})", height - 110, (255, 255, 255))
    put_text(frame, f"bad readings: {alarm.bad_count} of the last {HOLD + 1} (alert at {HOLD})", height - 20, (255, 255, 255))
    if alert:
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 14)
        put_text(frame, f"BAD POSTURE: {alarm.bad_label}", 200, (0, 0, 255), 0.8)

    cv2.imshow("Posture (q quit, c calibrate)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c") and calibrate(cap, pose):
        head = HeadCheck.load(CALIBRATION_FILE, NECK_MARGIN)
        alarm = Alarm(HOLD, GOOD_LABEL)

cap.release()
cv2.destroyAllWindows()