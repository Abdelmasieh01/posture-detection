"""Shows the camera with the body points and the angles (no model). Good for checking the camera view.

  python live_angles.py            # camera from .env
  python live_angles.py 1          # camera number 1 (or an rtsp:// address)
  python live_angles.py list       # which camera numbers work?

Press q to quit.
"""
import sys

import cv2

from camera import open_camera
from config import camera_source, mask_url
from posture_core import PoseExtractor, draw_overlay

if sys.argv[1:] == ["list"]:
    print("Working camera numbers:", [i for i in range(6) if open_camera(i).read()[0]])
    sys.exit()

source = camera_source(sys.argv[1] if len(sys.argv) > 1 else None)
cap = open_camera(source)
if not cap.isOpened():
    raise SystemExit(f"Cannot open {mask_url(source)}. Check .env or the camera number.")
pose = PoseExtractor("lite")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
    features = pose.extract(frame)
    if features is None:
        cv2.putText(frame, "No person detected", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        draw_overlay(frame, features)
    cv2.imshow("Angles (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()