"""Live posture angles from a camera, RTSP stream or video file.

  python live_angles.py --list                              # find USB/iVCam indices
  python live_angles.py                                     # RTSP camera from .env (default)
  python live_angles.py --camera 1                          # USB / iVCam camera index
  python live_angles.py --camera 1 --interval 1.5           # mimic the Pi: 1 detection / 1.5 s

Press q to quit.
"""
import argparse
import time

import cv2

from config import mask_url, resolve_source
from posture_core import FEATURE_NAMES, FrameSource, PoseExtractor, draw_overlay


def list_cameras(max_index=6):
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened() and cap.read()[0]:
            found.append(i)
        cap.release()
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="probe camera indices and exit")
    ap.add_argument("--camera", default=None,
                    help="default: RTSP stream from .env; or a camera index (0, 1, ...), rtsp:// URL, video file")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="seconds between detections (0 = every frame)")
    ap.add_argument("--variant", default="lite", choices=["lite", "full"])
    ap.add_argument("--width", type=int, default=640, help="resize frames to this width")
    ap.add_argument("--smooth", type=float, default=0.3, help="display smoothing (1 = none)")
    args = ap.parse_args()

    if args.list:
        print("Working camera indices:", list_cameras())
        return

    source = resolve_source(args.camera)
    src = FrameSource(source)
    if not src.wait_first_frame():
        src.release()
        raise SystemExit(f"No frames from {mask_url(source)}. Check .env (host, port, path, credentials) or the camera index.")

    extractor = PoseExtractor(args.variant)
    feats, smooth, last_run = None, None, 0.0

    while True:
        ok, frame = src.read()
        if not ok:
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
            continue
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (args.width, int(h * args.width / w)))

        now = time.time()
        if now - last_run >= args.interval:
            last_run = now
            feats = extractor.extract(frame)
            if feats is not None:
                if smooth is None:
                    smooth = dict(feats.values)
                else:
                    a = args.smooth
                    smooth = {n: a * feats.values[n] + (1 - a) * smooth[n] for n in FEATURE_NAMES}

        if feats is None:
            cv2.putText(frame, "No person detected", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            draw_overlay(frame, feats, smooth)

        cv2.imshow("Posture angles (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    src.release()
    extractor.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()