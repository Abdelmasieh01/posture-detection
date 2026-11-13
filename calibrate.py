"""Per-user calibration: sit upright for a few seconds so the head check knows YOUR baseline.

  python calibrate.py                 # RTSP camera from .env
  python calibrate.py --camera 1      # USB / phone camera
  python calibrate.py --seconds 10

live_predict.py runs this automatically on first start (and when you press c).
Sit as you would with good posture, head over your shoulders, looking at the screen.
Writes calibration.json. Press q to cancel.
"""
import argparse
import time

import cv2
import numpy as np

from config import mask_url, resolve_source
from posture_core import FEATURE_NAMES, MIN_VISIBILITY, FrameSource, PoseExtractor, draw_overlay
from posture_monitor import save_calibration

TITLE = "Calibration (q to cancel)"


def put_text(frame, text, org, color=(255, 255, 255), scale=0.7):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def calibrate(src, extractor, out="calibration.json", seconds=10.0, delay=3.0,
              interval=0.5, width=640):
    """Record the upright baseline from an open FrameSource + PoseExtractor.
    Returns the saved data dict, or None if cancelled / not enough good samples."""
    start, last_run, feats, samples = time.time(), 0.0, None, []
    while True:
        elapsed = time.time() - start
        if elapsed >= delay + seconds:
            break
        ok, frame = src.read()
        if not ok:
            time.sleep(0.01)
            if elapsed > delay + seconds - 0.5:
                break
            continue
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (width, int(h * width / w)))

        if time.time() - last_run >= interval:
            last_run = time.time()
            feats = extractor.extract(frame)
            if feats is not None and feats.visibility >= MIN_VISIBILITY and elapsed >= delay:
                samples.append(dict(feats.values))
        if feats is not None:
            draw_overlay(frame, feats)

        fh = frame.shape[0]
        if elapsed < delay:
            put_text(frame, f"Calibration: get ready... {delay - elapsed:.0f}", (10, fh - 20), (0, 255, 255))
        else:
            left = delay + seconds - elapsed
            put_text(frame, f"Sit upright, look at the screen. {left:.0f}s  ({len(samples)} samples)",
                     (10, fh - 20), (0, 255, 0), 0.6)
        cv2.imshow(TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyWindow(TITLE)
            print("Calibration cancelled.")
            return None
    cv2.destroyWindow(TITLE)

    if len(samples) < 5:
        print(f"Only {len(samples)} good samples. Make sure your side is clearly visible and retry.")
        return None
    data = save_calibration(out, samples)
    spread = float(np.std([s["neck_angle"] for s in samples]))
    print(f"Saved {out} from {data['samples']} samples:")
    for n in FEATURE_NAMES:
        print(f"  {n:16s} {data['baseline'][n]:8.2f}")
    if spread > 5:
        print(f"Warning: your neck angle varied a lot (std {spread:.1f} deg). Sit still and recalibrate (press c) for a cleaner baseline.")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default=None, help="default: RTSP from .env; or index / URL / file")
    ap.add_argument("--seconds", type=float, default=10.0, help="how long to record")
    ap.add_argument("--delay", type=float, default=3.0, help="get-ready time before recording")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between samples")
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--variant", default="lite", choices=["lite", "full"])
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    source = resolve_source(args.camera)
    src = FrameSource(source)
    if not src.wait_first_frame():
        src.release()
        raise SystemExit(f"No frames from {mask_url(source)}. Check .env or the camera index.")
    extractor = PoseExtractor(args.variant)
    data = calibrate(src, extractor, args.out, args.seconds, args.delay, args.interval, args.width)
    src.release()
    extractor.close()
    cv2.destroyAllWindows()
    if data is None:
        raise SystemExit("Calibration not saved.")


if __name__ == "__main__":
    main()