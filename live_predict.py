"""Live posture classification with the trained model.

  python live_predict.py                    # RTSP camera from .env (default)
  python live_predict.py --camera 1         # USB / phone camera index
  python live_predict.py --interval 1.5 --hold 5

On first start it calibrates automatically (sit upright for ~10 s). Keys while
running: c = recalibrate, q = quit. The red border simulates the RGB LED: it
shows only after bad posture persists for about --hold seconds.
"""
import argparse
import time

import cv2

from calibrate import calibrate
from config import mask_url, resolve_source
from posture_core import MIN_VISIBILITY, FrameSource, PoseExtractor, draw_overlay
from posture_monitor import HEAD_LABEL, HeadCheck, PostureMonitor, class_probs, load_model


def put_text(frame, text, org, color, scale=0.7):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default=None,
                    help="default: RTSP stream from .env; or a camera index (0, 1, ...), rtsp:// URL, video file")
    ap.add_argument("--model", default="models/posture_model.joblib")
    ap.add_argument("--interval", type=float, default=1.5, help="seconds between detections")
    ap.add_argument("--hold", type=float, default=5.0, help="seconds of bad posture before alert (4-8)")
    ap.add_argument("--good", default="goodposture", help="label of the good class")
    ap.add_argument("--variant", default="lite", choices=["lite", "full"])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--calibration", default="calibration.json", help="calibration file")
    ap.add_argument("--neck-margin", type=float, default=12.0,
                    help="degrees of head-forward beyond your upright baseline before it counts as bad")
    ap.add_argument("--recalibrate", action="store_true", help="calibrate now even if a calibration exists")
    ap.add_argument("--no-calibrate", action="store_true", help="skip calibration (head check off)")
    ap.add_argument("--calib-seconds", type=float, default=10.0, help="calibration recording time")
    args = ap.parse_args()

    bundle = load_model(args.model)
    monitor = PostureMonitor(hold=args.hold, good_label=args.good)

    source = resolve_source(args.camera)
    src = FrameSource(source)
    if not src.wait_first_frame():
        src.release()
        raise SystemExit(f"No frames from {mask_url(source)}. Check .env (host, port, path, credentials) or the camera index.")
    extractor = PoseExtractor(args.variant)

    def run_calibration():
        print("Calibration: sit upright and look at the screen.")
        return calibrate(src, extractor, args.calibration, seconds=args.calib_seconds, width=args.width)

    head = None if args.no_calibrate else HeadCheck.load(args.calibration, args.neck_margin)
    if not args.no_calibrate and (head is None or args.recalibrate):
        if run_calibration():
            head = HeadCheck.load(args.calibration, args.neck_margin)
    if head is None:
        print("Head check is OFF (no calibration). Press c to calibrate.")

    feats, label, probs, state, last_run, head_delta = None, None, [], None, 0.0, None
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
            if feats is not None and feats.visibility >= MIN_VISIBILITY:
                probs = class_probs(bundle, feats.as_vector())
                label = max(probs, key=lambda t: t[1])[0]
                head_delta = None
                if head is not None:
                    head_delta = head.delta(feats.values)
                    if label == args.good and head.is_forward(feats.values):
                        label = HEAD_LABEL  # model has no head class; the calibrated rule adds it
            else:
                label, probs, head_delta = None, [], None  # unknown: ignored by the monitor
            state = monitor.update(label, now)

        fh, fw = frame.shape[:2]
        if feats is not None:
            draw_overlay(frame, feats)
        if label is None:
            put_text(frame, "No reliable pose", (10, fh - 80), (0, 255, 255))
        else:
            good = label == args.good
            shown = label if label == HEAD_LABEL else f"{label} ({max(p for _, p in probs):.0%})"
            put_text(frame, shown, (10, fh - 80), (0, 255, 0) if good else (0, 165, 255))
            put_text(frame, " | ".join(f"{c[:4]} {p:.2f}" for c, p in probs), (10, fh - 50),
                     (255, 255, 255), 0.55)
        if head is None:
            put_text(frame, "head check OFF (press c to calibrate)", (10, fh - 110), (0, 255, 255), 0.55)
        elif head_delta is not None:
            put_text(frame, f"head {head_delta:+.0f} deg vs baseline (limit +{head.margin:.0f})",
                     (10, fh - 110), (255, 255, 255), 0.55)
        if state is not None:
            put_text(frame, f"bad for {state['bad_for']:.1f}s (alert at {args.hold:.0f}s)",
                     (10, fh - 20), (255, 255, 255), 0.6)
            if state["alert"]:
                cv2.rectangle(frame, (0, 0), (fw - 1, fh - 1), (0, 0, 255), 14)
                put_text(frame, f"BAD POSTURE: {state['bad_label']}", (10, 30 + 24 * 7), (0, 0, 255), 0.8)

        cv2.imshow("Posture (q quit, c recalibrate)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c") and run_calibration():
            head = HeadCheck.load(args.calibration, args.neck_margin)
            monitor = PostureMonitor(hold=args.hold, good_label=args.good)
            state, label, head_delta, last_run = None, None, None, 0.0

    src.release()
    extractor.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()