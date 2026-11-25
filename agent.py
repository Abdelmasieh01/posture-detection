"""Raspberry Pi agent. Run it with:   python agent.py

Once per second it looks at the camera and the distance sensor, decides if the posture is bad,
and sends the results to the backend. While a posture alert is on it also sends a few pictures.
Needs:
  models/posture_model.joblib      (from train.py)
  calibration.json                 (from calibrate.py, the camera baseline)
  distance_baseline.json           (from ultrasonic.py baseline)
Camera login and backend address are in .env (see .env.example).
"""
import datetime
import os
import signal
import sys
import time
import uuid

import cv2

import ultrasonic
from backend import Backend
from calibrate import calibrate
from camera import open_camera
from commands import CommandChannel
from config import camera_source, load_env, mask_url
from posture_core import FrameSmoother, PoseExtractor
from posture_monitor import (CALIBRATION_FILE, GOOD_LABEL, MODEL_FILE, Alarm, HeadCheck,
                             find_posture, load_model)

# ---- settings
INTERVAL = 1           # seconds between checks
ALERT_WINDOW = 10      # the alert looks at the last 10 checks (one check = one second)
ALERT_BAD = 7          # ...and turns on when at least 7 of them are bad (off again when fewer)
NECK_MARGIN = 12       # degrees the head may lean forward compared to the calibration
DISTANCE_MARGIN = 15   # cm closer than the usual distance counts as too close
STATUS_EVERY = 5       # seconds between status messages
FRAME_WIDTH = 640      # frames are shrunk to this width before the analysis
SAMPLES_PER_CHECK = 5  # MediaPipe looks at this many pictures per check; the median angles are used (steadier)
CALIBRATION_DELAY = 5     # seconds to get ready (and sit upright) after a calibration command
CALIBRATION_SECONDS = 10  # seconds the camera calibration records
DISTANCE_SECONDS = 8      # seconds the distance calibration records
SNAPSHOT_EVERY = 5     # seconds between pictures while a posture alert is on
SNAPSHOT_MAX = 3       # pictures per alert (the first one is sent right when the alert starts)
SNAPSHOT_WIDTH = 480   # pictures are shrunk to this width before sending
SNAPSHOT_QUALITY = 70  # JPEG quality


def iso(t):
    """Unix time -> '2026-11-14T03:50:41Z', the format the backend wants."""
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def find_distance(sensor, baseline):
    """Returns (distance in cm or None, 'ok' / 'tooclose' / 'unknown')."""
    if sensor is None:
        return None, "unknown"
    cm = ultrasonic.read_cm(sensor)
    if cm is None or baseline is None:
        return cm, "unknown"
    return cm, ("tooclose" if cm < baseline - DISTANCE_MARGIN else "ok")


def check_alert(kind, is_on, label, now, alerts, server):
    """Sends an event when an alert turns on or off. alerts[kind] is None, or a dict while on.
    Returns True when something was sent."""
    if is_on and alerts[kind] is None:
        alerts[kind] = {"start": now, "label": label, "event_id": None, "shots": 0, "last_shot": 0}
        event = {"state": "start", "label": label, "duration_s": None}
    elif not is_on and alerts[kind] is not None:
        alert, alerts[kind] = alerts[kind], None
        event = {"state": "end", "label": alert["label"], "duration_s": round(now - alert["start"], 1)}
    else:
        return False
    # the uuid is the event's id: if the same event is sent twice, the backend knows it is a repeat
    event.update(event_id=str(uuid.uuid4()), ts=iso(now), kind=kind)
    if event["state"] == "start":
        alerts[kind]["event_id"] = event["event_id"]  # the pictures point to this event
    server.send_later("/api/events/", event)
    print(f"Alert {event['state']}: {kind} ({event['label']})")
    return True


def send_snapshot(frame, label, alert, now, server):
    """While the posture alert is on: send a picture right away, then every SNAPSHOT_EVERY s, at most SNAPSHOT_MAX."""
    if alert is None or alert["shots"] >= SNAPSHOT_MAX or now - alert["last_shot"] < SNAPSHOT_EVERY:
        return
    small = cv2.resize(frame, (SNAPSHOT_WIDTH, int(frame.shape[0] * SNAPSHOT_WIDTH / frame.shape[1])))
    ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, SNAPSHOT_QUALITY])
    if not ok:
        return
    alert["shots"] += 1
    alert["last_shot"] = now
    server.send_image("/api/snapshots/", {"event_id": alert["event_id"], "ts": iso(now), "label": label}, jpeg.tobytes())


def new_minute(start):
    return {"start": start, "seconds": {}, "distances": [], "too_close": 0, "count": 0}


def add_to_minute(minute, label, distance, too_close):
    """Every check counts as INTERVAL seconds."""
    minute["count"] += 1
    minute["seconds"][label] = minute["seconds"].get(label, 0) + INTERVAL
    if distance is not None:
        minute["distances"].append(distance)
    if too_close:
        minute["too_close"] += INTERVAL


def minute_summary(minute):
    d = minute["distances"]
    return {"window_start": iso(minute["start"]), "window_seconds": 60,
            "seconds_per_label": minute["seconds"],
            "avg_distance_cm": round(sum(d) / len(d), 1) if d else None,
            "min_distance_cm": min(d) if d else None,
            "seconds_too_close": minute["too_close"], "detections": minute["count"],
            "avg_angles": None}


def run_command(name, cap, pose, sensor, channel):
    """Runs a calibration command from the app. While it runs, the agent does nothing else.
    It tells the app how it is going: get_ready, recording, then done or failed."""
    def tell(status, detail=""):
        print(f"Command {name}: {status} {detail}")
        channel.reply({"type": "command_result", "command": name, "status": status, "detail": detail})

    if name == "calibrate_camera":
        if calibrate(cap, pose, CALIBRATION_FILE, CALIBRATION_SECONDS, CALIBRATION_DELAY, window=False, on_progress=tell):
            tell("done")
        else:
            tell("failed", "not enough clear readings, make sure your side is visible")
    elif name == "calibrate_distance":
        if sensor is None:
            tell("failed", "no distance sensor")
            return
        tell("get_ready")
        time.sleep(CALIBRATION_DELAY)
        tell("recording")
        try:
            ultrasonic.record_baseline(sensor, DISTANCE_SECONDS)
            tell("done")
        except RuntimeError as error:
            tell("failed", str(error))
    else:
        tell("failed", "unknown command")


def main():
    load_env()
    server = Backend(os.environ.get("BACKEND_URL", "http://localhost:8000"))
    channel = CommandChannel(server.url.replace("http", "ws", 1) + "/ws/device/")  # http -> ws, https -> wss
    source = camera_source()
    model = load_model(MODEL_FILE)
    head = HeadCheck.load(CALIBRATION_FILE, NECK_MARGIN)  # None if calibrate.py was never run
    baseline = ultrasonic.load_baseline()                 # None if ultrasonic.py baseline was never run
    pose = PoseExtractor("lite")
    smoother = FrameSmoother()
    cap = open_camera(source)
    sensor = None
    if os.environ.get("ULTRASONIC", "on").lower() != "off":
        try:
            sensor = ultrasonic.open_sensor()
        except Exception as error:  # the posture part should still work without the distance sensor
            print(f"Distance sensor not available, continuing without it: {error}")

    print(f"Camera: {mask_url(source)} | backend: {server.url}")
    print("Head check:", "on" if head else "OFF (run calibrate.py)")
    print("Distance:", "off" if sensor is None else (f"baseline {baseline} cm" if baseline else "no baseline (run ultrasonic.py baseline)"))

    posture_alarm, distance_alarm = Alarm(ALERT_WINDOW, ALERT_BAD, GOOD_LABEL), Alarm(ALERT_WINDOW, ALERT_BAD, "ok")
    alerts = {"posture": None, "distance": None}
    minute = new_minute(0)
    last_check = time.time()  # the first check comes one INTERVAL from now
    last_sample = last_status = 0
    checks = 0
    signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))  # lets "systemctl stop" end the program cleanly

    try:
        while True:
            ok, frame = cap.read()
            if not ok:  # camera lost: wait, then open it again
                print("No picture from the camera, trying again in 2 s...")
                cap.release()
                time.sleep(2)
                cap = open_camera(source)
                continue
            command = channel.get()  # a calibration command from the app, if there is one
            if command and command.get("type") == "command":
                run_command(command.get("command"), cap, pose, sensor, channel)
                head = HeadCheck.load(CALIBRATION_FILE, NECK_MARGIN)  # use the new calibration right away
                baseline = ultrasonic.load_baseline()
                posture_alarm, distance_alarm = Alarm(ALERT_WINDOW, ALERT_BAD, GOOD_LABEL), Alarm(ALERT_WINDOW, ALERT_BAD, "ok")
                smoother = FrameSmoother()
                last_check = time.time()
                continue

            now = time.time()
            if now - last_sample >= INTERVAL / SAMPLES_PER_CHECK:  # the camera sends ~25 pictures a second, we only use a few
                last_sample = now
                small = cv2.resize(frame, (FRAME_WIDTH, int(frame.shape[0] * FRAME_WIDTH / frame.shape[1])))
                smoother.add(pose.extract(small))
            if now - last_check < INTERVAL:
                continue
            last_check = now

            # 1. what do we see? (median angles of the last few pictures)
            label, confidence, _ = find_posture(smoother.median(), model, head)
            distance, distance_state = find_distance(sensor, baseline)

            # 2. alerts (and pictures while the posture alert is on)
            posture_changed = check_alert("posture", posture_alarm.update(label), posture_alarm.bad_label, now, alerts, server)
            distance_changed = check_alert("distance", distance_alarm.update(distance_state), "tooclose", now, alerts, server)
            send_snapshot(frame, label, alerts["posture"], now, server)

            # 3. live status (right away when an alert changed, otherwise every STATUS_EVERY seconds)
            if posture_changed or distance_changed or now - last_status >= STATUS_EVERY:
                last_status = now
                server.send_status({"ts": iso(now), "label": label, "confidence": confidence,
                                    "distance_cm": distance, "distance_state": distance_state,
                                    "posture_alert": alerts["posture"] is not None,
                                    "distance_alert": alerts["distance"] is not None})

            # 4. one summary per minute
            start = int(now // 60) * 60
            if start != minute["start"]:
                if minute["count"] > 0:
                    server.send_later("/api/aggregates/", minute_summary(minute))
                minute = new_minute(start)
            add_to_minute(minute, label, distance, distance_state == "tooclose")

            server.flush()  # try again with anything that is still waiting
            checks += 1
            if checks % 10 == 0:
                print(f"{time.strftime('%H:%M:%S')}  {label}  distance {distance}  waiting to send: {len(server.waiting)}")
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping...")
        if minute["count"] > 0:
            server.send_later("/api/aggregates/", minute_summary(minute))
        channel.close()
        cap.release()


if __name__ == "__main__":
    main()