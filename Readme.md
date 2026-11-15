# Posture Detection

A camera watches your side view while you sit. The program finds your body points, turns them into angles, and a small trained model says if the posture is good or bad. An ultrasonic sensor measures how close you are to the screen. When something is bad for about 8 seconds, an alert is sent to the backend, and the mobile app shows a notification.

## How it works

1. **Camera** sends a picture (RTSP stream).
2. **MediaPipe** finds ear, shoulder, hip and nose. We turn them into 5 angles (`posture_core.py`).
3. **Model** (Random Forest) turns the angles into a label: `goodposture`, `forwardbadposture` or `backwardbadposture`. A **head check** adds `headforward` when your head is further forward than in your calibration (`posture_monitor.py`).
4. **Alarm rule**: an alert turns on when 8 of the last 9 checks (one check per second) are bad.
5. **Backend** gets a live status every 5 s, an event when an alert starts or ends, and a summary every minute (`agent.py`, `backend.py`).

## Files

Runs on the Pi:

| File | What it does |
| --- | --- |
| `agent.py` | The main program: check once per second, decide, send to the backend |
| `posture_core.py` | Picture to body points to angles |
| `posture_monitor.py` | Loads the model, chooses the label, head check, alarm rule |
| `camera.py` | Opens the camera or RTSP stream |
| `ultrasonic.py` | Reads the distance sensor (`read` and `baseline` commands) |
| `backend.py` | Sends data to the backend and retries when it is down |
| `config.py` | Reads `.env` and builds the RTSP address |

Tools you use before running:

| File | What it does |
| --- | --- |
| `download_dataset.py`, `inspect_dataset.py` | Get and look at the training pictures |
| `extract_features.py`, `train.py`, `class_stats.py` | Make the angle table, train the model, look at the angles per class |
| `calibrate.py` | Records how you sit upright |
| `live_angles.py`, `live_predict.py` | Windows to check the camera view and the decisions |

## Setup (Windows PowerShell)

```powershell
python -m venv posture
.\posture\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill `.env` (see `.env.example`): the camera login (`RTSP_*`) and the backend address (`BACKEND_URL`). For a USB or phone camera, set `CAMERA_SOURCE=1` (the camera number).

## Train the model (once)

1. Download a sitting-posture dataset (Roboflow Universe, CSV classification format) into `datasets/raw`. Check its license.
2. `python inspect_dataset.py datasets/raw` shows the class counts and saves `contact_sheet.jpg`. Check the pictures are side views.
3. `python extract_features.py datasets/raw` writes `datasets/features.csv`.
4. `python train.py` writes `models/posture_model.joblib` and prints the test accuracy.

## Calibrate (manual, once per person and camera position)

- Camera: `python calibrate.py`. Sit upright for about 10 seconds. Saves `calibration.json`.
- Distance (on the Pi): `python ultrasonic.py baseline`. Sit normally. Saves `distance_baseline.json`.
- Distance offset: compare the sensor with a ruler. If it is always off by the same amount, set `ULTRASONIC_OFFSET_CM` in `.env`.

## Run

```powershell
python live_predict.py     # a window to check what the program decides
python agent.py            # the real program (no window, sends data to the backend)
```

`agent.py` has no options. Its settings are the constants at the top of the file (check every 1 s, alert after 8, head margin 12 degrees, distance margin 15 cm). `.env` can turn the distance sensor off with `ULTRASONIC=off`. Stop it with Ctrl+C.

## Ultrasonic wiring (Raspberry Pi)

VCC to 5V (pin 2), GND to GND (pin 6), TRIG to GPIO23 (pin 16). ECHO gives 5 V but the Pi accepts only 3.3 V, so it goes through a voltage divider: ECHO, then R1, then GPIO24 (pin 18), then R2 to GND. R1 = R2 = 1.2 kΩ gives about 2.5 V (safe). R1 = 1.2 kΩ and R2 = 2.4 kΩ gives about 3.3 V. Test with `python ultrasonic.py read`. Not yet tested on real hardware.

## Troubleshooting

- **"No picture from the camera"**: check host, port, path and login in `.env`, and that the camera is on the same network.
- **"Model was trained with different features"**: run `extract_features.py` and `train.py` again.
- **Head check is OFF**: run `calibrate.py`.
- **Distance state is always "unknown"**: run `python ultrasonic.py baseline`, or check the wiring with `python ultrasonic.py read`.
- **Wrong labels**: use a true side view with ear, shoulder and hip visible, and compare your angles with `python class_stats.py`.