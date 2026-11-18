# Posture Detection

A camera watches your side view while you sit. The program finds your body points, turns them into angles, and a small trained model says if the posture is good or bad. An ultrasonic sensor measures how close you are to the screen. When something is bad for most of about 10 seconds, an alert is sent to the backend, and the mobile app shows a notification.

## How it works

1. **Camera** sends a picture (RTSP stream).
2. **MediaPipe** finds ear, shoulder, hip and nose. We turn them into 5 angles (`posture_core.py`).
3. **Model** (Random Forest) turns the angles into a label: `goodposture`, `forwardbadposture` or `backwardbadposture`. A **head check** adds `headforward` when your head is further forward than in your calibration (`posture_monitor.py`).
4. **Alarm rule**: an alert turns on when at least 7 of the last 10 checks (one check per second) are bad, and off when fewer are.
5. **Commands**: the mobile app can ask the Pi to calibrate (see "Calibrate from the app").
6. **Backend** gets a live status every 5 s, an event when an alert starts or ends, and a summary every minute (`agent.py`, `backend.py`).

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
| `commands.py` | Receives calibration commands from the backend over a WebSocket |
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

## Calibrate from the app

The Pi has no screen, so the app can start the calibrations. The app sends `POST /api/commands/` to the backend, and the backend pushes the command to the Pi over the WebSocket `ws://<backend>/ws/device/` (the agent connects to it by itself, the address comes from `BACKEND_URL`).

| Command | What the Pi does |
| --- | --- |
| `calibrate_camera` | Waits 5 s (get ready, sit upright), then records 10 s and saves `calibration.json` |
| `calibrate_distance` | Waits 5 s, then records 8 s and saves `distance_baseline.json` |

The Pi sends progress back (`get_ready`, `recording`, `done` or `failed`), and the backend passes it to the app. While calibrating, the agent does nothing else, and the new calibration is used right away. If the Pi is offline the command is lost, because commands are not stored.

## Run

```powershell
python live_predict.py     # a window to check what the program decides
python agent.py            # the real program (no window, sends data to the backend)
```

`agent.py` has no options. Its settings are the constants at the top of the file (check every 1 s, alert at 7 bad out of the last 10, head margin 12 degrees, distance margin 15 cm). `.env` can turn the distance sensor off with `ULTRASONIC=off`. Stop it with Ctrl+C.

## Ultrasonic wiring (Raspberry Pi)

VCC to 5V (pin 2), GND to GND (pin 6), TRIG to GPIO23 (pin 16). ECHO gives 5 V but the Pi accepts only 3.3 V, so it goes through a voltage divider: ECHO, then R1, then GPIO24 (pin 18), then R2 to GND. R1 = R2 = 1.2 kΩ gives about 2.5 V (safe). R1 = 1.2 kΩ and R2 = 2.4 kΩ gives about 3.3 V. Test with `python ultrasonic.py read`. Not yet tested on real hardware.

## Raspberry Pi setup

- Use a virtual environment and `pip install -r requirements.txt` (MediaPipe works on the Pi).
- The distance sensor needs a GPIO library. On the Pi 5, gpiozero uses `lgpio`. Install it inside the venv:

```bash
pip install lgpio
python -c "import lgpio; print('lgpio ok')"
```

If `pip install lgpio` fails with `cannot find -llgpio`, install the C library and try again: `sudo apt install -y liblgpio-dev` (a different build error may also need `swig python3-dev`). Another way is to let the venv use the system package: `sudo apt install -y python3-lgpio`, then set `include-system-site-packages = true` in `.venv/pyvenv.cfg`. Check the sensor with `python ultrasonic.py read`. The MediaPipe "Sparsity tensors" and "madvise" messages at start are harmless.
- If the sensor cannot be opened, `agent.py` prints why and keeps running without distance.

## Troubleshooting

- **"No picture from the camera"**: check host, port, path and login in `.env`, and that the camera is on the same network.
- **"Model was trained with different features"**: run `extract_features.py` and `train.py` again.
- **Head check is OFF**: run `calibrate.py`.
- **Distance state is always "unknown"**: run `python ultrasonic.py baseline`, or check the wiring with `python ultrasonic.py read`.
- **Wrong labels**: use a true side view with ear, shoulder and hip visible, and compare your angles with `python class_stats.py`.