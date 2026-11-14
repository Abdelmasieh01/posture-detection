# Posture Detection

Side-view sitting posture detection. A camera frame goes through MediaPipe Pose to get body landmarks, angles are computed from them, and a small trained classifier labels the posture. An alert fires only when bad posture lasts a few seconds.

- **Classes:** `goodposture`, `forwardbadposture`, `backwardbadposture` (model), plus `headforward` from a calibrated rule (the dataset has no head class)
- **Features:** `neck_angle`, `torso_angle`, `alignment_angle`, `head_offset`, `head_pitch` (ear, shoulder, hip and nose only; knees are not used)
- **Model:** Random Forest (scikit-learn), runs on CPU

## Requirements

- Python version supported by MediaPipe (3.10 to 3.12 is a safe choice)
- Internet on first run (downloads the MediaPipe pose model into `models/`)
- An RTSP camera on the same network (or a USB / phone camera for testing)

## Setup (Windows PowerShell)

```powershell
python -m venv posture
.\posture\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 1. Camera (RTSP is the default)

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

| Key | Meaning |
| --- | --- |
| `RTSP_USER` / `RTSP_PASSWORD` | Camera RTSP account |
| `RTSP_HOST` | Camera IP on the local network |
| `RTSP_PORT` | Usually `554` |
| `RTSP_PATH` | Stream path, camera-specific (e.g. `/stream1`, `/live`) |

Never commit `.env` (it is in `.gitignore`). Check the camera works:

```powershell
python live_angles.py
```

To use a USB or phone camera instead: `python live_angles.py --list` shows the working indices, then pass `--camera <index>`.

### 2. Dataset

Download a sitting-posture dataset from Roboflow Universe with the **Download Dataset** button, pick a CSV classification format, and unzip it into `datasets/raw` so it contains `train`, `valid` and `test`, each with a `_classes.csv`. Check your dataset's license and credit its author in your report.

Alternatively use `download_dataset.py` (needs a free Roboflow API key in `ROBOFLOW_API_KEY`; see the header of the file).

```powershell
python inspect_dataset.py datasets/raw
```

This prints class counts per split, checks for leakage between splits, and writes `contact_sheet.jpg`. Open it and confirm the images are side views.

### 3. Extract features and train

```powershell
python extract_features.py datasets/raw   # writes datasets/features.csv
python train.py                           # writes models/posture_model.joblib
```

Re-run both whenever the features in `posture_core.py` change.

### 4. Run live (calibrates itself on first start)

```powershell
python live_predict.py                    # RTSP camera from .env
python live_predict.py --camera 1         # USB / phone camera index
python live_predict.py --interval 1 --hold 8
```

- `--neck-margin`: degrees of head-forward beyond your baseline before it counts as bad (default 15); tune it using the "head ... vs baseline" line on screen
- `--interval`: seconds between detections (default 1)
- `--hold`: seconds of bad posture before the alert (default 8)
- **First start:** it asks you to sit upright and look at the screen for about 10 s, then saves `calibration.json`.
- **Keys:** `c` recalibrates (new user, or the camera moved), `q` quits.
- The red border simulates the future RGB LED.
- Other flags: `--recalibrate` (calibrate now), `--no-calibrate` (head check off), `--calib-seconds`.
- `python calibrate.py` also works on its own.

## Files

| File | Purpose |
| --- | --- |
| `posture_core.py` | Landmarks, angle features, overlay, camera / RTSP reader |
| `posture_monitor.py` | Model loading and the hold-time alert logic |
| `config.py` | Builds the RTSP URL from `.env` |
| `calibrate.py` | Records your upright baseline for the head check (also used by `live_predict.py`) |
| `live_angles.py` | Live camera view with angles (no model) |
| `live_predict.py` | Live classification with alert |
| `download_dataset.py`, `inspect_dataset.py` | Get and check the dataset |
| `extract_features.py`, `train.py` | Build features and train the model |
| `ultrasonic.py` | HC-SR04 distance on the Raspberry Pi: read, baseline, too-close alert (`--simulate` works on a laptop) |
| `class_stats.py` | Per-class feature averages, to compare with live values |

## Troubleshooting

- **"No frames from rtsp://..."**: check host, port, path and credentials in `.env`, that the camera is on the same network, and the stream path in the camera manual.
- **"Model was trained with different features"**: re-run `extract_features.py` and `train.py`.
- **Wrong camera opens**: run `python live_angles.py --list` and pass `--camera <index>`.
- **Dots land on wrong body parts or predictions are mostly "good"**: use a true side view, keep the shoulder, ear and hip visible, and compare live angles with `python class_stats.py`. If it persists, add your own labeled frames to the training data.
- **Head forward not detected**: press `c` to recalibrate while sitting upright, then adjust `--neck-margin` (raise it, e.g. 20, for fewer false alarms; lower it, e.g. 10, if it misses head-forward).
- **"No reliable pose"**: the body side facing the camera is too hidden; adjust camera position or lighting.

## Ultrasonic sensor (Raspberry Pi)

Wiring: VCC to 5V (pin 2), GND to GND (pin 6), TRIG to GPIO23 (pin 16). ECHO goes through a divider to GPIO24 (pin 18): ECHO, then R1, then GPIO24, then R2 to GND. Use R1 = R2 = 1.2 kΩ (about 2.5 V) or R1 = 1.2 kΩ and R2 = 2.4 kΩ (about 3.3 V).

```bash
python ultrasonic.py read        # live distance
python ultrasonic.py baseline    # sit normally, saves your usual distance
python ultrasonic.py watch       # too-close alert after --hold seconds (default 8)
```

Calibration is manual: compare the reading with a ruler, and if it is off by a constant, set `ULTRASONIC_OFFSET_CM` in `.env`. Add `--simulate` to try the logic on a laptop. The code is not yet tested on real hardware.

## Not built yet

Pi agent that sends data to the backend, the backend, and the mobile app.