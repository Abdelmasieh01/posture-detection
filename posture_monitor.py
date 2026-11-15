"""Loads the model, decides the posture label from the angles, and holds the alert rule.
No camera or window code in here, so agent.py and live_predict.py share exactly the same logic."""
import datetime
import json
import os

import numpy as np

from posture_core import FEATURE_NAMES, MIN_VISIBILITY

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, "models", "posture_model.joblib")
CALIBRATION_FILE = os.path.join(HERE, "calibration.json")  # written by calibrate.py
GOOD_LABEL = "goodposture"
HEAD_LABEL = "headforward"


def load_model(path=MODEL_FILE):
    import joblib
    bundle = joblib.load(path)
    if list(bundle["feature_names"]) != list(FEATURE_NAMES):
        raise SystemExit("Model was trained with different features. Re-run extract_features.py and train.py.")
    return bundle


def class_probs(bundle, vector):
    """[(class, probability), ...] for one list of angles."""
    proba = bundle["model"].predict_proba(np.array([vector]))[0]
    return list(zip(bundle["classes"], (float(p) for p in proba)))


def save_calibration(path, samples):
    """samples: angle dicts recorded while sitting upright. Saves the median of each angle."""
    baseline = {name: float(np.median([s[name] for s in samples])) for name in FEATURE_NAMES}
    data = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
            "samples": len(samples), "baseline": baseline}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


class HeadCheck:
    """Is the head further forward than when this user sat upright (calibration)?

    neck_angle = how far the ear is in front of the shoulder, in degrees from vertical.
    The training data has no head-forward class, so this rule adds it."""

    def __init__(self, baseline_neck, margin=12.0):
        self.baseline_neck, self.margin = baseline_neck, margin

    @classmethod
    def load(cls, path=CALIBRATION_FILE, margin=12.0):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return cls(json.load(f)["baseline"]["neck_angle"], margin)

    def delta(self, values):
        return values["neck_angle"] - self.baseline_neck

    def is_forward(self, values):
        return self.delta(values) > self.margin


def find_posture(features, model, head):
    """features: what PoseExtractor.extract() returned (None = nobody found).
    Returns (label, confidence, probabilities)."""
    if features is None or features.visibility < MIN_VISIBILITY:
        return "unknown", None, []
    probs = class_probs(model, features.as_vector())
    label, confidence = max(probs, key=lambda pair: pair[1])
    if label == GOOD_LABEL and head is not None and head.is_forward(features.values):
        return HEAD_LABEL, None, probs
    return label, round(confidence, 2), probs


class Alarm:
    """Turns on after `hold` bad readings, off again when the person is mostly good.

    It remembers the last hold + 1 readings and is ON when at least `hold` of them are bad.
    So one wrong reading (a glitch) does not restart the timer.
    'unknown' (nobody in view) and None are never bad."""

    def __init__(self, hold=8, good_label=GOOD_LABEL):
        self.hold, self.good = hold, good_label
        self.recent = []
        self.bad_count = 0
        self.bad_label = None  # the most common bad label while the alarm is on

    def update(self, label):
        """Give it the newest reading. Returns True while the alarm is on."""
        self.recent.append(label)
        self.recent = self.recent[-(self.hold + 1):]
        bad = [x for x in self.recent if x not in (None, "unknown", self.good)]
        self.bad_count = len(bad)
        is_on = len(bad) >= self.hold
        self.bad_label = max(set(bad), key=bad.count) if is_on else None
        return is_on