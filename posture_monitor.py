"""Model loading + smoothing/alert logic. No camera or GUI code, so the same
module runs on the Raspberry Pi later (alert -> RGB LED)."""
import collections
import datetime
import json
import os
import time

import numpy as np

from posture_core import FEATURE_NAMES


def load_model(path="models/posture_model.joblib"):
    import joblib
    bundle = joblib.load(path)
    if list(bundle["feature_names"]) != list(FEATURE_NAMES):
        raise SystemExit("Model was trained with different features. Re-run extract_features.py and train.py.")
    return bundle


def class_probs(bundle, vector):
    """Return [(class, probability), ...] for one feature vector."""
    proba = bundle["model"].predict_proba(np.array([vector]))[0]
    return list(zip(bundle["classes"], (float(p) for p in proba)))


def classify(bundle, vector):
    """Return (label, confidence) for one feature vector."""
    return max(class_probs(bundle, vector), key=lambda t: t[1])


class PostureMonitor:
    """Alerts when bad posture has lasted about `hold` seconds.

    - Majority vote over the last `smooth` predictions removes one-off glitches.
    - The bad streak starts at the first bad sample and resets only when the
      vote turns good, so short movement blips do not alert.
    - Unknown frames (no person / low visibility) are ignored; if nothing valid
      arrives for 2 x hold seconds the state resets.
    """

    def __init__(self, hold=8.0, smooth=3, good_label="goodposture"):
        self.hold, self.smooth, self.good = hold, smooth, good_label
        self.window = collections.deque(maxlen=smooth)  # (time, label)
        self.bad_since = None
        self.last_valid = None

    def update(self, label, t=None):
        t = time.time() if t is None else t
        if label is not None:
            self.window.append((t, label))
            self.last_valid = t
        elif self.last_valid is not None and t - self.last_valid > 2 * self.hold:
            self.window.clear()
            self.bad_since = None

        bad = [(ts, lab) for ts, lab in self.window if lab != self.good]
        if self.window and len(bad) > len(self.window) / 2:
            if self.bad_since is None:
                self.bad_since = min(ts for ts, _ in bad)
        else:
            self.bad_since = None

        bad_for = 0.0 if self.bad_since is None else t - self.bad_since
        alert = self.bad_since is not None and bad_for >= self.hold
        top_bad = collections.Counter(lab for _, lab in bad).most_common(1)[0][0] if alert else None
        return {"alert": alert, "bad_label": top_bad, "samples": len(self.window),
                "bad": len(bad), "bad_for": bad_for}


HEAD_LABEL = "headforward"


def save_calibration(path, samples):
    """samples: list of feature dicts recorded while sitting upright. Stores the median of each."""
    baseline = {n: float(np.median([smp[n] for smp in samples])) for n in FEATURE_NAMES}
    data = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
            "samples": len(samples), "baseline": baseline}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


class HeadCheck:
    """Rule-based forward-head check relative to the user's own upright baseline.

    neck_angle is the ear-over-shoulder angle from vertical (a proxy of the
    craniovertebral angle). The dataset has no head-forward class, so the model
    cannot learn it; this check fills that gap and doubles as per-user calibration.
    """

    def __init__(self, baseline_neck, margin=15.0):
        self.baseline_neck, self.margin = baseline_neck, margin

    @classmethod
    def load(cls, path, margin=15.0):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return cls(json.load(f)["baseline"]["neck_angle"], margin)

    def delta(self, values):
        return values["neck_angle"] - self.baseline_neck

    def is_forward(self, values):
        return self.delta(values) > self.margin