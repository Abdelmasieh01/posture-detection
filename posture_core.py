"""Shared posture logic: MediaPipe landmark extraction + angle features.

The SAME code is used for live testing now, dataset feature extraction and
training next, and inference on the Raspberry Pi later, so features always match.
"""
import collections
import math
import os
import urllib.request
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# MediaPipe landmark indices as (left, right)
NOSE = 0
EAR = (7, 8)
SHOULDER = (11, 12)
HIP = (23, 24)

# Feature order used everywhere (live display, training, Pi inference)
FEATURE_NAMES = ["neck_angle", "torso_angle", "alignment_angle", "head_offset", "head_pitch"]
MIN_VISIBILITY = 0.5  # below this the chosen side is probably occluded / unreliable


@dataclass
class PostureFeatures:
    side: str          # "left" or "right": the body side used (the one facing the camera)
    facing: int        # +1 = person faces image-right, -1 = image-left
    visibility: float  # mean visibility of the 3 points used: ear, shoulder, hip (0..1)
    values: dict = field(default_factory=dict)  # FEATURE_NAMES -> float
    points: dict = field(default_factory=dict)  # name -> (x, y) in pixels

    def as_vector(self):
        return [self.values[n] for n in FEATURE_NAMES]


def ensure_model(variant="lite"):
    """Download the MediaPipe pose model once and cache it in ./models."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"pose_landmarker_{variant}.task")
    if not os.path.exists(path):
        url = MODEL_URL.format(v=variant)
        print(f"Downloading pose model ({variant}) ...")
        urllib.request.urlretrieve(url, path)
    return path


def _angle_at(a, b, c):
    """Angle in degrees at point b between segments b->a and b->c (180 = straight line)."""
    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


def _from_vertical(p_from, p_to, facing):
    """Signed angle of the segment p_from->p_to relative to straight up.
    0 = vertical, positive = leaning toward the direction the person faces."""
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]  # image y points down
    return math.degrees(math.atan2(dx * facing, -dy))


def compute_features(lm, width, height, last_facing=1):
    """lm: sequence of 33 landmarks with .x/.y (normalized) and .visibility."""
    def vis(idx):
        return getattr(lm[idx], "visibility", 1.0) or 0.0

    score = {}
    for side, i in (("left", 0), ("right", 1)):
        score[side] = float(np.mean([vis(g[i]) for g in (EAR, SHOULDER, HIP)]))
    side = max(score, key=score.get)
    i = 0 if side == "left" else 1

    def pt(idx):
        return (lm[idx].x * width, lm[idx].y * height)  # pixels, so angles are not distorted

    ear, shoulder, hip = pt(EAR[i]), pt(SHOULDER[i]), pt(HIP[i])  # no knees: hidden by desks
    nose = pt(NOSE)

    # Facing direction: the nose is in front of the ear
    dx = nose[0] - ear[0]
    facing = (1 if dx > 0 else -1) if abs(dx) > 0.01 * width else last_facing

    torso_len = math.dist(hip, shoulder) + 1e-6
    values = {
        "neck_angle": _from_vertical(shoulder, ear, facing),      # head tilt forward (+)
        "torso_angle": _from_vertical(hip, shoulder, facing),     # lean forward (+) / back (-)
        "alignment_angle": _angle_at(ear, shoulder, hip),         # ~180 = ear-shoulder-hip straight
        "head_offset": (ear[0] - shoulder[0]) * facing / torso_len,  # ear ahead of shoulder, / torso length
        "head_pitch": math.degrees(math.atan2(nose[1] - ear[1], (nose[0] - ear[0]) * facing)),  # + = looking down
    }
    return PostureFeatures(
        side=side, facing=facing, visibility=score[side], values=values,
        points={"ear": ear, "shoulder": shoulder, "hip": hip, "nose": nose},
    )


class PoseExtractor:
    """Wraps MediaPipe PoseLandmarker (image mode: every frame is independent,
    exactly like dataset images and the Pi's 1 frame / 1-2 s capture)."""

    def __init__(self, variant="lite"):
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model(variant)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._facing = 1

    def extract(self, bgr_frame):
        """Return PostureFeatures, or None if no person was found."""
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.pose_landmarks:
            return None
        h, w = bgr_frame.shape[:2]
        feats = compute_features(result.pose_landmarks[0], w, h, self._facing)
        self._facing = feats.facing
        return feats

    def close(self):
        self.landmarker.close()


def _i(p):
    return int(p[0]), int(p[1])


def draw_overlay(frame, feats, values=None):
    """Draw the ear-shoulder-hip chain and the live feature values."""
    values = values or feats.values
    p = feats.points
    chain = ["ear", "shoulder", "hip"]
    for a, b in zip(chain, chain[1:]):
        cv2.line(frame, _i(p[a]), _i(p[b]), (0, 255, 0), 3)
    for k in chain:
        cv2.circle(frame, _i(p[k]), 6, (0, 0, 255), -1)
    sx, sy = _i(p["shoulder"])
    cv2.line(frame, (sx, sy - 120), (sx, sy + 120), (255, 255, 0), 1)  # vertical reference

    lines = [f"{n}: {values[n]:+.1f}" if n != "head_offset" else f"{n}: {values[n]:+.2f}"
             for n in FEATURE_NAMES]
    lines.append(f"side: {feats.side}  facing: {'right' if feats.facing > 0 else 'left'}")
    color = (0, 255, 255) if feats.visibility >= MIN_VISIBILITY else (0, 0, 255)
    lines.append(f"visibility: {feats.visibility:.2f}" +
                 ("" if feats.visibility >= MIN_VISIBILITY else "  LOW"))
    y = 28
    for text in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        y += 24
    return frame


class FrameSmoother:
    """MediaPipe's points move a little in every frame, even when you sit still.
    This collects the angles of several frames and gives the MEDIAN of each angle:
    one bad frame cannot move the result, and the noise gets about half as big."""

    def __init__(self):
        self.frames = []

    def add(self, features):
        """Give it what PoseExtractor.extract() returned (None and unclear frames are ignored)."""
        if features is not None and features.visibility >= MIN_VISIBILITY:
            self.frames.append(features)

    def median(self):
        """The median angles of the collected frames, as one PostureFeatures. Then forgets the frames.
        Returns None if fewer than 3 good frames were collected."""
        frames, self.frames = self.frames, []
        if len(frames) < 3:
            return None
        # Frames can disagree about which body side is visible or which way the person faces.
        # That changes the meaning (or the sign) of every angle, so keep only the most common case.
        side, facing = collections.Counter((f.side, f.facing) for f in frames).most_common(1)[0][0]
        frames = [f for f in frames if (f.side, f.facing) == (side, facing)]
        values = {name: float(np.median([f.values[name] for f in frames])) for name in FEATURE_NAMES}
        return PostureFeatures(side=side, facing=facing,
                               visibility=float(np.mean([f.visibility for f in frames])),
                               values=values, points=frames[-1].points)