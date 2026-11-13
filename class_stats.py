"""Per-class feature averages from the training data, to compare with the live
values shown on screen.  python class_stats.py"""
import csv
import sys
from collections import defaultdict

import numpy as np

from posture_core import FEATURE_NAMES

path = sys.argv[1] if len(sys.argv) > 1 else "datasets/features.csv"
rows = defaultdict(list)
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        if r["split"] == "train":
            rows[r["label"]].append([float(r[n]) for n in FEATURE_NAMES])

print(f"{'class':22s}{'n':>5s}  " + "  ".join(f"{n[:14]:>16s}" for n in FEATURE_NAMES))
for label, vals in sorted(rows.items()):
    a = np.array(vals)
    print(f"{label:22s}{len(a):5d}  " + "  ".join(f"{m:8.1f} ±{s:5.1f}" for m, s in zip(a.mean(0), a.std(0))))