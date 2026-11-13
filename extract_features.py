"""Run MediaPipe on every dataset image and save angle features to a CSV.

  python extract_features.py datasets/raw

Skips: Unlabeled and multi-label images, images with no person found,
and images where the visible body side is below the visibility threshold.
"""
import argparse
import collections
import csv
import os

import cv2

from inspect_dataset import collect
from posture_core import FEATURE_NAMES, MIN_VISIBILITY, PoseExtractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="datasets/features.csv")
    ap.add_argument("--variant", default="lite", choices=["lite", "full"])
    args = ap.parse_args()

    items, layout, _ = collect(args.root)
    extractor = PoseExtractor(args.variant)
    stats = collections.Counter()
    kept = collections.Counter()  # (split, label) -> n

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "split", "label", "side", "facing", "visibility"] + FEATURE_NAMES)
        for n, (path, split, labs) in enumerate(items, 1):
            if len(labs) != 1 or labs[0].lower() == "unlabeled":
                stats["skipped_label"] += 1
                continue
            img = cv2.imread(path)
            if img is None:
                stats["unreadable"] += 1
                continue
            feats = extractor.extract(img)
            if feats is None:
                stats["no_person"] += 1
                continue
            if feats.visibility < MIN_VISIBILITY:
                stats["low_visibility"] += 1
                continue
            w.writerow([path, split, labs[0], feats.side, feats.facing, f"{feats.visibility:.3f}"]
                       + [f"{v:.4f}" for v in feats.as_vector()])
            stats["kept"] += 1
            kept[(split, labs[0])] += 1
            if n % 100 == 0:
                print(f"{n}/{len(items)} ...")

    extractor.close()
    print("\n", dict(stats))
    for (split, label), c in sorted(kept.items()):
        print(f"  {split:6s} {label:22s} {c}")
    print(f"\nSaved {args.out}. Next: python train.py")


if __name__ == "__main__":
    main()