"""Train a small posture classifier on the angle features.

  python train.py

Fits on the train split, picks the best model by macro-F1 on valid,
reports once on test, and saves models/posture_model.joblib.
"""
import argparse
import csv
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from posture_core import FEATURE_NAMES


def load(path):
    X, y, split = [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            X.append([float(r[n]) for n in FEATURE_NAMES])
            y.append(r["label"])
            split.append(r["split"])
    return np.array(X), np.array(y), np.array(split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="datasets/features.csv")
    ap.add_argument("--out", default="models/posture_model.joblib")
    args = ap.parse_args()

    X, y, split = load(args.features)
    tr, va, te = split == "train", split == "valid", split == "test"
    print(f"train {tr.sum()} | valid {va.sum()} | test {te.sum()}")

    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=0, n_jobs=-1),
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    }

    best_name, best_f1 = None, -1
    for name, model in candidates.items():
        model.fit(X[tr], y[tr])
        f1 = f1_score(y[va], model.predict(X[va]), average="macro")
        print(f"{name:20s} valid macro-F1: {f1:.3f}")
        if f1 > best_f1:
            best_name, best_f1 = name, f1

    model = candidates[best_name]
    print(f"\nBest: {best_name}")
    pred = model.predict(X[te])
    print("\nTEST report")
    print(classification_report(y[te], pred, digits=3))
    print("Confusion matrix (rows = true, cols = predicted), classes:", [str(c) for c in model.classes_])
    print(confusion_matrix(y[te], pred, labels=model.classes_))

    if best_name == "random_forest":
        print("\nFeature importance:")
        for n, imp in sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda t: -t[1]):
            print(f"  {n:18s} {imp:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES,
                 "classes": [str(c) for c in model.classes_]}, args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()