# predict_diabetes.py
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Feature order must match training
FEATURES = ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"]

def load_model():
    model_path = Path("models/diabetes_model.joblib")
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Train it first with train_diabetes.py")
    return joblib.load(model_path)

def main():
    parser = argparse.ArgumentParser(description="Predict diabetes risk (0=Low, 1=High)")
    for f in FEATURES:
        parser.add_argument(f"--{f}", type=float, required=True)
    args = parser.parse_args()

    # Build a single-row DataFrame in correct feature order
    values = [[getattr(args, f) for f in FEATURES]]
    X = pd.DataFrame(values, columns=FEATURES)

    model = load_model()
    prob = float(model.predict_proba(X)[:, 1][0])
    pred = int(prob >= 0.5)

    print(f"Risk (0=Low, 1=High): {pred}  |  Probability: {prob:.3f}")

if __name__ == "__main__":
    main()