# train_diabetes.py
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

# 1) Load dataset (Pima Indians Diabetes from OpenML)
#    Features: preg, plas, pres, skin, insu, mass, pedi, age
#    Target: class -> tested_positive/negative
openml = fetch_openml(name="diabetes", version=1, as_frame=True)
df = openml.frame.copy()

# 2) Clean/prepare
# Replace biologically impossible zeros with NaN, then impute (standard practice for this dataset)
zero_as_missing = ["plas", "pres", "skin", "insu", "mass"]  # glucose, bp, skin, insulin, BMI
for col in zero_as_missing:
    df[col] = df[col].replace(0, np.nan)

# Map target to 0/1
df["class"] = df["class"].map({"tested_negative": 0, "tested_positive": 1}).astype(int)

X = df.drop(columns=["class"])
y = df["class"]

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4) Build pipeline: Impute -> Scale -> Logistic Regression
pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

pipe.fit(X_train, y_train)

# 5) Evaluate
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1": float(f1_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
}

print("✅ Metrics:", json.dumps(metrics, indent=2))

# 6) Save model + metadata
models_dir = Path("models"); models_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, models_dir / "diabetes_model.joblib")

metadata = {
    "features": list(X.columns),
    "target": "class",
    "model": "LogisticRegression (balanced, max_iter=1000)",
    "preprocessing": ["SimpleImputer(median)", "StandardScaler()"],
    "metrics": metrics,
}
with open(models_dir / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("💾 Saved:", str(models_dir / "diabetes_model.joblib"))
print("💾 Saved:", str(models_dir / "model_metadata.json"))