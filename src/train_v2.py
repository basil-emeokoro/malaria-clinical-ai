"""
train_v2.py
----------------------------------------
Clean Clinical ML Training Pipeline
----------------------------------------

Goals:
✅ No scaling
✅ Balanced RandomForest
✅ Stable inference
✅ Explainability-ready
✅ SHAP-ready architecture
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from sklearn.ensemble import (
    RandomForestClassifier
)

# ----------------------------------------
# LOAD DATA
# ----------------------------------------

print("=" * 60)
print(" CLEAN MALARIA MODEL TRAINING (V2)")
print("=" * 60)

df = pd.read_csv("data/Malaria-Data.csv")

TARGET = "severe_maleria"

FEATURES = [
    c for c in df.columns
    if c != TARGET
]

X = df[FEATURES]
y = df[TARGET]

print(f"\n[DATA] Rows: {len(df)}")
print(f"[DATA] Features: {len(FEATURES)}")

# ----------------------------------------
# TRAIN / TEST SPLIT
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print(f"\n[SPLIT] Train: {len(X_train)}")
print(f"[SPLIT] Test : {len(X_test)}")

# ----------------------------------------
# MODEL
# ----------------------------------------
# IMPORTANT:
# Tree models DO NOT need scaling.
#
# We now use:
# ✅ balanced class weights
# ✅ stable architecture
# ✅ SHAP compatibility
# ----------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42
)

print("\n[TRAINING] RandomForestClassifier...")
model.fit(X_train, y_train)

# ----------------------------------------
# EVALUATION
# ----------------------------------------

print("\n[EVALUATION]")

y_pred = model.predict(X_test)

y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)

auc = roc_auc_score(y_test, y_proba)

cm = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy : {acc:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# ----------------------------------------
# FEATURE IMPORTANCE
# ----------------------------------------

importances = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print("\n[TOP FEATURES]")

for feat, imp in importances.head(10).items():

    bar = "█" * int(imp * 50)

    print(
        f"{feat:<20} "
        f"{bar:<30} "
        f"{imp:.4f}"
    )

# ----------------------------------------
# SAVE MODEL
# ----------------------------------------

joblib.dump(
    model,
    "model/model_v2.joblib"
)

joblib.dump(
    FEATURES,
    "model/features_v2.joblib"
)

print("\n[SAVED]")
print("model/model_v2.joblib")
print("model/features_v2.joblib")

print("\nTraining Complete.")