"""
train.py — Malaria Severity Prediction Model (FINAL STABLE VERSION)
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# SKLEARN IMPORTS
# -------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# -------------------------------
# IMBLEARN (SMOTE only)
# -------------------------------
from imblearn.over_sampling import SMOTE

print("=" * 60)
print("  MALARIA SEVERITY PREDICTION — MODEL TRAINING")
print("=" * 60)

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("data/Malaria-Data.csv")
print(f"\n[DATA]  Loaded {df.shape[0]} patients, {df.shape[1]} columns")

TARGET   = "severe_maleria"
FEATURES = [c for c in df.columns if c != TARGET]

X = df[FEATURES]
y = df[TARGET]

print(f"[DATA]  Features  : {len(FEATURES)} total")
print(f"[DATA]  Balance   — 0: {(y==0).sum()}  |  1: {(y==1).sum()}")

# -------------------------------
# TRAIN / TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"\n[SPLIT] Train: {len(X_train)}  |  Test: {len(X_test)}")

# -------------------------------
# MODEL CANDIDATES
# -------------------------------
print("\n[MODELS] Evaluating candidates (STABLE MODE — no pipeline)...")

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
    "SVM (RBF)":           SVC(probability=True, random_state=42),
}

results = {}

# -------------------------------
# APPLY SMOTE ONCE (GLOBAL)
# -------------------------------
# This avoids repeated resampling inside loop (cleaner & faster)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# -------------------------------
# SCALE DATA
# -------------------------------
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# TRAIN + EVALUATE EACH MODEL
# -------------------------------
for name, clf in candidates.items():

    # Train model on resampled + scaled data
    clf.fit(X_resampled_scaled, y_resampled)

    # Predict probabilities on test set
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    # Compute ROC-AUC
    auc = roc_auc_score(y_test, y_proba)

    results[name] = auc

    print(f"  {name:<28}  ROC-AUC = {auc:.4f}")

# -------------------------------
# SELECT BEST MODEL
# -------------------------------
best_name = max(results, key=results.get)
best_clf  = candidates[best_name]

print(f"\n[BEST] '{best_name}' selected  (AUC = {results[best_name]:.4f})")

# -------------------------------
# FINAL MODEL TRAINING
# -------------------------------
# Retrain best model on full resampled data
best_clf.fit(X_resampled_scaled, y_resampled)

# -------------------------------
# FINAL EVALUATION
# -------------------------------
print("\n[EVAL] Test-set performance:")

y_pred  = best_clf.predict(X_test_scaled)
y_proba = best_clf.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm  = confusion_matrix(y_test, y_pred)

print(f"  Accuracy : {acc:.4f}")
print(f"  ROC-AUC  : {auc:.4f}")

print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
print(f"                Pred:0   Pred:1")
print(f"  Actual:0   TN={cm[0,0]:<5}  FP={cm[0,1]}")
print(f"  Actual:1   FN={cm[1,0]:<5}  TP={cm[1,1]}")

print(f"\n{classification_report(y_test, y_pred, target_names=['Not Severe','Severe'])}")

# -------------------------------
# FEATURE IMPORTANCE (if available)
# -------------------------------
if hasattr(best_clf, "feature_importances_"):
    importances = pd.Series(best_clf.feature_importances_, index=FEATURES).sort_values(ascending=False)

    print("\n[IMPORTANCE] Top predictors:")
    for feat, imp in importances.head(10).items():
        bar = "█" * int(imp * 50)
        print(f"  {feat:<20} {bar:<30}  {imp:.4f}")

# -------------------------------
# SAVE MODEL (UPDATED STRUCTURE)
# -------------------------------
joblib.dump(best_clf, "model/model.joblib")      # trained model
joblib.dump(scaler, "model/scaler.joblib")       # preprocessing scaler
joblib.dump(FEATURES, "model/features.joblib")   # feature list

print("\n[SAVED] model/model.joblib, scaler.joblib, features.joblib")
print("  Done. Next: update app.py to use new model format.\n")