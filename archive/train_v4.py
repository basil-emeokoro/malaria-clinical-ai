"""
train_v4.py
------------------------------------------------
SMOTE-Enhanced Stable Malaria Model
------------------------------------------------

Purpose:
✅ Clean schema
✅ SMOTE only on training data
✅ Stable RandomForest
✅ Better severe-case learning
✅ SHAP-ready architecture
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

print("=" * 60)
print(" SMOTE-ENHANCED MALARIA MODEL TRAINING (V4)")
print("=" * 60)

df = pd.read_csv("data/Malaria-Data.csv")

TARGET = "severe_malaria"

FEATURES = [
    "age",
    "sex",
    "fever",
    "cold",
    "rigor",
    "fatigue",
    "headache",
    "bitter_tongue",
    "vomiting",
    "diarrhea",
    "convulsion",
    "anemia",
    "jaundice",
    "coca_cola_urine",
    "hypoglycemia",
    "prostration",
    "hyperpyrexia",
]

required_columns = FEATURES + [TARGET]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")

X = df[FEATURES]
y = df[TARGET]

print(f"\n[DATA] Rows: {len(df)}")
print(f"[DATA] Features: {len(FEATURES)}")
print("[DATA] Original class balance:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print(f"\n[SPLIT] Train: {len(X_train)}")
print(f"[SPLIT] Test : {len(X_test)}")

# ------------------------------------------------
# SMOTE ONLY ON TRAINING DATA
# ------------------------------------------------
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\n[SMOTE] Resampled training class balance:")
print(y_train_smote.value_counts())

# ------------------------------------------------
# STABLE RANDOM FOREST
# ------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=3,
    random_state=42
)

print("\n[TRAINING] RandomForestClassifier + SMOTE...")
model.fit(X_train_smote, y_train_smote)

# ------------------------------------------------
# EVALUATION
# ------------------------------------------------
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
        y_pred,
        target_names=["Not Severe", "Severe"]
    )
)

# ------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------
importances = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print("\n[TOP FEATURES]")

for feat, imp in importances.head(10).items():
    bar = "█" * int(imp * 50)
    print(f"{feat:<20} {bar:<30} {imp:.4f}")

# ------------------------------------------------
# SAVE MODEL
# ------------------------------------------------
joblib.dump(model, "model/model_v4.joblib")
joblib.dump(FEATURES, "model/features_v4.joblib")

print("\n[SAVED]")
print("model/model_v4.joblib")
print("model/features_v4.joblib")

print("\nTraining Complete.")