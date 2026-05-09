"""
train_v2.py
------------------------------------------------
Clean Clinical ML Training Pipeline (V2)
------------------------------------------------

Goals:
✅ No scaling
✅ Balanced RandomForest
✅ Stable inference
✅ Explainability-ready
✅ SHAP-ready architecture
"""

# ------------------------------------------------
# WARNING CONTROL
# ------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------
# IMPORTS
# ------------------------------------------------
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from imblearn.ensemble import (
    BalancedRandomForestClassifier
)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
print("=" * 60)
print(" CLEAN MALARIA MODEL TRAINING (V2)")
print("=" * 60)

# Load dataset
df = pd.read_csv("data/Malaria-Data.csv")

# Target column
TARGET = "severe_maleria"

# Feature columns
FEATURES = [
    c for c in df.columns
    if c != TARGET
]

# Split into features and target
X = df[FEATURES]
y = df[TARGET]

print(f"\n[DATA] Rows: {len(df)}")
print(f"[DATA] Features: {len(FEATURES)}")

# ------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------
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
# MODEL
# ------------------------------------------------
# IMPORTANT:
# Tree-based models do NOT require scaling.
#
# BalancedRandomForest helps:
# ✅ severe-case learning
# ✅ class imbalance handling
# ✅ clinical consistency
# ✅ better sensitivity
# ✅ explainability readiness
# ------------------------------------------------

model = BalancedRandomForestClassifier(

    n_estimators=300,

    max_depth=10,

    random_state=42

)

print("\n[TRAINING] BalancedRandomForestClassifier...")

# Train model
model.fit(X_train, y_train)

# ------------------------------------------------
# EVALUATION
# ------------------------------------------------
print("\n[EVALUATION]")

# Predictions
y_pred = model.predict(X_test)

# Probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Metrics
acc = accuracy_score(y_test, y_pred)

auc = roc_auc_score(y_test, y_proba)

cm = confusion_matrix(y_test, y_pred)

# Display metrics
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

    print(
        f"{feat:<20} "
        f"{bar:<30} "
        f"{imp:.4f}"
    )

# ------------------------------------------------
# SAVE MODEL
# ------------------------------------------------
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