
from pathlib import Path
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "ETP_4_New_Data_Accidents.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading accident dataset...")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Place ETP_4_New_Data_Accidents.csv in the ml folder."
    )

df = pd.read_csv(DATA_FILE, low_memory=False)
df.columns = df.columns.astype(str).str.strip()

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Clean column names and common missing-value markers.
df = df.replace(
    ["", " ", "NA", "N/A", "na", "null", "NULL", "-", "?"],
    pd.NA
)
df = df.drop_duplicates()

# Find the accident-severity target column.
target_candidates = [
    c for c in df.columns
    if "severity" in c.lower()
]

if not target_candidates:
    print("\nCould not automatically find a severity column.")
    print("Available columns:")
    for col in df.columns:
        print("-", col)
    raise ValueError(
        "No column containing 'severity' was found. "
        "Check the printed columns and update the target column in the code."
    )

target = target_candidates[0]
print("\nSelected target column:", target)

# Remove rows where the target is missing.
df = df.dropna(subset=[target]).copy()
y_text = df[target].astype(str).str.strip()

# Remove columns that are not useful for prediction.
X = df.drop(columns=[target]).copy()

# Remove likely ID columns and columns with excessive missing values.
remove_cols = []

for col in X.columns:
    name = col.lower().strip()
    if name in ["id", "index", "serial number", "s.no", "sno"]:
        remove_cols.append(col)
    elif X[col].isna().mean() > 0.60:
        remove_cols.append(col)
    elif X[col].nunique(dropna=True) <= 1:
        remove_cols.append(col)

X = X.drop(columns=remove_cols, errors="ignore")

# Convert text columns to numeric when almost all values are numbers.
for col in X.select_dtypes(include="object").columns:
    converted = pd.to_numeric(X[col], errors="coerce")
    non_missing = X[col].notna().sum()
    if non_missing > 0 and converted.notna().sum() / non_missing >= 0.90:
        X[col] = converted

# Remove columns that look like unique record identifiers.
high_unique_cols = [
    col for col in X.columns
    if X[col].nunique(dropna=True) / max(len(X), 1) > 0.95
]
X = X.drop(columns=high_unique_cols, errors="ignore")

if X.shape[1] == 0:
    raise ValueError("No usable input features remain after cleaning.")

# Convert target labels to numeric class IDs.
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

print("\nTarget class names:", label_encoder.classes_.tolist())
print("\nTarget distribution:")
print(y_text.value_counts())

# Save dataset summary.
summary = {
    "rows_after_cleaning": int(len(df)),
    "number_of_features": int(X.shape[1]),
    "target_column": target,
    "target_classes": label_encoder.classes_.tolist(),
    "feature_columns": X.columns.tolist(),
    "removed_columns": remove_cols + high_unique_cols,
}
with open(OUTPUT_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, default=str)

# Save target distribution graph.
plt.figure(figsize=(9, 5))
y_text.value_counts().plot(kind="bar")
plt.title("Accident Severity Class Distribution")
plt.xlabel("Severity class")
plt.ylabel("Number of records")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "target_distribution.png")
plt.close()

# Separate numerical and categorical columns.
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [
    c for c in X.columns if c not in numeric_cols
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

transformers = []
if numeric_cols:
    transformers.append(("numeric", numeric_pipeline, numeric_cols))
if categorical_cols:
    transformers.append(("categorical", categorical_pipeline, categorical_cols))

preprocessor = ColumnTransformer(transformers=transformers)

# Use a stratified split when every class has at least two records.
class_counts = pd.Series(y).value_counts()
stratify_arg = y if class_counts.min() >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=stratify_arg
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
       n_jobs=1
    ),
}

# Compare models with cross-validation when possible.
results = {}
for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    smallest_class = pd.Series(y_train).value_counts().min()
    folds = min(5, int(smallest_class))

    if folds >= 2:
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=folds,
            scoring="f1_macro",
            n_jobs=None
        )
        results[name] = float(scores.mean())
        print(f"{name} cross-validation macro F1: {scores.mean():.4f}")
    else:
        results[name] = 0.0
        print(f"{name}: not enough examples per class for cross-validation.")

best_name = max(results, key=results.get)
print("\nSelected model:", best_name)

best_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", models[best_name]),
])

best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, average="macro", zero_division=0
)
recall = recall_score(
    y_test, predictions, average="macro", zero_division=0
)
f1 = f1_score(
    y_test, predictions, average="macro", zero_division=0
)

print("\n========== TEST RESULTS ==========")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Macro F1:  {f1:.4f}")
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=list(range(len(label_encoder.classes_))),
    target_names=[str(c) for c in label_encoder.classes_],
    zero_division=0
))

# Save confusion matrix.
cm = confusion_matrix(
    y_test,
    predictions,
    labels=list(range(len(label_encoder.classes_)))
)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted severity")
plt.ylabel("Actual severity")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.close()

# Save model and metadata for the Streamlit app.
joblib.dump(best_model, MODEL_DIR / "accident_severity_model.joblib")

metadata = {
    "target_column": target,
    "feature_columns": X.columns.tolist(),
    "numeric_columns": numeric_cols,
    "categorical_columns": categorical_cols,
    "target_classes": label_encoder.classes_.tolist(),
    "best_model": best_name,
    "metrics": {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    },
}

with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, default=str)

with open(OUTPUT_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metadata["metrics"], f, indent=4)

print("\nTraining completed successfully!")
print("Model saved in:", MODEL_DIR)
print("Graphs and reports saved in:", OUTPUT_DIR)
print("Next step: run streamlit run app.py")
