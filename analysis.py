"""
Heart Disease UCI - Exploratory Data Analysis
Week 1 (Series 1) deliverable for the 3-week data analysis project.

Dataset: Heart Disease UCI (Cleveland database subset), 303 patient records,
14 attributes. Source: UCI Machine Learning Repository, mirrored as a clean
CSV at https://github.com/sharmaroshan/Heart-UCI-Dataset

Run with:
    python analysis.py
"""

import time

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "heart.csv"

# Human-readable names for the coded columns, used for prettier grouping output.
CP_LABELS = {
    0: "typical angina",
    1: "atypical angina",
    2: "non-anginal pain",
    3: "asymptomatic",
}
SEX_LABELS = {0: "female", 1: "male"}


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# 1. Import the dataset
section("1. IMPORT DATASET")
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from {DATA_PATH}")


# 2. Inspect the data
section("2. DATA INSPECTION - head()")
print(df.head())

section("2. DATA INSPECTION - info()")
df.info()

section("2. DATA INSPECTION - describe()")
print(df.describe())

section("2. DATA INSPECTION - missing values & duplicates")
print("Missing values per column:")
print(df.isnull().sum())
n_dupes = df.duplicated().sum()
print(f"\nDuplicate rows: {n_dupes}")
if n_dupes:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped duplicates -> {df.shape[0]} rows remain")


# 3. Basic filtering and grouping
section("3. FILTERING - patients over 50 diagnosed with heart disease")
older_with_disease = df[(df["age"] > 50) & (df["target"] == 1)]
print(
    f"{len(older_with_disease)} of {len(df)} patients are over 50 and "
    f"diagnosed with heart disease "
    f"({len(older_with_disease) / len(df):.1%} of the full dataset)"
)
print(older_with_disease[["age", "sex", "cp", "chol", "thalach", "target"]].head())

section("3. GROUPING - summary stats by sex")
df["sex_label"] = df["sex"].map(SEX_LABELS)
by_sex = df.groupby("sex_label").agg(
    count=("target", "count"),
    mean_age=("age", "mean"),
    mean_cholesterol=("chol", "mean"),
    mean_max_heart_rate=("thalach", "mean"),
    disease_rate=("target", "mean"),
)
print(by_sex.round(2))

section("3. GROUPING - summary stats by chest pain type (cp)")
df["cp_label"] = df["cp"].map(CP_LABELS)
by_cp = df.groupby("cp_label").agg(
    count=("target", "count"),
    mean_age=("age", "mean"),
    disease_rate=("target", "mean"),
).sort_values("disease_rate", ascending=False)
print(by_cp.round(2))

# 4. Explore a machine learning algorithm 
section("4. ML EXPLORATION - Logistic Regression")
feature_cols = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
X = df[feature_cols]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nConfusion matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["no disease", "disease"]))

# Which features push the prediction most (largest absolute coefficients)
coefs = pd.Series(model.coef_[0], index=feature_cols).sort_values(key=abs, ascending=False)
print("Top 5 most influential features (by |coefficient|):")
print(coefs.head())


# 5. Visualization
section("5. VISUALIZATION")
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="age",
    y="thalach",
    hue="target",
    palette={0: "#4C72B0", 1: "#C44E52"},
    alpha=0.8,
    ax=ax,
)
ax.set_title("Age vs. Maximum Heart Rate Achieved, by Heart Disease Diagnosis")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Maximum Heart Rate Achieved (thalach, bpm)")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ["No disease", "Disease"], title="Diagnosis")
fig.tight_layout()
fig.savefig("age_vs_heartrate_by_diagnosis.png", dpi=150)
print("Saved plot to age_vs_heartrate_by_diagnosis.png")


# 6. Bonus - Polars comparison
section("6. BONUS - Pandas vs Polars performance comparison")

t0 = time.perf_counter()
for _ in range(50):
    pdf = pd.read_csv(DATA_PATH, encoding="utf-8-sig").drop_duplicates()
    pdf.groupby("sex")["chol"].mean()
pandas_time = time.perf_counter() - t0

# Polars timing: equivalent read + groupby from scratch
t0 = time.perf_counter()
for _ in range(50):
    pldf = pl.read_csv(DATA_PATH, encoding="utf8-lossy").unique()
    pldf.group_by("sex").agg(pl.col("chol").mean())
polars_time = time.perf_counter() - t0

print(f"Pandas: {pandas_time:.4f}s for 50 iterations (read_csv + drop_duplicates + groupby)")
print(f"Polars: {polars_time:.4f}s for 50 iterations (read_csv + unique + group_by)")
faster_lib, slower_lib = ("Polars", "Pandas") if polars_time < pandas_time else ("Pandas", "Polars")
faster_time, slower_time = min(pandas_time, polars_time), max(pandas_time, polars_time)
print(
    f"{faster_lib} was faster on this run ({slower_time / faster_time:.2f}x vs. "
    f"{slower_lib}). At only 302 rows, both finish in milliseconds and the "
    "result is dominated by per-call overhead (e.g. Polars spinning up its "
    "thread pool) rather than by actual computation - so which library 'wins' "
    "here is more about measurement noise than a real performance signal. "
    "Polars' multithreaded, Rust-based engine tends to pull ahead once the "
    "dataset is large enough that computation time dominates overhead."
)

# Show the Polars groupby result once, to confirm it matches the Pandas one above
pldf = pl.read_csv(DATA_PATH, encoding="utf8-lossy").unique()
print("\nPolars groupby result (mean cholesterol by sex):")
print(pldf.group_by("sex").agg(pl.col("chol").mean().alias("mean_cholesterol")).sort("sex"))

print("\nDone.")
