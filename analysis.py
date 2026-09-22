"""
Heart Disease UCI - reproducible data analysis workflow.

This module keeps the exploratory analysis from Assignment 2, while exposing
the main steps as reusable functions so they can be unit tested and run in CI.

Run with:
    python analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "heart.csv"
PLOT_PATH = "age_vs_heartrate_by_diagnosis.png"

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
REQUIRED_COLUMNS = FEATURE_COLS + ["target"]

CP_LABELS = {
    0: "typical angina",
    1: "atypical angina",
    2: "non-anginal pain",
    3: "asymptomatic",
}
SEX_LABELS = {0: "female", 1: "male"}


def section(title: str) -> None:
    """Print a readable section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _validate_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if required project columns are missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the heart-disease CSV file."""
    return pd.read_csv(path, encoding="utf-8-sig")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and remove duplicate rows without mutating input."""
    _validate_columns(df)
    return df.drop_duplicates().reset_index(drop=True).copy()


def prepare_features(df: pd.DataFrame):
    """Split a cleaned dataset into model features X and target y."""
    _validate_columns(df)

    X = df[FEATURE_COLS].copy()
    y = df["target"].copy()

    if X.isnull().any().any() or y.isnull().any():
        raise ValueError("Modeling data contains missing values.")
    if y.nunique() < 2:
        raise ValueError("Target must contain at least two classes.")

    return X, y


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split, standardize, and train a logistic-regression classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_train, X_test, y_train, y_test


def predict(model, scaler, X: pd.DataFrame):
    """Generate class predictions."""
    return model.predict(scaler.transform(X))


def evaluate_model(y_true: pd.Series, y_pred) -> dict:
    """Return classification metrics used by the project."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["no disease", "disease"],
            output_dict=True,
            zero_division=0,
        ),
    }


def create_visualization(
    df: pd.DataFrame,
    output_path: str | Path = PLOT_PATH,
) -> Path:
    """Save the age-vs-max-heart-rate visualization."""
    output_path = Path(output_path)

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
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def run_pipeline(
    data_path: str | Path = DATA_PATH,
    output_path: str | Path = PLOT_PATH,
) -> dict:
    """Run the complete data-analysis workflow end to end."""
    raw_df = load_data(data_path)
    clean_df = preprocess_data(raw_df)
    X, y = prepare_features(clean_df)

    model, scaler, X_train, X_test, y_train, y_test = train_model(X, y)
    y_pred = predict(model, scaler, X_test)
    metrics = evaluate_model(y_test, y_pred)
    plot_path = create_visualization(clean_df, output_path)

    return {
        "rows_loaded": len(raw_df),
        "rows_after_preprocessing": len(clean_df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": metrics["accuracy"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "plot_path": plot_path,
        "model": model,
        "scaler": scaler,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def main() -> None:
    """Run the analysis from the command line."""
    section("1. IMPORT AND PREPROCESS DATA")
    raw_df = load_data(DATA_PATH)
    clean_df = preprocess_data(raw_df)
    print(f"Loaded {len(raw_df)} rows; {len(clean_df)} rows after preprocessing.")

    section("2. MODEL AND EVALUATION")
    results = run_pipeline(DATA_PATH, PLOT_PATH)
    print(f"Train size: {results['train_size']}")
    print(f"Test size: {results['test_size']}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print("\nConfusion matrix:")
    print(results["confusion_matrix"])
    print("\nClassification report:")
    print(
        classification_report(
            results["y_test"],
            results["y_pred"],
            target_names=["no disease", "disease"],
            zero_division=0,
        )
    )

    section("3. VISUALIZATION")
    print(f"Saved plot to {results['plot_path']}")


if __name__ == "__main__":
    main()
