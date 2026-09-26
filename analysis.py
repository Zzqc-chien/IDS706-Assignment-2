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
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_COL = "target"
REQUIRED_COLUMNS = FEATURE_COLS + [TARGET_COL]
TARGET_NAMES = ["no disease", "disease"]


def print_section_header(title: str) -> None:
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
    y = df[TARGET_COL].copy()

    if X.isnull().any().any() or y.isnull().any():
        raise ValueError("Modeling data contains missing values.")
    if y.nunique() < 2:
        raise ValueError("Target must contain at least two classes.")

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Create a stratified train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Standardize features and fit a logistic-regression classifier."""
    scaler = StandardScaler()
    model = LogisticRegression(max_iter=1000)
    model.fit(scaler.fit_transform(X_train), y_train)
    return model, scaler


def predict(model, scaler, X: pd.DataFrame):
    """Generate class predictions."""
    return model.predict(scaler.transform(X))


def evaluate_model(y_true: pd.Series, y_pred) -> dict:
    """Return classification metrics used by the project."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report_text": classification_report(
            y_true,
            y_pred,
            target_names=TARGET_NAMES,
            zero_division=0,
        ),
    }


def plot_age_vs_max_hr(
    df: pd.DataFrame,
    output_path: str | Path = PLOT_PATH,
) -> Path:
    """Save a scatter plot of age vs. maximum heart rate, colored by diagnosis."""
    output_path = Path(output_path)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="age",
        y="thalach",
        hue=TARGET_COL,
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
    """Run the complete workflow and return a summary of the results."""
    raw_df = load_data(data_path)
    clean_df = preprocess_data(raw_df)
    X, y = prepare_features(clean_df)

    X_train, X_test, y_train, y_test = split_data(X, y)
    model, scaler = train_model(X_train, y_train)
    metrics = evaluate_model(y_test, predict(model, scaler, X_test))

    return {
        "rows_loaded": len(raw_df),
        "rows_after_preprocessing": len(clean_df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        **metrics,
        "plot_path": plot_age_vs_max_hr(clean_df, output_path),
    }


def main() -> None:
    """Run the analysis from the command line."""
    results = run_pipeline(DATA_PATH, PLOT_PATH)

    print_section_header("1. IMPORT AND PREPROCESS DATA")
    print(
        f"Loaded {results['rows_loaded']} rows; "
        f"{results['rows_after_preprocessing']} rows after preprocessing."
    )

    print_section_header("2. MODEL AND EVALUATION")
    print(f"Train size: {results['train_size']}")
    print(f"Test size: {results['test_size']}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print("\nConfusion matrix:")
    print(results["confusion_matrix"])
    print("\nClassification report:")
    print(results["report_text"])

    print_section_header("3. VISUALIZATION")
    print(f"Saved plot to {results['plot_path']}")


if __name__ == "__main__":
    main()
