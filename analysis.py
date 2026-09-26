"""
Heart Disease UCI - reproducible data analysis workflow.

Question: which patient attributes are associated with a heart-disease
diagnosis, and how well can a simple model predict it?

Data note: the Kaggle/GitHub mirror of the Cleveland data (heart.csv) differs
from the original UCI file in two ways, both verified against
processed.cleveland.data:
  * the ``target`` column is inverted (1 = NO disease), and chest-pain codes
    run in reverse order (0 = asymptomatic, 3 = typical angina);
  * missing values are disguised as out-of-range codes (thal = 0, ca = 4).
This module corrects both before any analysis.

Run with:
    python analysis.py
Environment variables:
    DATA_PATH   input CSV (default: heart.csv)
    OUTPUT_DIR  where figures are written (default: figures)
"""

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.getenv("DATA_PATH", "heart.csv")
DEFAULT_OUTPUT_DIR = "figures"

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
RAW_TARGET_COL = "target"  # as shipped in heart.csv: 1 = no disease
TARGET_COL = "disease"  # corrected: 1 = heart disease present
RAW_COLUMNS = FEATURE_COLS + [RAW_TARGET_COL]
CLEAN_COLUMNS = FEATURE_COLS + [TARGET_COL]
TARGET_NAMES = ["no disease", "disease"]

# Codes that do not exist in the UCI data dictionary; they stand in for "?".
DISGUISED_MISSING = {"thal": 0, "ca": 4}
OUTLIER_COLS = ["trestbps", "chol", "thalach", "oldpeak"]

CP_LABELS = {
    0: "asymptomatic",
    1: "atypical angina",
    2: "non-anginal pain",
    3: "typical angina",
}
SEX_LABELS = {0: "female", 1: "male"}

COLOR_NO_DISEASE = "#2a78d6"
COLOR_DISEASE = "#eb6834"
TEXT_COLOR = "#52514e"


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------
def print_section_header(title: str) -> None:
    """Print a readable section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def get_output_dir() -> Path:
    """Return the figure directory, configurable through OUTPUT_DIR."""
    return Path(os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))


def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the heart-disease CSV file."""
    return pd.read_csv(path, encoding="utf-8-sig")


def mark_disguised_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace out-of-range sentinel codes (thal=0, ca=4) with NaN."""
    df = df.copy()
    for col, code in DISGUISED_MISSING.items():
        df[col] = df[col].mask(df[col] == code)
    return df


def recode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``disease`` (1 = present) by inverting the shipped ``target``."""
    df = df.copy()
    df[TARGET_COL] = 1 - df[RAW_TARGET_COL]
    return df.drop(columns=RAW_TARGET_COL)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, de-duplicate, drop disguised-missing rows, fix the label."""
    _validate_columns(df, RAW_COLUMNS)
    clean = df.drop_duplicates()
    clean = mark_disguised_missing(clean).dropna()
    clean = recode_target(clean)
    clean[["ca", "thal"]] = clean[["ca", "thal"]].astype(int)
    return clean.reset_index(drop=True)


def count_iqr_outliers(df: pd.DataFrame, cols: list[str]) -> dict[str, int]:
    """Count values outside 1.5 * IQR for each column (rows are kept)."""
    counts = {}
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outside = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        counts[col] = int(outside.sum())
    return counts


def summarize_data_quality(raw_df: pd.DataFrame) -> dict:
    """Report duplicates, disguised missing values, and IQR outliers."""
    deduped = raw_df.drop_duplicates()
    missing_rows = mark_disguised_missing(deduped).isna().any(axis=1)
    return {
        "duplicates": int(raw_df.duplicated().sum()),
        "disguised_missing": {
            col: int((deduped[col] == code).sum())
            for col, code in DISGUISED_MISSING.items()
        },
        "rows_with_missing": int(missing_rows.sum()),
        "iqr_outliers": count_iqr_outliers(deduped, OUTLIER_COLS),
    }


# ---------------------------------------------------------------------------
# Exploratory analysis
# ---------------------------------------------------------------------------
def filter_older_with_disease(df: pd.DataFrame, min_age: int = 50) -> pd.DataFrame:
    """Patients older than ``min_age`` who have heart disease."""
    return df[(df["age"] > min_age) & (df[TARGET_COL] == 1)]


def group_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Summary statistics and disease rate per group (sex or cp)."""
    labels = {"sex": SEX_LABELS, "cp": CP_LABELS}.get(by, {})
    summary = (
        df.assign(group=df[by].map(labels).fillna(df[by]))
        .groupby("group")
        .agg(
            count=(TARGET_COL, "size"),
            mean_age=("age", "mean"),
            mean_chol=("chol", "mean"),
            mean_max_hr=("thalach", "mean"),
            disease_rate=(TARGET_COL, "mean"),
        )
        .sort_values("disease_rate", ascending=False)
    )
    summary.index.name = by
    return summary.round(2)


def compare_pandas_polars(path: str | Path = DATA_PATH, n_iter: int = 20) -> dict:
    """Time the same read + de-duplicate + group-by in Pandas and Polars."""
    start = time.perf_counter()
    for _ in range(n_iter):
        pandas_result = (
            pd.read_csv(path, encoding="utf-8-sig")
            .drop_duplicates()
            .groupby("sex")["chol"]
            .mean()
        )
    pandas_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(n_iter):
        polars_result = (
            pl.read_csv(path, encoding="utf8-lossy")
            .unique()
            .group_by("sex")
            .agg(pl.col("chol").mean())
            .sort("sex")
        )
    polars_seconds = time.perf_counter() - start

    return {
        "pandas_seconds": pandas_seconds,
        "polars_seconds": polars_seconds,
        "pandas_result": pandas_result,
        "polars_result": polars_result,
    }


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame):
    """Split a cleaned dataset into model features X and target y."""
    _validate_columns(df, CLEAN_COLUMNS)

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


def feature_coefficients(model, feature_names: list[str]) -> pd.Series:
    """Standardized logistic-regression coefficients, largest effect first."""
    coefs = pd.Series(model.coef_[0], index=feature_names)
    return coefs.sort_values(key=abs, ascending=False)


def cross_validate_models(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Compare logistic regression and random forest with stratified k-fold CV."""
    models = {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state
        ),
    }
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=folds, scoring=["accuracy", "roc_auc"])
        rows.append(
            {
                "model": name,
                "accuracy_mean": scores["test_accuracy"].mean(),
                "accuracy_std": scores["test_accuracy"].std(),
                "roc_auc_mean": scores["test_roc_auc"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("model").round(3)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def _save(fig, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_age_vs_max_hr(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Scatter plot of age vs. maximum heart rate, colored by diagnosis."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="age",
        y="thalach",
        hue=TARGET_COL,
        palette={0: COLOR_NO_DISEASE, 1: COLOR_DISEASE},
        s=40,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )
    ax.set_title("Patients with heart disease reach lower maximum heart rates")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Maximum heart rate achieved (bpm)")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["No disease", "Disease"], title="Diagnosis")
    return _save(fig, Path(output_path))


def plot_disease_rate_by_cp(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Bar chart of disease rate per chest-pain type."""
    rates = group_summary(df, "cp")["disease_rate"].sort_values()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(rates.index, rates.values, color=COLOR_DISEASE, height=0.6)
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in rates.values], padding=4)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Asymptomatic chest pain carries the highest disease rate")
    ax.set_xlabel("Share of patients with heart disease")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    return _save(fig, Path(output_path))


def plot_feature_coefficients(coefs: pd.Series, output_path: str | Path) -> Path:
    """Horizontal bar chart of standardized logistic-regression coefficients."""
    ordered = coefs.reindex(coefs.abs().sort_values().index)
    colors = [COLOR_DISEASE if v > 0 else COLOR_NO_DISEASE for v in ordered]
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(ordered.index, ordered.values, color=colors, height=0.6)
    ax.axvline(0, color=TEXT_COLOR, linewidth=1)
    ax.set_title("What the logistic regression relies on (standardized features)")
    ax.set_xlabel("Coefficient  (orange = raises disease risk, blue = lowers it)")
    ax.grid(axis="y", visible=False)
    return _save(fig, Path(output_path))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    data_path: str | Path = DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cv_splits: int = 5,
) -> dict:
    """Run the complete workflow and return a summary of the results."""
    output_dir = Path(output_dir)
    raw_df = load_data(data_path)
    clean_df = preprocess_data(raw_df)
    X, y = prepare_features(clean_df)

    X_train, X_test, y_train, y_test = split_data(X, y)
    model, scaler = train_model(X_train, y_train)
    metrics = evaluate_model(y_test, predict(model, scaler, X_test))
    coefs = feature_coefficients(model, FEATURE_COLS)

    plot_paths = [
        plot_age_vs_max_hr(clean_df, output_dir / "age_vs_max_hr.png"),
        plot_disease_rate_by_cp(clean_df, output_dir / "disease_rate_by_cp.png"),
        plot_feature_coefficients(coefs, output_dir / "feature_coefficients.png"),
    ]

    return {
        "rows_loaded": len(raw_df),
        "rows_after_preprocessing": len(clean_df),
        "data_quality": summarize_data_quality(raw_df),
        "older_with_disease": len(filter_older_with_disease(clean_df)),
        "by_sex": group_summary(clean_df, "sex"),
        "by_cp": group_summary(clean_df, "cp"),
        "train_size": len(X_train),
        "test_size": len(X_test),
        **metrics,
        "coefficients": coefs,
        "cv_results": cross_validate_models(X, y, n_splits=cv_splits),
        "plot_paths": plot_paths,
    }


def main() -> None:
    """Run the analysis from the command line."""
    results = run_pipeline(DATA_PATH, get_output_dir())
    quality = results["data_quality"]

    print_section_header("1. DATA QUALITY AND CLEANING")
    print(f"Rows loaded: {results['rows_loaded']}")
    print(f"Duplicate rows removed: {quality['duplicates']}")
    print(f"Disguised missing codes: {quality['disguised_missing']}")
    print(f"Rows dropped for missing values: {quality['rows_with_missing']}")
    print(f"Rows after cleaning: {results['rows_after_preprocessing']}")
    print(f"IQR outliers (flagged, kept): {quality['iqr_outliers']}")

    print_section_header("2. FILTERING AND GROUPING")
    n_clean = results["rows_after_preprocessing"]
    older = results["older_with_disease"]
    print(
        f"{older} of {n_clean} patients ({older / n_clean:.1%}) are over 50 "
        "and have heart disease."
    )
    print("\nBy sex:")
    print(results["by_sex"])
    print("\nBy chest pain type:")
    print(results["by_cp"])

    print_section_header("3. LOGISTIC REGRESSION (80/20 HOLD-OUT)")
    print(f"Train size: {results['train_size']}, Test size: {results['test_size']}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print("\nConfusion matrix:")
    print(results["confusion_matrix"])
    print("\nClassification report:")
    print(results["report_text"])
    print("Top 5 features by |standardized coefficient|:")
    print(results["coefficients"].head().round(3))

    print_section_header("4. MODEL COMPARISON (5-FOLD STRATIFIED CV)")
    print(results["cv_results"])

    print_section_header("5. VISUALIZATION")
    for path in results["plot_paths"]:
        print(f"Saved {path}")

    print_section_header("6. BONUS - PANDAS VS POLARS")
    timing = compare_pandas_polars(DATA_PATH)
    print(f"Pandas: {timing['pandas_seconds']:.4f}s for 20 runs")
    print(f"Polars: {timing['polars_seconds']:.4f}s for 20 runs")
    print("At ~300 rows the gap reflects per-call overhead, not computation speed.")


if __name__ == "__main__":
    main()
