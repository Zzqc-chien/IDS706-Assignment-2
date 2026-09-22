"""Tests for the Heart Disease UCI analysis workflow."""

from pathlib import Path

import pandas as pd
import pytest

from analysis import (
    FEATURE_COLS,
    load_data,
    predict,
    prepare_features,
    preprocess_data,
    run_pipeline,
    train_model,
)


DATA_PATH = Path("heart.csv")


def test_load_data_has_expected_schema():
    """Data loading should return the project dataset and expected columns."""
    df = load_data(DATA_PATH)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "target" in df.columns
    assert set(FEATURE_COLS).issubset(df.columns)


def test_load_data_missing_file_raises_error(tmp_path):
    """Edge case: a missing CSV should raise FileNotFoundError."""
    missing_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        load_data(missing_path)


def test_preprocess_removes_duplicates_without_mutating_input():
    """Preprocessing should remove duplicates and preserve the input DataFrame."""
    raw_df = load_data(DATA_PATH)
    original_length = len(raw_df)
    duplicate_count = int(raw_df.duplicated().sum())

    clean_df = preprocess_data(raw_df)

    assert len(raw_df) == original_length
    assert clean_df.duplicated().sum() == 0
    assert len(clean_df) == original_length - duplicate_count


def test_prepare_features_returns_model_inputs():
    """Feature preparation should create X and y with no missing values."""
    clean_df = preprocess_data(load_data(DATA_PATH))

    X, y = prepare_features(clean_df)

    assert list(X.columns) == FEATURE_COLS
    assert "target" not in X.columns
    assert y.name == "target"
    assert len(X) == len(y) == len(clean_df)
    assert not X.isnull().any().any()
    assert not y.isnull().any()


def test_model_training_and_prediction():
    """The trained model should return one valid binary prediction per row."""
    clean_df = preprocess_data(load_data(DATA_PATH))
    X, y = prepare_features(clean_df)

    model, scaler, _, X_test, _, y_test = train_model(X, y)
    predictions = predict(model, scaler, X_test)

    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})
    assert hasattr(model, "coef_")
    assert hasattr(scaler, "mean_")


def test_end_to_end_pipeline(tmp_path):
    """System test: load -> preprocess -> model -> evaluate -> visualization."""
    output_path = tmp_path / "test_plot.png"

    result = run_pipeline(DATA_PATH, output_path)

    assert result["rows_loaded"] > 0
    assert result["rows_after_preprocessing"] <= result["rows_loaded"]
    assert result["train_size"] + result["test_size"] == result["rows_after_preprocessing"]
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["confusion_matrix"].shape == (2, 2)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
