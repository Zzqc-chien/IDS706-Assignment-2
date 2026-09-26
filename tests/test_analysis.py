"""Tests for the Heart Disease UCI analysis workflow."""

from pathlib import Path

import pandas as pd
import pytest

from analysis import (
    CLEAN_COLUMNS,
    FEATURE_COLS,
    RAW_TARGET_COL,
    TARGET_COL,
    compare_pandas_polars,
    count_iqr_outliers,
    cross_validate_models,
    group_summary,
    load_data,
    mark_disguised_missing,
    predict,
    prepare_features,
    preprocess_data,
    recode_target,
    run_pipeline,
    split_data,
    summarize_data_quality,
    train_model,
)

DATA_PATH = Path("heart.csv")


@pytest.fixture(scope="module")
def raw_df():
    return load_data(DATA_PATH)


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return preprocess_data(raw_df)


# --- Loading ---------------------------------------------------------------
def test_load_data_has_expected_schema(raw_df):
    assert isinstance(raw_df, pd.DataFrame)
    assert not raw_df.empty
    assert set(FEATURE_COLS + [RAW_TARGET_COL]).issubset(raw_df.columns)


def test_load_data_missing_file_raises_error(tmp_path):
    """Edge case: a missing CSV should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "does_not_exist.csv")


def test_load_data_empty_file_raises_error(tmp_path):
    """Edge case: an empty CSV should fail loudly, not return an empty frame."""
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(pd.errors.EmptyDataError):
        load_data(empty)


# --- Cleaning --------------------------------------------------------------
def test_mark_disguised_missing_only_replaces_sentinel_codes():
    df = pd.DataFrame({"thal": [0, 2, 3], "ca": [4, 0, 1]})
    marked = mark_disguised_missing(df)
    assert marked["thal"].isna().tolist() == [True, False, False]
    assert marked["ca"].isna().tolist() == [True, False, False]
    assert df["thal"].tolist() == [0, 2, 3]  # input not mutated


def test_recode_target_inverts_label():
    df = pd.DataFrame({RAW_TARGET_COL: [1, 0, 1]})
    recoded = recode_target(df)
    assert recoded[TARGET_COL].tolist() == [0, 1, 0]
    assert RAW_TARGET_COL not in recoded.columns


def test_preprocess_removes_duplicates_and_disguised_missing(raw_df, clean_df):
    assert len(raw_df) == 303  # input not mutated
    assert clean_df.duplicated().sum() == 0
    assert not clean_df.isna().any().any()
    assert (clean_df["thal"] != 0).all()
    assert (clean_df["ca"] != 4).all()
    assert list(clean_df.columns) == CLEAN_COLUMNS
    assert len(clean_df) == 296  # 303 - 1 duplicate - 6 disguised-missing rows


def test_preprocess_missing_column_raises(raw_df):
    """Edge case: a clear error names the missing column."""
    with pytest.raises(ValueError, match="chol"):
        preprocess_data(raw_df.drop(columns=["chol"]))


def test_summarize_data_quality_counts(raw_df):
    report = summarize_data_quality(raw_df)
    assert report["duplicates"] == 1
    assert report["disguised_missing"] == {"thal": 2, "ca": 4}
    assert report["rows_with_missing"] == 6


def test_count_iqr_outliers_flags_extreme_value():
    df = pd.DataFrame({"x": [10, 11, 12, 13, 14, 100], "flat": [5] * 6})
    counts = count_iqr_outliers(df, ["x", "flat"])
    assert counts == {"x": 1, "flat": 0}  # zero spread -> no outliers


# --- Domain sanity (guards the label fix) ----------------------------------
def test_asymptomatic_chest_pain_has_highest_disease_rate(clean_df):
    by_cp = group_summary(clean_df, "cp")
    assert by_cp.index[0] == "asymptomatic"
    assert by_cp["disease_rate"].between(0, 1).all()


def test_men_have_higher_disease_rate_than_women(clean_df):
    by_sex = group_summary(clean_df, "sex")
    assert by_sex.loc["male", "disease_rate"] > by_sex.loc["female", "disease_rate"]


# --- Modeling --------------------------------------------------------------
def test_prepare_features_returns_model_inputs(clean_df):
    X, y = prepare_features(clean_df)
    assert list(X.columns) == FEATURE_COLS
    assert y.name == TARGET_COL
    assert len(X) == len(y) == len(clean_df)
    assert set(y.unique()) == {0, 1}


def test_prepare_features_single_class_raises(clean_df):
    """Edge case: a one-class target cannot train a classifier."""
    with pytest.raises(ValueError, match="two classes"):
        prepare_features(clean_df[clean_df[TARGET_COL] == 1])


def test_prepare_features_nan_raises(clean_df):
    """Edge case: NaN reaching the model is rejected."""
    df = clean_df.copy()
    df.loc[0, "chol"] = float("nan")
    with pytest.raises(ValueError, match="missing values"):
        prepare_features(df)


def test_model_training_and_prediction(clean_df):
    X, y = prepare_features(clean_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model, scaler = train_model(X_train, y_train)
    predictions = predict(model, scaler, X_test)

    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})
    assert hasattr(model, "coef_")


def test_cross_validate_models_returns_both_models(clean_df):
    X, y = prepare_features(clean_df)
    results = cross_validate_models(X, y, n_splits=3)
    assert list(results.index) == ["Logistic regression", "Random forest"]
    assert results["accuracy_mean"].between(0, 1).all()


# --- Pandas vs Polars ------------------------------------------------------
def test_polars_matches_pandas_groupby():
    timing = compare_pandas_polars(DATA_PATH, n_iter=1)
    pandas_means = timing["pandas_result"].round(6).tolist()
    polars_means = timing["polars_result"]["chol"].round(6).to_list()
    assert pandas_means == polars_means


# --- System test -----------------------------------------------------------
def test_end_to_end_pipeline(tmp_path):
    result = run_pipeline(DATA_PATH, tmp_path, cv_splits=3)

    assert result["rows_after_preprocessing"] < result["rows_loaded"]
    assert result["train_size"] + result["test_size"] == (
        result["rows_after_preprocessing"]
    )
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["confusion_matrix"].shape == (2, 2)
    assert len(result["plot_paths"]) == 3
    for path in result["plot_paths"]:
        assert path.exists() and path.stat().st_size > 0
