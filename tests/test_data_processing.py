import pytest
import pandas as pd
import numpy as np
from src.data_processing import FeatureEngineer


# ── Sample data we use in all tests ─────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Create a small fake dataset that looks like the real Xente data."""
    return pd.DataFrame({
        "TransactionId": ["T1", "T2", "T3", "T4", "T5"],
        "CustomerId": ["C1", "C1", "C2", "C2", "C3"],
        "AccountId": ["A1", "A1", "A2", "A2", "A3"],
        "Amount": [1000, -200, 5000, 300, 1500],
        "Value": [1000, 200, 5000, 300, 1500],
        "ProductCategory": ["airtime", "airtime", "financial_services",
                             "utility_bill", "airtime"],
        "ChannelId": ["ChannelId_1", "ChannelId_2", "ChannelId_3",
                      "ChannelId_1", "ChannelId_2"],
        "ProviderId": ["ProviderId_1", "ProviderId_2", "ProviderId_1",
                       "ProviderId_3", "ProviderId_1"],
        "PricingStrategy": [2, 2, 1, 3, 2],
        "TransactionStartTime": [
            "2018-11-15T02:18:49Z",
            "2018-11-15T10:30:00Z",
            "2018-12-01T08:00:00Z",
            "2018-12-15T14:00:00Z",
            "2019-01-10T20:00:00Z",
        ],
        "FraudResult": [0, 0, 0, 1, 0],
    })


# ── Test 1: Pipeline runs without errors ────────────────────────────────────
def test_pipeline_runs(sample_df):
    """The full pipeline should run without raising any errors."""
    fe = FeatureEngineer()
    result = fe.fit_transform(sample_df)
    assert result is not None


# ── Test 2: Output has more columns than input ───────────────────────────────
def test_output_has_more_columns(sample_df):
    """After feature engineering, we should have more columns than before."""
    fe = FeatureEngineer()
    result = fe.fit_transform(sample_df)
    assert result.shape[1] > sample_df.shape[1]


# ── Test 3: Aggregate features are created ──────────────────────────────────
def test_aggregate_features_created(sample_df):
    """Customer-level aggregate features should exist in output."""
    fe = FeatureEngineer()
    result = fe.fit_transform(sample_df)
    assert "transaction_count" in result.columns
    assert "total_spend" in result.columns
    assert "avg_value" in result.columns


# ── Test 4: Datetime features are extracted ──────────────────────────────────
def test_datetime_features_extracted(sample_df):
    """Hour, day, month features should be created from timestamp."""
    fe = FeatureEngineer()
    result = fe.fit_transform(sample_df)
    assert "tx_hour" in result.columns
    assert "tx_month" in result.columns
    assert "tx_dayofweek" in result.columns


# ── Test 5: Log transform column is created ──────────────────────────────────
def test_log_transform_created(sample_df):
    """Amount_log column should exist after transformation."""
    fe = FeatureEngineer()
    result = fe.fit_transform(sample_df)
    assert "Amount_log" in result.columns


# ── Test 6: No missing values after pipeline ─────────────────────────────────
def test_no_missing_values(sample_df):
    """Pipeline should produce no missing values in key columns."""
    fe = FeatureEngineer()
    result = fe.fit_transform(sample_df)
    assert result["transaction_count"].isnull().sum() == 0
    assert result["tx_hour"].isnull().sum() == 0


# ── Test 7: Empty dataframe raises error ─────────────────────────────────────
def test_empty_dataframe_raises_error():
    """Passing an empty DataFrame should raise a ValueError."""
    fe = FeatureEngineer()
    with pytest.raises(ValueError):
        fe.fit_transform(pd.DataFrame())


# ── Test 8: Wrong input type raises error ────────────────────────────────────
def test_wrong_input_type_raises_error():
    """Passing a list instead of DataFrame should raise TypeError."""
    fe = FeatureEngineer()
    with pytest.raises(TypeError):
        fe.fit_transform([1, 2, 3])


# ── Test 9: Transform without fit raises error ───────────────────────────────
def test_transform_without_fit_raises_error(sample_df):
    """Calling transform before fit_transform should raise ValueError."""
    fe = FeatureEngineer()
    with pytest.raises(ValueError):
        fe.transform(sample_df)


# ── Test 10: Missing required column raises error ────────────────────────────
def test_missing_column_raises_error():
    """If CustomerId is missing, pipeline should raise ValueError."""
    fe = FeatureEngineer()
    bad_df = pd.DataFrame({
        "TransactionId": ["T1"],
        "Amount": [100],
        "Value": [100],
    })
    with pytest.raises(ValueError):
        fe.fit_transform(bad_df)