import pandas as pd


def test_dataframe_not_empty():
    """Test that a sample dataframe is not empty"""
    df = pd.DataFrame({"amount": [100, 200, 300], "customer_id": [1, 2, 3]})
    assert len(df) > 0


def test_transaction_count_column():
    """Test that we can calculate transaction count per customer"""
    df = pd.DataFrame({"customer_id": [1, 1, 2], "amount": [100, 200, 300]})
    result = df.groupby("customer_id")["amount"].count().reset_index()
    result.columns = ["customer_id", "transaction_count"]
    assert "transaction_count" in result.columns
