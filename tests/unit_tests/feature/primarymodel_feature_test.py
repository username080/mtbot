import pandas as pd
import numpy as np
import os
import sys
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from shared.features import PRIMARY_FEATURES

# 1. ARRANGE: Load the dataset once at the top so all tests can use it
@pytest.fixture(params=["BTCUSDT", "ETHUSDT"])
def df(request):
    symbol = request.param
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../../backtest/data/training_{symbol}.parquet"))
    if not os.path.exists(data_path):
        pytest.skip(f"Dataset for {symbol} not found at {data_path}")
    return pd.read_parquet(data_path)

features = PRIMARY_FEATURES

def test_dataset_loaded_successfully(df):
    """Verify the data actually loaded and has rows."""
    # ASSERT
    assert df is not None, "Dataframe failed to load"
    assert len(df) > 0, "Dataset is completely empty!"
    print("Data uploaded")


def test_all_features_data_quality(df):
    """Loops through all features and ensures they are mathematically valid."""
    for col in features:
        # 1. Check for Nulls (NaN)
        null_count = df[col].isnull().sum()
        assert null_count == 0, f"FAILED: {col} contains {null_count} NaN values!"
        
        # 2. Check for Infinity
        has_inf = np.isinf(df[col]).any()
        assert not has_inf, f"FAILED: {col} contains Infinity!"
        
        # 3. Check if the feature is completely dead (std == 0)
        # Note: We skip 'symbol_id' since it might be perfectly 0.0 for BTC
        if col != 'symbol_id':
            std_dev = df[col].std()
            assert std_dev > 0, f"FAILED: {col} is frozen (Standard Deviation is 0)!"
        print(f"{col} passed the test")



if __name__ == "__main__":
    datas = ["BTCUSDT","ETHUSDT"]
    for x in datas:
        df = None
        DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../../backtest/data/training_{x}.parquet"))
        df = pd.read_parquet(DATA_PATH)
        test_dataset_loaded_successfully(df)
        test_all_features_data_quality(df)