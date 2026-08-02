import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from shared.features import META_FEATURES

# 1. ARRANGE: Load the dataset once at the top so all tests can use it
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../backtest/data/training_meta_combined.parquet"))

# Load dataset if it exists, otherwise tests will fail on load
df = None
if os.path.exists(DATA_PATH):
    df = pd.read_parquet(DATA_PATH)

features = META_FEATURES

def test_dataset_loaded_successfully():
    """Verify the data actually loaded and has rows."""
    # ASSERT
    assert df is not None, "Dataframe failed to load (file might not exist)"
    assert len(df) > 0, "Dataset is completely empty!"
    print("Data uploaded")


def test_all_features_data_quality():
    """Loops through all features and ensures they are mathematically valid."""
    assert df is not None, "Dataframe failed to load"
    
    for col in features:
        if col not in df.columns:
            print(f"Warning: {col} not found in dataframe.")
            continue
            
        # 1. Check for Nulls (NaN)
        null_count = df[col].isnull().sum()
        assert null_count == 0, f"FAILED: {col} contains {null_count} NaN values!"
        
        # 2. Check for Infinity
        has_inf = np.isinf(df[col]).any()
        assert not has_inf, f"FAILED: {col} contains Infinity!"
        
        # 3. Check if the feature is completely dead (std == 0)
        std_dev = df[col].std()
        assert std_dev > 0, f"FAILED: {col} is frozen (Standard Deviation is 0)!"
        
        print(f"{col} passed the test")


if __name__ == "__main__":
    test_dataset_loaded_successfully()
    test_all_features_data_quality()
