import os
import sys
import pandas as pd
import xgboost as xgb

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from shared.features import PRIMARY_FEATURES, META_FEATURES

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backtest/data"))

def train_and_get_importance(X, y, feature_names, title):
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")
    
    # Use standard hyperparameters for a quick robust fit
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Get feature importances (Gain is default for XGBClassifier)
    importances = model.feature_importances_
    
    # Create DataFrame and sort
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance (Gain)': importances
    }).sort_values('Importance (Gain)', ascending=False)
    
    # Calculate cumulative importance
    df_imp['Cumulative'] = df_imp['Importance (Gain)'].cumsum()
    
    # Print formatted table
    print(f"{'Feature':<30} | {'Importance':<12} | {'Cumulative':<12}")
    print("-" * 60)
    for _, row in df_imp.iterrows():
        imp_str = f"{row['Importance (Gain)']:.4f}"
        cum_str = f"{row['Cumulative']:.4f}"
        print(f"{row['Feature']:<30} | {imp_str:<12} | {cum_str:<12}")
        
    return df_imp

def main():
    print("Starting Feature Importance Analysis...")
    
    # 1. Primary Model Importance
    primary_path = os.path.join(DATA_DIR, "training_BTCUSDT.parquet")
    if os.path.exists(primary_path):
        print(f"\nLoading Primary Data: {primary_path}")
        df = pd.read_parquet(primary_path)
        
        # Ensure all PRIMARY_FEATURES exist in df
        available_features = [f for f in PRIMARY_FEATURES if f in df.columns]
        
        # Drop NaNs
        df_clean = df.dropna(subset=available_features + ["label_long", "label_short"])
        X_primary = df_clean[available_features]
        
        # Long Model Importance
        y_long = df_clean["label_long"]
        train_and_get_importance(X_primary, y_long, available_features, "PRIMARY MODEL (LONG TARGET) FEATURE IMPORTANCE")
        
        # Short Model Importance
        y_short = df_clean["label_short"]
        train_and_get_importance(X_primary, y_short, available_features, "PRIMARY MODEL (SHORT TARGET) FEATURE IMPORTANCE")
    else:
        print(f"Primary data not found at {primary_path}")
        
    # 2. Meta Model Importance
    meta_path = os.path.join(DATA_DIR, "training_meta_combined.parquet")
    if os.path.exists(meta_path):
        print(f"\nLoading Meta Data: {meta_path}")
        df_meta = pd.read_parquet(meta_path)
        
        available_meta = [f for f in META_FEATURES if f in df_meta.columns]
        df_meta_clean = df_meta.dropna(subset=available_meta + ["meta_label"])
        
        X_meta = df_meta_clean[available_meta]
        y_meta = df_meta_clean["meta_label"]
        
        train_and_get_importance(X_meta, y_meta, available_meta, "META MODEL FEATURE IMPORTANCE")
    else:
        print(f"\nMeta data not found at {meta_path}")

if __name__ == "__main__":
    main()
