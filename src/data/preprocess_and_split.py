import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Configuration
RAW_DATA_PATH = r'c:\Workspaces\SKN22-2nd-4Team\data\01_raw'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
TARGET_COL = 'churn'
RANDOM_STATE = 42

def load_data():
    """Load and merge train and test datasets."""
    train_path = os.path.join(RAW_DATA_PATH, TRAIN_FILE)
    test_path = os.path.join(RAW_DATA_PATH, TEST_FILE)
    
    print(f"Loading data from {train_path} and {test_path}...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Check if test set has target
    if TARGET_COL not in df_test.columns:
        print(f"WARNING: '{TARGET_COL}' column missing in {TEST_FILE}. "
              "These rows cannot be used for stratified splitting or evaluation.")
        # We can't use unlabeled data for supervised training/splitting
        # But to proceed with 'Merge', we fill with NaN
        df_test[TARGET_COL] = np.nan
    
    # Add source flag for tracking (optional)
    df_train['dataset_source'] = 'train'
    df_test['dataset_source'] = 'test'

    # Merge for consistent preprocessing
    df_total = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    print(f"Total dataset shape after merge: {df_total.shape}")
    return df_total

def preprocess_data(df):
    """
    1. Binary Mapping (yes/no -> 1/0)
    2. Label Encoding (state, area_code)
    """
    df_processed = df.copy()
    
    # Binary Mapping
    # Note: 'churn' might be NaN for test rows
    binary_cols = ['international_plan', 'voice_mail_plan']
    for col in binary_cols:
        if col in df_processed.columns:
            # fillna first if necessary, but these should be clean
            df_processed[col] = df_processed[col].map({'yes': 1, 'no': 0})
    
    # Map Target separately to preserve NaNs if any
    if TARGET_COL in df_processed.columns:
        # Only map 'yes'/'no'. Leaves NaNs as NaN.
        df_processed[TARGET_COL] = df_processed[TARGET_COL].map({'yes': 1, 'no': 0})

    # Label Encoding for categorical features
    le = LabelEncoder()
    # State
    if 'state' in df_processed.columns:
        df_processed['state'] = le.fit_transform(df_processed['state'].astype(str))
        
    # Area Code
    if 'area_code' in df_processed.columns:
        df_processed['area_code'] = le.fit_transform(df_processed['area_code'].astype(str))
        
    return df_processed

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove rows containing outliers based on IQR.
    Note: Usage of this function will reduce dataset size.
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # Filter
        df_clean = df_clean[
            (df_clean[col] >= lower_bound) & 
            (df_clean[col] <= upper_bound)
        ]
        
    final_rows = len(df_clean)
    print(f"Outlier Removal: Removed {initial_rows - final_rows} rows.")
    return df_clean

def main():
    # 1. Load
    df = load_data()
    
    # 2. Preprocess
    df_processed = preprocess_data(df)
    
    # 3. Outlier Removal (Disabled)
    APPLY_OUTLIER_REMOVAL = False
    if APPLY_OUTLIER_REMOVAL:
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['churn', 'international_plan', 'voice_mail_plan', 'state', 'area_code', 'dataset_source', 'id']
        target_cols = [c for c in numeric_cols if c not in exclude_cols]
        df_processed = remove_outliers_iqr(df_processed, target_cols)
    else:
        print("Skipping outlier removal.")

    # 4. Prepare for Split
    # We must DROP rows where Target is NaN for stratified split
    initial_len = len(df_processed)
    df_clean_for_split = df_processed.dropna(subset=[TARGET_COL])
    dropped_rows = initial_len - len(df_clean_for_split)
    
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with missing target (likely from test.csv).")
        print("Note: Stratified split requires known labels.")

    # Drop helper columns
    drop_cols = [TARGET_COL, 'dataset_source']
    if 'id' in df_clean_for_split.columns:
        drop_cols.append('id')
        
    y = df_clean_for_split[TARGET_COL]
    X = df_clean_for_split.drop(columns=drop_cols)
    
    # Determine split size
    # User requested: Train 4250, Test 750. (Total 5000)
    # Available data might be less if test.csv was unlabeled (only 4250 total).
    
    total_samples = len(X)
    print(f"Available labeled samples for split: {total_samples}")
    
    if total_samples == 5000:
        test_size_count = 750
    elif total_samples == 4250:
        # If we only have train.csv data, we need to split it 85/15 to simulate the ratio
        # 4250 * 0.15 = 637.5 -> 638
        test_size_count = 0.15
        print("Using 0.15 test split ratio on available 4250 samples.")
    else:
        test_size_count = 0.15 # Fallback
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size_count, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    
    # 5. Output Verification
    print("-" * 30)
    print("Final Shapes Verification:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test:  {y_test.shape}")
    print("-" * 30)
    
    if X_train.shape[0] == 4250 and X_test.shape[0] == 750:
        print("SUCCESS: Data split shapes match the experiment specifications (4250, 750).")
    else:
        print(f"WARNING: Data split shapes do NOT exact match (4250, 750). Received ({X_train.shape[0]}, {X_test.shape[0]}).")
        print("Reason: Provided test.csv likely lacked labels, reducing total usable dataset to 4250.")

    # 6. Save Split Data
    OUTPUT_DIR = r'c:\Workspaces\SKN22-2nd-4Team\data\03_resampled'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nSaving split data to {OUTPUT_DIR}...")
    X_train.to_csv(os.path.join(OUTPUT_DIR, 'X_train_original.csv'), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, 'y_train_original.csv'), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, 'y_test.csv'), index=False)
    print("Files saved: X_train_original.csv, X_test.csv, y_train_original.csv, y_test.csv")



    print("Preprocessing & Splitting Completed.")
if __name__ == "__main__":
    main()