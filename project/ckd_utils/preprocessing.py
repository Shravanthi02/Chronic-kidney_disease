# utils/preprocessing.py
# ─────────────────────────────────────────────────────────────────
# Handles all data loading, cleaning, encoding, and splitting tasks.
# ─────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# ---------------------------------------------------------------------------
# Feature & label constants
# ---------------------------------------------------------------------------
TARGET_COL = "Target"

# The 35 features used by the patient-friendly prediction model.
# Height and Weight are NOT in the dataset; BMI is computed in the UI and
# passed directly. This list must match the column names in the CSV exactly.
SELECTED_FEATURES = [
    "Age", "Gender", "BMI", "Systolic_BP", "Diastolic_BP", "Heart_Rate",
    "Serum_Creatinine", "Blood_Urea_Nitrogen", "eGFR", "Urine_Specific_Gravity",
    "Urine_Albumin", "Urine_Protein", "Albumin_Creatinine_Ratio",
    "Sodium", "Potassium", "Calcium", "Phosphorus", "Chloride", "Bicarbonate",
    "Hemoglobin", "RBC_Count", "WBC_Count", "Platelet_Count", "Packed_Cell_Volume",
    "Blood_Glucose_Random", "Fasting_Glucose", "HbA1c", "Cholesterol",
    "Triglycerides", "Serum_Albumin", "Total_Protein", "Diabetes",
    "Hypertension", "Smoking_Status", "Family_History_Kidney"
]

# Columns that are binary yes/no strings
BINARY_COLS = ["Diabetes", "Hypertension", "Family_History_Kidney", "Smoking_Status"]

# Columns that are numeric 0/1 in the dataset (Gender)
NUMERIC_CAT_COLS = ["Gender"]


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a Pandas DataFrame.

    Parameters
    ----------
    filepath : str  Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    return df


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Return a summary dict describing data quality:
    missing values, duplicates, and dtypes.
    """
    summary = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    return summary


def encode_features(df: pd.DataFrame, fit: bool = True, encoders: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical columns.

    - Binary yes/no columns → 1 / 0
    - Gender column (already 0/1 int) → kept as-is
    - Target column → LabelEncoded

    Parameters
    ----------
    df       : Input DataFrame (copy is made internally).
    fit      : If True, fit new encoders; if False, use provided `encoders`.
    encoders : Dict of pre-fitted encoders (used when fit=False).

    Returns
    -------
    (encoded_df, encoders_dict)
    """
    df = df.copy()

    if encoders is None:
        encoders = {}

    # ── Binary yes/no columns ──────────────────────────────────────────────
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(0).astype(int)

    # ── Target column ──────────────────────────────────────────────────────
    if TARGET_COL in df.columns:
        if fit:
            le = LabelEncoder()
            df[TARGET_COL] = le.fit_transform(df[TARGET_COL])
            encoders["target_encoder"] = le
        else:
            le = encoders.get("target_encoder")
            if le is not None:
                df[TARGET_COL] = le.transform(df[TARGET_COL])

    return df, encoders


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the DataFrame into feature matrix X and target series y.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame = None,
                   fit: bool = True,
                   scaler=None):
    """
    Standardise numeric features.

    Returns (X_train_scaled, X_test_scaled, scaler).
    If X_test is None, X_test_scaled will be None.
    """
    if fit:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    # Convert back to DataFrames to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    if X_test_scaled is not None:
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Full preprocessing pipeline used during model training.
    Only the columns listed in SELECTED_FEATURES (plus the target) are kept.

    Returns
    -------
    X_train, X_test, y_train, y_test, encoders, scaler, class_names
    """
    # ── Keep only selected features + target ──────────────────────────────
    keep_cols = SELECTED_FEATURES + [TARGET_COL]
    train_df = train_df[[c for c in keep_cols if c in train_df.columns]].copy()
    test_df  = test_df[[c for c in keep_cols if c in test_df.columns]].copy()

    # Drop duplicates
    train_df = train_df.drop_duplicates()
    test_df  = test_df.drop_duplicates()

    # Fill missing values (median for numeric, mode for categorical/string)
    for col in train_df.columns:
        # Treat object and string-backed Arrow dtypes as categorical
        if train_df[col].dtype == object or hasattr(train_df[col].dtype, 'pyarrow_dtype') \
                or str(train_df[col].dtype) in ('string', 'StringDtype'):
            mode_val = train_df[col].mode()[0] if not train_df[col].mode().empty else "No"
            train_df[col] = train_df[col].fillna(mode_val)
            test_df[col]  = test_df[col].fillna(mode_val)
        else:
            try:
                median_val = train_df[col].median()
                train_df[col] = train_df[col].fillna(median_val)
                test_df[col]  = test_df[col].fillna(median_val)
            except TypeError:
                # Fallback: non-numeric column — use mode
                mode_val = train_df[col].mode()[0] if not train_df[col].mode().empty else "No"
                train_df[col] = train_df[col].fillna(mode_val)
                test_df[col]  = test_df[col].fillna(mode_val)

    # Encode
    train_enc, encoders = encode_features(train_df, fit=True)
    test_enc,  _        = encode_features(test_df,  fit=False, encoders=encoders)

    # Split X / y
    X_train, y_train = split_features_target(train_enc)
    X_test,  y_test  = split_features_target(test_enc)

    # Scale
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    # Class names from original training data
    class_names = list(encoders["target_encoder"].classes_)

    return X_train_s, X_test_s, y_train, y_test, encoders, scaler, class_names


def preprocess_single_input(input_dict: dict, encoders: dict, scaler) -> pd.DataFrame:
    """
    Preprocess a single user input (from the Streamlit form) for prediction.

    Parameters
    ----------
    input_dict : dict   Keys = feature column names, values = raw user inputs.
                        Must contain exactly the keys in SELECTED_FEATURES.
    encoders   : dict   Fitted encoders from training.
    scaler              Fitted StandardScaler from training.

    Returns
    -------
    pd.DataFrame  (1 row, scaled)
    """
    row = pd.DataFrame([input_dict])

    # Ensure column order matches training
    row = row[SELECTED_FEATURES]

    # Encode binary columns
    for col in BINARY_COLS:
        if col in row.columns:
            row[col] = row[col].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(0).astype(int)

    # Scale
    row_scaled = scaler.transform(row)
    row_scaled = pd.DataFrame(row_scaled, columns=row.columns)

    return row_scaled
