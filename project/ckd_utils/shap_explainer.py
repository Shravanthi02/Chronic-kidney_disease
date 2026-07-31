# ckd_utils/shap_explainer.py
# ─────────────────────────────────────────────────────────────────
# Handles SHAP explainability, plot generation and caching.
# ─────────────────────────────────────────────────────────────────

import os
import shap
import streamlit as st
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_background_data(bundle: dict, num_samples: int = 100) -> pd.DataFrame:
    """
    Safely loads and preprocesses background data from Training_CKD_dataset.csv.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_path = os.path.join(base_dir, "dataset", "Training_CKD_dataset.csv")
        if os.path.exists(train_path):
            df = pd.read_csv(train_path, nrows=num_samples * 2)
            from ckd_utils.preprocessing import encode_features, SELECTED_FEATURES
            keep_cols = SELECTED_FEATURES
            df = df[[c for c in keep_cols if c in df.columns]].copy()
            df = df.drop_duplicates().head(num_samples)
            
            # Impute missing values using mode for categoricals and median for numerics
            for col in df.columns:
                if df[col].dtype == object or str(df[col].dtype) in ('string', 'category'):
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else "No"
                    df[col] = df[col].fillna(mode_val)
                else:
                    try:
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val)
                    except Exception:
                        df[col] = df[col].fillna(0)
                        
            # Encode features using saved encoders from the bundle
            encoded_df, _ = encode_features(df, fit=False, encoders=bundle["encoders"])
            
            # Scale features
            X_scaled = bundle["scaler"].transform(encoded_df[SELECTED_FEATURES])
            return pd.DataFrame(X_scaled, columns=SELECTED_FEATURES)
    except Exception as e:
        logger.warning(f"Failed to load background data for SHAP explainer: {e}")
        
    # Fallback if CSV not found or preprocessing failed
    features = bundle.get("features", [])
    return pd.DataFrame(np.zeros((1, len(features))), columns=features)

@st.cache_resource
def get_cached_explainer(_model, model_name: str, _bundle: dict):
    """
    Initialise and cache the SHAP explainer depending on the trained model type.
    """
    if "Random Forest" in model_name or "Decision Tree" in model_name or hasattr(_model, "tree_"):
        return shap.TreeExplainer(_model)
    elif "Logistic Regression" in model_name or hasattr(_model, "coef_"):
        X_background = load_background_data(_bundle)
        return shap.LinearExplainer(_model, X_background)
    else:
        X_background = load_background_data(_bundle)
        return shap.Explainer(_model, X_background)

def explain_prediction_shap(input_df: pd.DataFrame, bundle: dict, pred_idx: int, feature_labels: dict = None) -> dict:
    """
    Generates SHAP explanations for a single prediction and outputs top-10 list and plot paths.
    """
    model = bundle["model"]
    model_name = bundle["model_name"]
    classes = bundle["classes"]
    
    # Retrieve/initialize the cached SHAP explainer
    explainer = get_cached_explainer(model, model_name, bundle)
    
    # Generate SHAP explanation object for the single prediction row
    explanation = explainer(input_df)
    
    # Extract prediction class slice from explanation object
    # For multiclass, shape is (1, n_features, n_classes)
    # For binary/single, shape is (1, n_features)
    if len(explanation.shape) == 3:
        explanation_slice = explanation[0, :, pred_idx]
    elif len(explanation.shape) == 2:
        explanation_slice = explanation[0]
    else:
        explanation_slice = explanation
        
    # Apply human-readable feature labels if provided
    if feature_labels is not None:
        explanation_slice.feature_names = [feature_labels.get(name, name) for name in explanation_slice.feature_names]
        
    # Extract and sort feature contributions
    shap_contribs = []
    for name, val, raw_val in zip(explanation_slice.feature_names, explanation_slice.values, explanation_slice.data):
        shap_contribs.append({
            "feature": name,
            "shap_value": float(val),
            "raw_value": float(raw_val)
        })
    shap_contribs.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    top_10 = shap_contribs[:10]
    
    return {
        "top_10": top_10,
        "explanation_slice": explanation_slice
    }
