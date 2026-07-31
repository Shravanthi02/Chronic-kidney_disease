# utils/prediction.py
# ─────────────────────────────────────────────────────────────────
# Handles model loading and single/batch prediction logic.
# ─────────────────────────────────────────────────────────────────

import joblib
import numpy as np
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "kidney_model.pkl")


def load_model_bundle(model_path: str = MODEL_PATH) -> dict:
    """
    Load the persisted model bundle from disk.

    The bundle is a dict containing:
        'model'     – trained sklearn estimator
        'encoders'  – dict of fitted encoders
        'scaler'    – fitted StandardScaler
        'classes'   – list of class label strings
        'features'  – ordered list of feature column names
        'metrics'   – dict of evaluation metrics on the test set

    Parameters
    ----------
    model_path : str  Path to the .pkl bundle.

    Returns
    -------
    dict
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Please run the training notebook (03_Model_Training.ipynb) first."
        )
    bundle = joblib.load(model_path)
    return bundle


def predict_single(input_df: pd.DataFrame, bundle: dict) -> tuple[str, np.ndarray]:
    """
    Predict the CKD stage for a single preprocessed input row.

    Parameters
    ----------
    input_df : pd.DataFrame  (1 × n_features, already scaled)
    bundle   : dict          Model bundle from load_model_bundle().

    Returns
    -------
    (predicted_class_label: str, probabilities: np.ndarray or None)
    """
    model   = bundle["model"]
    classes = bundle["classes"]

    pred_idx = model.predict(input_df)[0]
    label    = classes[pred_idx]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]

    return label, proba


def predict_batch(X: pd.DataFrame, bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Run batch prediction on a preprocessed feature matrix.

    Parameters
    ----------
    X      : pd.DataFrame  (n_samples × n_features, already scaled)
    bundle : dict          Model bundle.

    Returns
    -------
    (predictions_array, probabilities_array or None)
    """
    model = bundle["model"]
    preds = model.predict(X)

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

    return preds, proba


def get_class_label(idx: int, bundle: dict) -> str:
    """Return the human-readable class label for a numeric class index."""
    return bundle["classes"][idx]


def explain_prediction_local(input_df: pd.DataFrame, bundle: dict) -> list[tuple[str, float]]:
    """
    Explain the prediction locally by perturbing each feature.
    Calculates the impact of each feature on the predicted class probability.

    Parameters
    ----------
    input_df : pd.DataFrame  (1 row, preprocessed and scaled)
    bundle   : dict          Model bundle

    Returns
    -------
    list of (feature_name, contribution) sorted by absolute contribution descending.
    """
    model = bundle["model"]
    features = bundle["features"]

    # Get predicted class index
    pred_idx = model.predict(input_df)[0]
    
    if not hasattr(model, "predict_proba"):
        # Fallback if model doesn't support probabilities (e.g. SVM without probability=True)
        # return empty or uniform importance
        return [(f, 0.0) for f in features]
        
    base_probs = model.predict_proba(input_df)[0]
    base_prob = base_probs[pred_idx]

    explanations = []

    # Perturb each feature by setting it to its average (0 in standard scaled space)
    for col in features:
        perturbed_df = input_df.copy()
        # In scaled space, 0 is the mean of the training data
        perturbed_df[col] = 0.0

        perturbed_probs = model.predict_proba(perturbed_df)[0]
        perturbed_prob = perturbed_probs[pred_idx]

        # Contribution is how much the feature's actual value changed the probability
        # compared to its mean value.
        contribution = base_prob - perturbed_prob
        explanations.append((col, float(contribution)))

    # Sort by absolute contribution descending
    explanations.sort(key=lambda x: abs(x[1]), reverse=True)
    return explanations

