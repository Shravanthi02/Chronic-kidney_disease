# tests/test_team_leader.py
# ─────────────────────────────────────────────────────────────────────────────
# TDD — Team Leader Responsibilities
#   • ML Model Training / Validation
#   • Data Preprocessing  (BMI formula)
#   • CKD Prediction      (predict_single, load_model_bundle)
#   • Explainable AI      (explain_prediction_local)
#
# Run:  python -m pytest tests/test_team_leader.py -v
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import unittest
import numpy as np
import pandas as pd

# ── Add project root to path ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ckd_utils.prediction import load_model_bundle, predict_single, explain_prediction_local
from ckd_utils.preprocessing import preprocess_single_input

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(BASE_DIR, "model", "kidney_model.pkl")
MODEL_AVAILABLE = os.path.exists(MODEL_PATH)


def _make_sample_input():
    return {
        "Age": 45,
        "Gender": 1,
        "BMI": 26.5,
        "Systolic_BP": 120,
        "Diastolic_BP": 80,
        "Heart_Rate": 75,
        "Serum_Creatinine": 1.0,
        "Blood_Urea_Nitrogen": 15,
        "eGFR": 90,
        "Urine_Specific_Gravity": 1.015,
        "Urine_Albumin": 15,
        "Urine_Protein": 10,
        "Albumin_Creatinine_Ratio": 25,
        "Sodium": 140,
        "Potassium": 4.0,
        "Calcium": 9.5,
        "Phosphorus": 3.5,
        "Chloride": 100,
        "Bicarbonate": 24,
        "Hemoglobin": 14.0,
        "RBC_Count": 4.5,
        "WBC_Count": 7000,
        "Platelet_Count": 250000,
        "Packed_Cell_Volume": 40,
        "Blood_Glucose_Random": 100,
        "Fasting_Glucose": 90,
        "HbA1c": 5.5,
        "Cholesterol": 180,
        "Triglycerides": 120,
        "Serum_Albumin": 4.0,
        "Total_Protein": 7.0,
        "Diabetes": "No",
        "Hypertension": "No",
        "Smoking_Status": "No",
        "Family_History_Kidney": "No",
    }


# ═════════════════════════════════════════════════════════════════════════════
# 1. Data Preprocessing — BMI Computation
# ═════════════════════════════════════════════════════════════════════════════

class TestBMI(unittest.TestCase):
    """Validates the BMI formula used during feature engineering."""

    def _bmi(self, weight_kg, height_cm):
        return weight_kg / ((height_cm / 100) ** 2)

    def test_bmi_normal(self):
        bmi = self._bmi(70, 175)
        self.assertAlmostEqual(bmi, 22.86, places=1)

    def test_bmi_obese(self):
        bmi = self._bmi(100, 170)
        self.assertGreater(bmi, 30.0)

    def test_bmi_underweight(self):
        bmi = self._bmi(45, 170)
        self.assertLess(bmi, 18.5)

    def test_bmi_overweight(self):
        bmi = self._bmi(80, 170)
        self.assertGreater(bmi, 25.0)
        self.assertLess(bmi, 30.0)

    def test_bmi_formula_correctness(self):
        bmi = self._bmi(60, 160)
        expected = 60 / (1.6 ** 2)
        self.assertAlmostEqual(bmi, expected, places=5)


# ═════════════════════════════════════════════════════════════════════════════
# 2. CKD Prediction & Explainable AI (requires trained model)
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(MODEL_AVAILABLE, "Model file not found – run train_model.py first")
class TestPrediction(unittest.TestCase):
    """Tests for model loading, single prediction, preprocessing, and XAI."""

    @classmethod
    def setUpClass(cls):
        cls.bundle = load_model_bundle(MODEL_PATH)

    def _preprocess(self, d=None):
        if d is None:
            d = _make_sample_input()
        return preprocess_single_input(d, self.bundle["encoders"], self.bundle["scaler"])

    # ── load_model_bundle ─────────────────────────────────────────────────────
    def test_bundle_has_required_keys(self):
        for key in ("model", "encoders", "scaler", "classes", "features"):
            self.assertIn(key, self.bundle)

    def test_classes_is_list(self):
        self.assertIsInstance(self.bundle["classes"], (list, np.ndarray))

    def test_features_count(self):
        """Model bundle must expose exactly 35 features."""
        self.assertEqual(len(self.bundle["features"]), 35)

    # ── predict_single ────────────────────────────────────────────────────────
    def test_predict_single_returns_label_and_proba(self):
        input_df = self._preprocess()
        label, proba = predict_single(input_df, self.bundle)
        self.assertIsInstance(label, str)
        self.assertIn(label, list(self.bundle["classes"]))

    def test_predict_single_proba_sums_to_one(self):
        input_df = self._preprocess()
        _, proba = predict_single(input_df, self.bundle)
        if proba is not None:
            self.assertAlmostEqual(float(proba.sum()), 1.0, places=5)

    def test_predict_single_proba_shape(self):
        input_df = self._preprocess()
        _, proba = predict_single(input_df, self.bundle)
        if proba is not None:
            self.assertEqual(len(proba), len(self.bundle["classes"]))

    def test_predict_single_proba_values_in_range(self):
        input_df = self._preprocess()
        _, proba = predict_single(input_df, self.bundle)
        if proba is not None:
            self.assertTrue(all(0.0 <= p <= 1.0 for p in proba))

    # ── preprocess_single_input ───────────────────────────────────────────────
    def test_preprocess_output_is_dataframe(self):
        input_df = self._preprocess()
        self.assertIsInstance(input_df, pd.DataFrame)

    def test_preprocess_output_shape(self):
        input_df = self._preprocess()
        self.assertEqual(input_df.shape[0], 1)
        self.assertEqual(input_df.shape[1], len(self.bundle["features"]))

    def test_preprocess_column_names_match_features(self):
        input_df = self._preprocess()
        self.assertListEqual(list(input_df.columns), list(self.bundle["features"]))

    # ── explain_prediction_local (Explainable AI) ─────────────────────────────
    def test_explain_returns_list(self):
        input_df = self._preprocess()
        result = explain_prediction_local(input_df, self.bundle)
        self.assertIsInstance(result, list)

    def test_explain_length_equals_features(self):
        input_df = self._preprocess()
        result = explain_prediction_local(input_df, self.bundle)
        self.assertEqual(len(result), len(self.bundle["features"]))

    def test_explain_sorted_by_abs_contribution(self):
        input_df = self._preprocess()
        result = explain_prediction_local(input_df, self.bundle)
        vals = [abs(v) for _, v in result]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_explain_each_item_is_tuple(self):
        input_df = self._preprocess()
        result = explain_prediction_local(input_df, self.bundle)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    def test_explain_feature_names_are_strings(self):
        input_df = self._preprocess()
        result = explain_prediction_local(input_df, self.bundle)
        for feat, _ in result:
            self.assertIsInstance(feat, str)

    def test_explain_contribution_values_are_floats(self):
        input_df = self._preprocess()
        result = explain_prediction_local(input_df, self.bundle)
        for _, contrib in result:
            self.assertIsInstance(contrib, float)

    # ── Model Validation — determinism ────────────────────────────────────────
    def test_predict_same_input_twice_is_deterministic(self):
        """Same input must always produce the same prediction (model validation)."""
        d = _make_sample_input()
        input_df1 = self._preprocess(d)
        input_df2 = self._preprocess(d)
        label1, proba1 = predict_single(input_df1, self.bundle)
        label2, proba2 = predict_single(input_df2, self.bundle)
        self.assertEqual(label1, label2)
        if proba1 is not None and proba2 is not None:
            np.testing.assert_array_almost_equal(proba1, proba2)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
