# tests/test_ckd.py
# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for the CKD Prediction System
# Run:  python -m pytest tests/ -v
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import gc
import math
import time
import sqlite3
import tempfile
import unittest
import numpy as np
import pandas as pd

# ── Add project root to path ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from auth import (
    init_db,
    register_user,
    login_user,
    validate_email,
    save_prediction,
    get_predictions,
    hash_password,
    create_patient,
    get_patients,
    update_patient,
    delete_patient,
    schedule_appointment,
    get_appointments,
    delete_appointment,
    submit_feedback,
    get_all_feedback,
    log_audit_action,
    get_audit_logs,
    get_admin_metrics
)
from ckd_utils.report_generator import safe_text, generate_detailed_recommendations
from ckd_utils.prediction import load_model_bundle, predict_single, explain_prediction_local
from ckd_utils.preprocessing import preprocess_single_input


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(BASE_DIR, "model", "kidney_model.pkl")
MODEL_AVAILABLE = os.path.exists(MODEL_PATH)


def _make_temp_db():
    """Return a path to a fresh temporary SQLite database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


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
# 1. Authentication Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestAuth(unittest.TestCase):

    def setUp(self):
        """Use a temp DB so tests don't pollute the real users.db."""
        import auth as auth_module
        self.orig_db = auth_module.DB_FILE
        self.temp_db = _make_temp_db()
        auth_module.DB_FILE = self.temp_db
        auth_module.init_db()

    def tearDown(self):
        import auth as auth_module
        auth_module.DB_FILE = self.orig_db
        # On Windows the SQLite file may still be held; force GC then try to remove
        gc.collect()
        try:
            os.unlink(self.temp_db)
        except PermissionError:
            pass  # Windows: file will be cleaned up by OS on process exit

    # ── validate_email ────────────────────────────────────────────────────────
    def test_validate_email_valid(self):
        self.assertTrue(validate_email("user@example.com"))

    def test_validate_email_valid_subdomain(self):
        self.assertTrue(validate_email("user@mail.example.co.uk"))

    def test_validate_email_no_at(self):
        self.assertFalse(validate_email("userexample.com"))

    def test_validate_email_no_domain(self):
        self.assertFalse(validate_email("user@"))

    def test_validate_email_empty(self):
        self.assertFalse(validate_email(""))

    # ── register_user ─────────────────────────────────────────────────────────
    def test_register_user_success(self):
        ok, msg = register_user("alice", "alice@test.com", "Secret1234")
        self.assertTrue(ok)
        self.assertIn("successful", msg.lower())

    def test_register_duplicate_username(self):
        register_user("bob", "bob@test.com", "Pass1234!")
        ok, msg = register_user("bob", "bob2@test.com", "Pass1234!")
        self.assertFalse(ok)
        self.assertIn("username", msg.lower())

    def test_register_duplicate_email(self):
        register_user("carol", "carol@test.com", "Pass1234!")
        ok, msg = register_user("carol2", "carol@test.com", "Pass1234!")
        self.assertFalse(ok)
        self.assertIn("email", msg.lower())

    # ── login_user ────────────────────────────────────────────────────────────
    def test_login_correct_credentials(self):
        register_user("dave", "dave@test.com", "MyPass999!")
        self.assertTrue(login_user("dave", "MyPass999!"))

    def test_login_wrong_password(self):
        register_user("eve", "eve@test.com", "Correct123!")
        self.assertFalse(login_user("eve", "WrongPass!"))

    def test_login_unknown_user(self):
        self.assertFalse(login_user("ghost", "anything"))

    # ── hash_password ─────────────────────────────────────────────────────────
    def test_password_is_hashed(self):
        h = hash_password("mypassword")
        self.assertNotEqual(h, "mypassword")
        self.assertEqual(len(h), 64)  # SHA-256 hex digest length

    def test_password_hash_deterministic(self):
        h1 = hash_password("same_password")
        h2 = hash_password("same_password")
        self.assertEqual(h1, h2)


# ═════════════════════════════════════════════════════════════════════════════
# 2. BMI Computation Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestBMI(unittest.TestCase):

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
# 3. Report Generator Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestReportGenerator(unittest.TestCase):

    # ── safe_text ─────────────────────────────────────────────────────────────
    def test_safe_text_en_dash(self):
        result = safe_text("Stage 1–2")
        self.assertNotIn("–", result)
        self.assertIn("-", result)

    def test_safe_text_em_dash(self):
        result = safe_text("hello—world")
        self.assertNotIn("—", result)

    def test_safe_text_superscript(self):
        result = safe_text("m²")
        self.assertNotIn("²", result)

    def test_safe_text_plain_ascii(self):
        result = safe_text("Hello World 123")
        self.assertEqual(result, "Hello World 123")

    # ── generate_detailed_recommendations ────────────────────────────────────
    def test_recommendations_healthy(self):
        recs = generate_detailed_recommendations(_make_sample_input(), "Healthy Kidney")
        titles = [t for t, _ in recs]
        self.assertIn("Overall Assessment", titles)
        self.assertIn("Dietary Suggestions", titles)

    def test_recommendations_stage3(self):
        recs = generate_detailed_recommendations(_make_sample_input(), "Moderate CKD (Stage 3)")
        text_all = " ".join(d for _, d in recs)
        self.assertIn("nephrologist", text_all.lower())

    def test_recommendations_stage5(self):
        recs = generate_detailed_recommendations(_make_sample_input(), "Kidney Failure (Stage 5)")
        text_all = " ".join(d for _, d in recs)
        self.assertIn("dialysis", text_all.lower())

    def test_recommendations_diabetes_included(self):
        d = dict(_make_sample_input())
        d["Diabetes"] = "Yes"
        recs = generate_detailed_recommendations(d, "Mild CKD (Stage 1-2)")
        titles = [t for t, _ in recs]
        self.assertIn("Glycemic Control", titles)

    def test_recommendations_proteinuria_included(self):
        d = dict(_make_sample_input())
        d["Urine_Albumin"] = 40   # >= 30 triggers recommendation
        recs = generate_detailed_recommendations(d, "Healthy Kidney")
        titles = [t for t, _ in recs]
        self.assertIn("Proteinuria Management", titles)

    def test_recommendations_returns_list_of_tuples(self):
        recs = generate_detailed_recommendations(_make_sample_input(), "Healthy Kidney")
        self.assertIsInstance(recs, list)
        for item in recs:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Prediction History (Database) Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestPredictionHistory(unittest.TestCase):

    def setUp(self):
        import auth as auth_module
        self.orig_db = auth_module.DB_FILE
        self.temp_db = _make_temp_db()
        auth_module.DB_FILE = self.temp_db
        auth_module.init_db()

    def tearDown(self):
        import auth as auth_module
        auth_module.DB_FILE = self.orig_db
        gc.collect()
        try:
            os.unlink(self.temp_db)
        except PermissionError:
            pass  # Windows file lock fallback

    def _save_one(self, patient_name="John Doe", label="Healthy Kidney", conf=98.5):
        features_dict = _make_sample_input()
        save_prediction(
            username        = "testuser",
            patient_name    = patient_name,
            prediction_label= label,
            confidence      = conf,
            features_dict   = features_dict,
        )

    def test_save_and_retrieve(self):
        self._save_one()
        rows = get_predictions("testuser")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["patient_name"], "John Doe")

    def test_retrieve_empty_for_other_user(self):
        self._save_one()
        rows = get_predictions("otheruser")
        self.assertEqual(len(rows), 0)

    def test_search_by_patient_name(self):
        self._save_one("Alice Smith")
        self._save_one("Bob Jones")
        rows = get_predictions("testuser", search_query="Alice")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["patient_name"], "Alice Smith")

    def test_search_case_insensitive_partial(self):
        self._save_one("Maria Garcia")
        rows = get_predictions("testuser", search_query="gar")
        self.assertEqual(len(rows), 1)

    def test_multiple_predictions_ordered_desc(self):
        self._save_one("First")
        time.sleep(1.1)  # Ensure distinct timestamp second
        self._save_one("Second")
        rows = get_predictions("testuser")
        self.assertEqual(len(rows), 2)
        # Most recent first
        self.assertEqual(rows[0]["patient_name"], "Second")

    def test_filter_by_date(self):
        import datetime
        self._save_one("Future")
        rows = get_predictions("testuser",
                               start_date="2020-01-01",
                               end_date="2099-12-31")
        self.assertEqual(len(rows), 1)

    def test_confidence_stored_correctly(self):
        self._save_one(conf=75.25)
        rows = get_predictions("testuser")
        self.assertAlmostEqual(rows[0]["confidence"], 75.25, places=1)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Prediction & Explainability Tests (requires model file)
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(MODEL_AVAILABLE, "Model file not found – run train_model.py first")
class TestPrediction(unittest.TestCase):

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
        # Should have exactly 35 features
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

    # ── explain_prediction_local ──────────────────────────────────────────────
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

    # ── Batch prediction ──────────────────────────────────────────────────────
    def test_predict_same_input_twice_is_deterministic(self):
        d = _make_sample_input()
        input_df1 = self._preprocess(d)
        input_df2 = self._preprocess(d)
        label1, proba1 = predict_single(input_df1, self.bundle)
        label2, proba2 = predict_single(input_df2, self.bundle)
        self.assertEqual(label1, label2)
        if proba1 is not None and proba2 is not None:
            np.testing.assert_array_almost_equal(proba1, proba2)

# ═════════════════════════════════════════════════════════════════════════════
# 6. CLINICAL MANAGEMENT (Patients, Appointments, Audits, Feedback)
# ═════════════════════════════════════════════════════════════════════════════
class TestClinicalManagement(unittest.TestCase):
    def setUp(self):
        import auth as auth_module
        self.orig_db = auth_module.DB_FILE
        self.temp_db = _make_temp_db()
        auth_module.DB_FILE = self.temp_db
        auth_module.init_db()
        self.username = "testdoc"
        register_user(self.username, "doc@example.com", "pass123")

    def tearDown(self):
        import auth as auth_module
        auth_module.DB_FILE = self.orig_db
        gc.collect()
        try:
            os.unlink(self.temp_db)
        except PermissionError:
            pass

    def test_patient_crud(self):
        # Create
        pid = create_patient(self.username, "Alice Smith", 55, "Female", 165.0, 65.0, "No", "Yes", "No", "No")
        self.assertTrue(pid > 0)
        
        # Read
        patients = get_patients(self.username)
        self.assertEqual(len(patients), 1)
        self.assertEqual(patients[0]["name"], "Alice Smith")
        self.assertEqual(patients[0]["age"], 55)

        # Update
        update_patient(self.username, pid, "Alice Doe", 56, "Female", 165.0, 64.0, "Yes", "Yes", "No", "No")
        patients = get_patients(self.username)
        self.assertEqual(patients[0]["name"], "Alice Doe")
        self.assertEqual(patients[0]["age"], 56)
        self.assertEqual(patients[0]["diabetes"], "Yes")

        # Delete
        delete_patient(self.username, pid)
        patients = get_patients(self.username)
        self.assertEqual(len(patients), 0)

    def test_appointments(self):
        aid = schedule_appointment(self.username, "Bob Lee", "2024-05-10 14:00:00", "Routine checkup")
        self.assertTrue(aid > 0)
        
        appts = get_appointments(self.username)
        self.assertEqual(len(appts), 1)
        self.assertEqual(appts[0]["patient_name"], "Bob Lee")
        self.assertEqual(appts[0]["notes"], "Routine checkup")
        
        delete_appointment(self.username, aid)
        appts = get_appointments(self.username)
        self.assertEqual(len(appts), 0)

    def test_feedback(self):
        submit_feedback(self.username, 5, "Perfect prediction", 1)
        fb = get_all_feedback()
        self.assertEqual(len(fb), 1)
        self.assertEqual(fb[0]["rating"], 5)
        self.assertEqual(fb[0]["comments"], "Perfect prediction")

    def test_audit_logs(self):
        log_audit_action(self.username, "Custom Test Action", "Test detail")
        logs = get_audit_logs()
        
        # There should be logs for registration and the custom action
        self.assertTrue(len(logs) >= 2)
        
        custom_log = [l for l in logs if l["action"] == "Custom Test Action"][0]
        self.assertEqual(custom_log["username"], self.username)
        self.assertEqual(custom_log["details"], "Test detail")

    def test_admin_metrics(self):
        metrics = get_admin_metrics()
        self.assertEqual(metrics["total_users"], 1)  # the one created in setUp
        self.assertEqual(metrics["total_predictions"], 0)
        self.assertEqual(metrics["users_list"][0]["username"], self.username)

# ═════════════════════════════════════════════════════════════════════════════
# SHAP Integration Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestSHAPIntegration(unittest.TestCase):
    def setUp(self):
        self.bundle = load_model_bundle()
        # Create a sample input
        raw_input = _make_sample_input()
        self.input_df = preprocess_single_input(raw_input, self.bundle["encoders"], self.bundle["scaler"])

    def test_explain_prediction_shap_succeeds(self):
        from ckd_utils.shap_explainer import explain_prediction_shap
        
        # Test with predicted class
        pred_label, proba = predict_single(self.input_df, self.bundle)
        pred_idx = list(self.bundle["classes"]).index(pred_label)
        
        # Call explanation
        res = explain_prediction_shap(self.input_df, self.bundle, pred_idx)
        
        # Asserts
        self.assertIsNotNone(res)
        self.assertIn("top_10", res)
        self.assertIn("bar_plot_path", res)
        self.assertIn("waterfall_plot_path", res)
        
        # Check files exist
        self.assertTrue(os.path.exists(res["bar_plot_path"]))
        self.assertTrue(os.path.exists(res["waterfall_plot_path"]))
        
        # Check top_10 structure
        self.assertEqual(len(res["top_10"]), 10)
        self.assertIn("feature", res["top_10"][0])
        self.assertIn("shap_value", res["top_10"][0])
        self.assertIn("raw_value", res["top_10"][0])

    def test_predictions_remain_identical(self):
        # Predict once
        label_before, proba_before = predict_single(self.input_df, self.bundle)
        
        # Run SHAP
        from ckd_utils.shap_explainer import explain_prediction_shap
        pred_idx = list(self.bundle["classes"]).index(label_before)
        explain_prediction_shap(self.input_df, self.bundle, pred_idx)
        
        # Predict again
        label_after, proba_after = predict_single(self.input_df, self.bundle)
        
        # Check equality
        self.assertEqual(label_before, label_after)
        if proba_before is not None:
            np.testing.assert_array_almost_equal(proba_before, proba_after)


# ─────────────────────────────────────────────────────────────────────────────

# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
