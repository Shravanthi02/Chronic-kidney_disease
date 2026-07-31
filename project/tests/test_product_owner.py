# tests/test_product_owner.py
# ─────────────────────────────────────────────────────────────────────────────
# TDD — Product Owner Responsibilities
#   • User Registration
#   • Login
#   • Email Validation
#   • Password Hashing
#   • Prediction History
#   • User Acceptance Test Cases
#
# Run:  python -m pytest tests/test_product_owner.py -v
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import gc
import time
import tempfile
import unittest

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
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

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
# 1. Authentication — User Registration, Login, Email Validation,
#    Password Hashing (User Acceptance Tests)
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
        gc.collect()
        try:
            os.unlink(self.temp_db)
        except PermissionError:
            pass  # Windows: file will be cleaned up by OS on process exit

    # ── Email Validation ──────────────────────────────────────────────────────
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

    # ── User Registration ─────────────────────────────────────────────────────
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

    # ── Login ─────────────────────────────────────────────────────────────────
    def test_login_correct_credentials(self):
        register_user("dave", "dave@test.com", "MyPass999!")
        self.assertTrue(login_user("dave", "MyPass999!"))

    def test_login_wrong_password(self):
        register_user("eve", "eve@test.com", "Correct123!")
        self.assertFalse(login_user("eve", "WrongPass!"))

    def test_login_unknown_user(self):
        self.assertFalse(login_user("ghost", "anything"))

    # ── Password Hashing ──────────────────────────────────────────────────────
    def test_password_is_hashed(self):
        h = hash_password("mypassword")
        self.assertNotEqual(h, "mypassword")
        self.assertEqual(len(h), 64)  # SHA-256 hex digest length

    def test_password_hash_deterministic(self):
        h1 = hash_password("same_password")
        h2 = hash_password("same_password")
        self.assertEqual(h1, h2)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Prediction History (User Acceptance Tests)
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
        save_prediction(
            username         = "testuser",
            patient_name     = patient_name,
            prediction_label = label,
            confidence       = conf,
            features_dict    = _make_sample_input(),
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
        self._save_one("Future")
        rows = get_predictions("testuser",
                               start_date="2020-01-01",
                               end_date="2099-12-31")
        self.assertEqual(len(rows), 1)

    def test_confidence_stored_correctly(self):
        self._save_one(conf=75.25)
        rows = get_predictions("testuser")
        self.assertAlmostEqual(rows[0]["confidence"], 75.25, places=1)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
