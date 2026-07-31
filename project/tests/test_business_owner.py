# tests/test_business_owner.py
# ─────────────────────────────────────────────────────────────────────────────
# TDD — Business Owner Responsibilities
#   • PDF Report Generation
#   • Health Recommendations
#   • Admin Dashboard
#   • Audit Logs
#   • Business Rule Validation
#   • Functional Acceptance Tests
#
# Run:  python -m pytest tests/test_business_owner.py -v
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import gc
import tempfile
import unittest

# ── Add project root to path ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from auth import (
    init_db,
    register_user,
    log_audit_action,
    get_audit_logs,
    get_admin_metrics,
)
from ckd_utils.report_generator import safe_text, generate_detailed_recommendations, create_pdf_report

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
# Base DB mixin
# ═════════════════════════════════════════════════════════════════════════════

class _TempDbMixin:
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
            pass  # Windows file lock fallback


# ═════════════════════════════════════════════════════════════════════════════
# 1. PDF Report Generator — Business Rule Validation
# ═════════════════════════════════════════════════════════════════════════════

class TestReportGenerator(unittest.TestCase):
    """Tests for safe_text(), health recommendations, and PDF output."""

    # ── safe_text — character encoding business rules ─────────────────────────
    def test_safe_text_en_dash(self):
        """En-dash in stage labels must be normalised to hyphen for PDF output."""
        result = safe_text("Stage 1\u20132")
        self.assertNotIn("\u2013", result)
        self.assertIn("-", result)

    def test_safe_text_em_dash(self):
        result = safe_text("hello\u2014world")
        self.assertNotIn("\u2014", result)

    def test_safe_text_superscript(self):
        result = safe_text("m\u00b2")
        self.assertNotIn("\u00b2", result)

    def test_safe_text_plain_ascii(self):
        """Plain ASCII must pass through unchanged."""
        result = safe_text("Hello World 123")
        self.assertEqual(result, "Hello World 123")

    # ── Health Recommendations — functional acceptance tests ──────────────────
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
        d["Urine_Albumin"] = 40  # >= 30 triggers business rule
        recs = generate_detailed_recommendations(d, "Healthy Kidney")
        titles = [t for t, _ in recs]
        self.assertIn("Proteinuria Management", titles)

    def test_recommendations_returns_list_of_tuples(self):
        """Output contract: must be a list of 2-element tuples."""
        recs = generate_detailed_recommendations(_make_sample_input(), "Healthy Kidney")
        self.assertIsInstance(recs, list)
        for item in recs:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    # ── PDF Report Generation ─────────────────────────────────────────────────
    def test_pdf_report_returns_bytes(self):
        """PDF output must be a non-empty bytes object (binary PDF content)."""
        recs = generate_detailed_recommendations(_make_sample_input(), "Healthy Kidney")
        pdf_bytes = create_pdf_report(
            input_dict   = _make_sample_input(),
            label        = "Healthy Kidney",
            conf         = 97.5,
            recommendations = recs,
            patient_name = "Test Patient",
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 0)

    def test_pdf_report_starts_with_pdf_header(self):
        """Valid PDF files must begin with the %PDF magic bytes."""
        recs = generate_detailed_recommendations(_make_sample_input(), "Healthy Kidney")
        pdf_bytes = create_pdf_report(
            input_dict      = _make_sample_input(),
            label           = "Healthy Kidney",
            conf            = 97.5,
            recommendations = recs,
            patient_name    = "Test Patient",
        )
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_pdf_report_generated_for_all_stages(self):
        """PDF must be generatable for every CKD stage without error."""
        stages = [
            "Healthy Kidney",
            "Mild CKD (Stage 1-2)",
            "Moderate CKD (Stage 3)",
            "Severe CKD (Stage 4)",
            "Kidney Failure (Stage 5)",
        ]
        for stage in stages:
            recs = generate_detailed_recommendations(_make_sample_input(), stage)
            pdf_bytes = create_pdf_report(
                input_dict      = _make_sample_input(),
                label           = stage,
                conf            = 85.0,
                recommendations = recs,
                patient_name    = "Stage Patient",
            )
            self.assertGreater(len(pdf_bytes), 0, msg=f"PDF empty for stage: {stage}")


# ═════════════════════════════════════════════════════════════════════════════
# 2. Audit Logs — Business Rule Validation
# ═════════════════════════════════════════════════════════════════════════════

class TestAuditLogs(_TempDbMixin, unittest.TestCase):
    """Tests that audit log entries are correctly recorded and retrievable."""

    def test_audit_log_written_and_retrieved(self):
        log_audit_action(self.username, "Custom Test Action", "Test detail")
        logs = get_audit_logs()
        # Logs include the registration entry + our custom action
        self.assertTrue(len(logs) >= 2)

    def test_audit_log_correct_action_and_detail(self):
        log_audit_action(self.username, "Custom Test Action", "Test detail")
        logs = get_audit_logs()
        custom_log = [l for l in logs if l["action"] == "Custom Test Action"][0]
        self.assertEqual(custom_log["username"], self.username)
        self.assertEqual(custom_log["details"], "Test detail")

    def test_audit_log_multiple_entries(self):
        log_audit_action(self.username, "Action A", "Detail A")
        log_audit_action(self.username, "Action B", "Detail B")
        logs = get_audit_logs()
        actions = [l["action"] for l in logs]
        self.assertIn("Action A", actions)
        self.assertIn("Action B", actions)

    def test_audit_log_has_required_fields(self):
        """Each audit log entry must expose username, action, details fields."""
        log_audit_action(self.username, "Field Check", "Checking fields")
        logs = get_audit_logs()
        target = next(l for l in logs if l["action"] == "Field Check")
        for field in ("username", "action", "details"):
            self.assertIn(field, target)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Admin Dashboard — Functional Acceptance Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestAdminMetrics(_TempDbMixin, unittest.TestCase):
    """Tests for admin dashboard metrics — user counts, prediction counts."""

    def test_admin_metrics_has_required_keys(self):
        metrics = get_admin_metrics()
        for key in ("total_users", "total_predictions", "users_list"):
            self.assertIn(key, metrics)

    def test_admin_metrics_user_count(self):
        """setUp registers exactly one user, so total_users must be 1."""
        metrics = get_admin_metrics()
        self.assertEqual(metrics["total_users"], 1)

    def test_admin_metrics_initial_prediction_count(self):
        """No predictions have been saved, count must be 0."""
        metrics = get_admin_metrics()
        self.assertEqual(metrics["total_predictions"], 0)

    def test_admin_metrics_users_list_contains_registered_user(self):
        metrics = get_admin_metrics()
        usernames = [u["username"] for u in metrics["users_list"]]
        self.assertIn(self.username, usernames)

    def test_admin_metrics_users_list_is_list(self):
        metrics = get_admin_metrics()
        self.assertIsInstance(metrics["users_list"], list)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
