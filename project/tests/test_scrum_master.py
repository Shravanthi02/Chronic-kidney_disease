# tests/test_scrum_master.py
# ─────────────────────────────────────────────────────────────────────────────
# TDD — Scrum Master Responsibilities
#   • Patient Management (CRUD)
#   • Appointment Scheduling
#   • Feedback Module
#   • Integration Testing
#   • Workflow Validation
#   • End-to-End Test Cases
#
# Run:  python -m pytest tests/test_scrum_master.py -v
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
    create_patient,
    get_patients,
    update_patient,
    delete_patient,
    schedule_appointment,
    get_appointments,
    delete_appointment,
    submit_feedback,
    get_all_feedback,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_temp_db():
    """Return a path to a fresh temporary SQLite database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


# ═════════════════════════════════════════════════════════════════════════════
# Base setup mixin — reused by all test classes
# ═════════════════════════════════════════════════════════════════════════════

class _TempDbMixin:
    """Mixin that sets up and tears down a fresh temp DB per test."""

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
# 1. Patient Management — CRUD Integration Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestPatientManagement(_TempDbMixin, unittest.TestCase):
    """End-to-end Create / Read / Update / Delete workflow for patients."""

    def test_patient_create_returns_positive_id(self):
        pid = create_patient(self.username, "Alice Smith", 55, "Female",
                             165.0, 65.0, "No", "Yes", "No", "No")
        self.assertIsInstance(pid, int)
        self.assertGreater(pid, 0)

    def test_patient_read_after_create(self):
        create_patient(self.username, "Alice Smith", 55, "Female",
                       165.0, 65.0, "No", "Yes", "No", "No")
        patients = get_patients(self.username)
        self.assertEqual(len(patients), 1)
        self.assertEqual(patients[0]["name"], "Alice Smith")
        self.assertEqual(patients[0]["age"], 55)

    def test_patient_update(self):
        pid = create_patient(self.username, "Alice Smith", 55, "Female",
                             165.0, 65.0, "No", "Yes", "No", "No")
        update_patient(self.username, pid, "Alice Doe", 56, "Female",
                       165.0, 64.0, "Yes", "Yes", "No", "No")
        patients = get_patients(self.username)
        self.assertEqual(patients[0]["name"], "Alice Doe")
        self.assertEqual(patients[0]["age"], 56)
        self.assertEqual(patients[0]["diabetes"], "Yes")

    def test_patient_delete(self):
        pid = create_patient(self.username, "Alice Smith", 55, "Female",
                             165.0, 65.0, "No", "Yes", "No", "No")
        delete_patient(self.username, pid)
        patients = get_patients(self.username)
        self.assertEqual(len(patients), 0)

    def test_patient_crud_full_workflow(self):
        """End-to-end: create → read → update → delete in one workflow."""
        # Create
        pid = create_patient(self.username, "Bob Kumar", 42, "Male",
                             175.0, 80.0, "No", "No", "Yes", "No")
        self.assertGreater(pid, 0)

        # Read
        patients = get_patients(self.username)
        self.assertEqual(len(patients), 1)
        self.assertEqual(patients[0]["name"], "Bob Kumar")

        # Update
        update_patient(self.username, pid, "Bob Kumar", 43, "Male",
                       175.0, 78.0, "No", "No", "No", "No")
        updated = get_patients(self.username)
        self.assertEqual(updated[0]["age"], 43)
        self.assertEqual(updated[0]["smoking_status"], "No")

        # Delete
        delete_patient(self.username, pid)
        self.assertEqual(len(get_patients(self.username)), 0)

    def test_patients_scoped_to_user(self):
        """Patients created by one user must not appear for another."""
        create_patient(self.username, "Private Patient", 30, "Male",
                       170.0, 70.0, "No", "No", "No", "No")
        other_patients = get_patients("other_doctor")
        self.assertEqual(len(other_patients), 0)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Appointment Scheduling — Workflow Validation
# ═════════════════════════════════════════════════════════════════════════════

class TestAppointments(_TempDbMixin, unittest.TestCase):
    """Tests for scheduling, retrieval, and deletion of appointments."""

    def test_schedule_appointment_returns_positive_id(self):
        aid = schedule_appointment(self.username, "Bob Lee",
                                   "2024-05-10 14:00:00", "Routine checkup")
        self.assertIsInstance(aid, int)
        self.assertGreater(aid, 0)

    def test_scheduled_appointment_is_retrievable(self):
        schedule_appointment(self.username, "Bob Lee",
                             "2024-05-10 14:00:00", "Routine checkup")
        appts = get_appointments(self.username)
        self.assertEqual(len(appts), 1)
        self.assertEqual(appts[0]["patient_name"], "Bob Lee")
        self.assertEqual(appts[0]["notes"], "Routine checkup")

    def test_delete_appointment(self):
        aid = schedule_appointment(self.username, "Bob Lee",
                                   "2024-05-10 14:00:00", "Routine checkup")
        delete_appointment(self.username, aid)
        appts = get_appointments(self.username)
        self.assertEqual(len(appts), 0)

    def test_multiple_appointments_all_retrieved(self):
        schedule_appointment(self.username, "Patient A",
                             "2024-06-01 09:00:00", "Follow-up")
        schedule_appointment(self.username, "Patient B",
                             "2024-06-02 10:00:00", "Initial consult")
        appts = get_appointments(self.username)
        self.assertEqual(len(appts), 2)

    def test_appointments_scoped_to_user(self):
        """Appointments must be isolated per user."""
        schedule_appointment(self.username, "Patient X",
                             "2024-07-01 08:00:00", "Check-up")
        other_appts = get_appointments("other_doctor")
        self.assertEqual(len(other_appts), 0)

    def test_appointment_full_workflow(self):
        """End-to-end: schedule → retrieve → delete."""
        aid = schedule_appointment(self.username, "Maria Santos",
                                   "2025-01-15 11:30:00", "Kidney biopsy follow-up")
        self.assertGreater(aid, 0)

        appts = get_appointments(self.username)
        self.assertEqual(appts[0]["patient_name"], "Maria Santos")
        self.assertIn("biopsy", appts[0]["notes"].lower())

        delete_appointment(self.username, aid)
        self.assertEqual(len(get_appointments(self.username)), 0)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Feedback Module — Integration Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestFeedback(_TempDbMixin, unittest.TestCase):
    """Tests for the patient/user feedback submission and retrieval."""

    def test_submit_feedback_and_retrieve(self):
        submit_feedback(self.username, 5, "Perfect prediction", 1)
        fb = get_all_feedback()
        self.assertEqual(len(fb), 1)
        self.assertEqual(fb[0]["rating"], 5)
        self.assertEqual(fb[0]["comments"], "Perfect prediction")

    def test_feedback_rating_stored_correctly(self):
        submit_feedback(self.username, 3, "Average results", None)
        fb = get_all_feedback()
        self.assertEqual(fb[0]["rating"], 3)

    def test_multiple_feedback_entries(self):
        submit_feedback(self.username, 4, "Good", 1)
        submit_feedback(self.username, 2, "Needs improvement", 2)
        fb = get_all_feedback()
        self.assertGreaterEqual(len(fb), 2)

    def test_feedback_without_prediction_id(self):
        """Feedback should be submittable with no linked prediction (prediction_id=None)."""
        submit_feedback(self.username, 5, "General feedback", None)
        fb = get_all_feedback()
        self.assertTrue(any(f["comments"] == "General feedback" for f in fb))


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
