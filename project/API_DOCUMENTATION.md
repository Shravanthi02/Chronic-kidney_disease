# CKD Prediction System - Backend API & Database Reference

This document serves as the developer and administrator reference for the backend functions, model utilities, and database schemas implemented in the CKD Prediction project.

---

## 🗄️ Database Schemas (`users.db`)

All data persistence is handled via SQLite. The schema contains the following tables:

### 1. `users`
Stores user credentials for clinicians.
*   `id` (INTEGER, Primary Key)
*   `username` (TEXT, Unique)
*   `email` (TEXT, Unique)
*   `password` (TEXT, Hashed using SHA-256)

### 2. `predictions_v2`
Stores the results of patient risk predictions.
*   `id` (INTEGER, Primary Key)
*   `username` (TEXT) - The clinician who initiated the request.
*   `patient_name` (TEXT)
*   `prediction_label` (TEXT) - E.g. 'Healthy Kidney', 'Stage 3 CKD', etc.
*   `confidence` (REAL) - Percentage confidence.
*   `created_at` (TEXT) - ISO datetime string.
*   `features_json` (TEXT) - Key-value JSON string containing all 35 clinical input features.

### 3. `patients`
Stores secure profiles of patient demographics and medical histories.
*   `id` (INTEGER, Primary Key)
*   `username` (TEXT) - The user owning this patient's records.
*   `name` (TEXT)
*   `age` (INTEGER)
*   `gender` (TEXT) - 'Male' or 'Female'
*   `height` (REAL)
*   `weight` (REAL)
*   `diabetes` (TEXT) - 'Yes' or 'No'
*   `hypertension` (TEXT) - 'Yes' or 'No'
*   `smoking_status` (TEXT) - 'Yes' or 'No'
*   `family_history` (TEXT) - 'Yes' or 'No'
*   `created_at` (TEXT)

### 4. `appointments`
Manages follow-up calendar alerts.
*   `id` (INTEGER, Primary Key)
*   `username` (TEXT)
*   `patient_name` (TEXT)
*   `appointment_date` (TEXT) - Datetime string.
*   `notes` (TEXT)
*   `created_at` (TEXT)

### 5. `feedback`
Audits clinician ratings on model outputs.
*   `id` (INTEGER, Primary Key)
*   `username` (TEXT)
*   `prediction_id` (INTEGER)
*   `rating` (INTEGER) - 1 to 5 scale.
*   `comments` (TEXT)
*   `created_at` (TEXT)

### 6. `audit_logs`
Automated trail of all clinician and admin activities.
*   `id` (INTEGER, Primary Key)
*   `username` (TEXT)
*   `action` (TEXT) - E.g., 'User Logged In', 'Delete Patient', etc.
*   `details` (TEXT)
*   `timestamp` (TEXT)

---

## 🛠️ Backend API Reference (`auth.py`)

### Authentication Functions

#### `register_user(username, email, password) -> (bool, str)`
Registers a new clinician account. Returns status and error/success message.

#### `login_user(username, password) -> bool`
Validates clinician credentials. Logs audit records.

#### `logout_user()`
Resets session state parameters.

---

### Patient Profile Functions

#### `create_patient(username, name, age, gender, height, weight, diabetes, hypertension, smoking_status, family_history) -> int`
Saves a new patient profile. Returns the ID of the created row.

#### `get_patients(username) -> list[dict]`
Lists all patients registered under the clinician's username.

#### `update_patient(username, patient_id, name, age, gender, height, weight, diabetes, hypertension, smoking_status, family_history) -> bool`
Updates patient profile fields. Returns True if successful.

#### `delete_patient(username, patient_id) -> bool`
Deletes a patient record.

---

### Appointment Scheduler

#### `schedule_appointment(username, patient_name, appointment_date, notes) -> int`
Schedules a new reminder.

#### `get_appointments(username) -> list[dict]`
Retrieves upcoming appointments sorted by date.

#### `delete_appointment(username, appointment_id) -> bool`
Cancels a scheduled appointment.

---

### Auditing & Metrics

#### `log_audit_action(username, action, details) -> bool`
Appends a dynamic event record to the audit logs.

#### `get_audit_logs() -> list[dict]`
Lists all security audit entries (most recent first).

#### `get_admin_metrics() -> dict`
Compiles general statistics for the admin dashboard.
