import sqlite3
import hashlib
import re
import streamlit as st
import datetime
import os
# Force reload

DB_FILE = "users.db"


def get_conn():
    """Return a SQLite connection with WAL mode and a 30-second busy timeout.

    WAL (Write-Ahead Logging) allows concurrent reads and writes so that
    Streamlit's multi-threaded / multi-process reruns don't cause
    'attempt to write a readonly database' errors on Windows.
    """
    # Resolve to an absolute path so connections from different working
    # directories all point at the same file.
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_FILE)
    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn

def init_db():
    create_users_table()
    create_predictions_table()

def create_users_table():
    conn = get_conn()
    try:
        with conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )
            ''')
    finally:
        conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, email, password):
    # Username Validation
    if not (4 <= len(username) <= 20):
        return False, "Username must be between 4 and 20 characters."
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Username can only contain letters, numbers, and underscores."
    if not re.search(r'[a-zA-Z]', username):
        return False, "Username must contain at least one alphabet."
        
    # Email Validation
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_regex, email):
        return False, "Invalid email format."

    conn = get_conn()
    try:
        with conn:
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE username = ?', (username,))
            if c.fetchone():
                return False, "Username already exists."
                
            c.execute('SELECT * FROM users WHERE email = ?', (email,))
            if c.fetchone():
                return False, "Email already exists."
                
            hashed_pwd = hash_password(password)
            c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', 
                      (username, email, hashed_pwd))
    finally:
        conn.close()
    # Open a fresh connection for audit log (avoids nested lock on Windows)
    log_audit_action(username, "User Registered", f"Email: {email}")
    return True, "Registration successful! Please login."

def login_user(username, password):
    hashed_pwd = hash_password(password)
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hashed_pwd))
        found = c.fetchone() is not None
    finally:
        conn.close()  # Always close before calling log_audit_action
    # Open a fresh connection for audit log (avoids nested lock on Windows)
    if found:
        log_audit_action(username, "User Logged In", "Successfully authenticated")
        return True
    log_audit_action(username, "Login Attempt Failed", "Incorrect credentials")
    return False

def logout_user():
    st.session_state["logged_in"] = False
    st.session_state["auth_page"] = "login"

def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def create_predictions_table():
    conn = get_conn()
    try:
        with conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS predictions_v2 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    patient_name TEXT NOT NULL,
                    prediction_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    features_json TEXT NOT NULL
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    name TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    gender TEXT NOT NULL,
                    height REAL NOT NULL,
                    weight REAL NOT NULL,
                    diabetes TEXT NOT NULL,
                    hypertension TEXT NOT NULL,
                    smoking_status TEXT NOT NULL,
                    family_history TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS appointments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    patient_name TEXT NOT NULL,
                    appointment_date TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    prediction_id INTEGER,
                    rating INTEGER NOT NULL,
                    comments TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            ''')
            # ── Non-destructive migration: add extended clinical columns ──────
            # Each ALTER TABLE is wrapped in try/except so that columns which
            # already exist simply produce an "OperationalError: duplicate column"
            # which we silently ignore.
            new_columns = [
                ("patient_id",               "TEXT    DEFAULT ''"),
                ("diastolic_bp",             "REAL    DEFAULT 0"),
                ("systolic_bp",              "REAL    DEFAULT 0"),
                ("serum_creatinine",         "REAL    DEFAULT 0"),
                ("egfr",                     "REAL    DEFAULT 0"),
                ("bun",                      "REAL    DEFAULT 0"),
                ("urine_protein",            "REAL    DEFAULT 0"),
                ("urine_albumin",            "REAL    DEFAULT 0"),
                ("acr",                      "REAL    DEFAULT 0"),
                ("serum_albumin_clinical",   "REAL    DEFAULT 0"),
                ("bicarbonate",              "REAL    DEFAULT 0"),
                ("last_ckd_stage",           "TEXT    DEFAULT ''"),
                ("last_confidence",          "REAL    DEFAULT 0"),
                ("last_risk_level",          "TEXT    DEFAULT ''"),
                ("last_shap_features",       "TEXT    DEFAULT '[]'"),
                ("last_recommendation",      "TEXT    DEFAULT ''"),
                ("last_prediction_at",       "TEXT    DEFAULT ''"),
            ]
            for col_name, col_type in new_columns:
                try:
                    c.execute(f"ALTER TABLE patients ADD COLUMN {col_name} {col_type}")
                except Exception:
                    pass  # Column already exists – safe to ignore
    finally:
        conn.close()

def save_prediction(username, patient_name, prediction_label, confidence, features_dict, created_at=None):
    import json
    if created_at is None:
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    features_json = json.dumps(features_dict)
    conn = get_conn()
    try:
        with conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO predictions_v2 (
                    username, patient_name, prediction_label, confidence, created_at, features_json
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                username, patient_name, prediction_label, confidence, created_at, features_json
            ))
            pred_id = c.lastrowid
    finally:
        conn.close()
    log_audit_action(username, "Prediction Generated", f"Patient: {patient_name}, Stage: {prediction_label} (ID: {pred_id})")
    return pred_id

def get_predictions(username, search_query=None, start_date=None, end_date=None):
    conn = get_conn()
    try:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        query = "SELECT * FROM predictions_v2 WHERE username = ?"
        params = [username]
        
        if search_query:
            query += " AND patient_name LIKE ?"
            params.append(f"%{search_query}%")
            
        if start_date:
            query += " AND date(created_at) >= ?"
            params.append(start_date)
            
        if end_date:
            query += " AND date(created_at) <= ?"
            params.append(end_date)
            
        query += " ORDER BY created_at DESC"
        
        c.execute(query, params)
        rows = c.fetchall()
        
        import json
        result = []
        for r in rows:
            d = dict(r)
            try:
                features = json.loads(d['features_json'])
                d.update(features)
            except:
                pass
            result.append(d)
        return result
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Patient Profile Management
# ---------------------------------------------------------------------------
def create_patient(username, name, age, gender, height, weight, diabetes, hypertension, smoking_status, family_history):
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_conn()
    try:
        with conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO patients (
                    username, name, age, gender, height, weight, diabetes, hypertension, smoking_status, family_history, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (username, name, age, gender, height, weight, diabetes, hypertension, smoking_status, family_history, created_at))
            patient_id = c.lastrowid
    finally:
        conn.close()
    log_audit_action(username, "Create Patient", f"Created patient: {name} (ID: {patient_id})")
    return patient_id


def upsert_patient_from_prediction(
    username, existing_patient_id,
    name, age, gender, height, weight,
    diabetes, hypertension, smoking_status, family_history,
    diastolic_bp=0, systolic_bp=0, serum_creatinine=0, egfr=0, bun=0,
    urine_protein=0, urine_albumin=0, acr=0,
    serum_albumin_clinical=0, bicarbonate=0,
    last_ckd_stage="", last_confidence=0.0, last_risk_level="",
    last_shap_features="[]", last_recommendation="",
    last_prediction_at="",
):
    """Create or update a patient profile from prediction data.

    Matching priority:
      1. If ``existing_patient_id`` is given (doctor selected a saved profile)
         → update that row.
      2. Otherwise look for a row where (username, name, age) match.
      3. If no match → insert a new row with an auto-generated patient_id.
    """
    import json
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not last_prediction_at:
        last_prediction_at = now

    conn = get_conn()
    try:
        with conn:
            c = conn.cursor()

            target_id = None

            # Priority 1: explicit patient selected
            if existing_patient_id:
                c.execute(
                    "SELECT id FROM patients WHERE id = ? AND username = ?",
                    (existing_patient_id, username),
                )
                row = c.fetchone()
                if row:
                    target_id = row[0]

            # Priority 2: match by name + age
            if target_id is None:
                c.execute(
                    "SELECT id FROM patients WHERE username = ? AND name = ? AND age = ?",
                    (username, name, age),
                )
                row = c.fetchone()
                if row:
                    target_id = row[0]

            if target_id is not None:
                # UPDATE existing profile
                c.execute('''
                    UPDATE patients SET
                        name = ?, age = ?, gender = ?, height = ?, weight = ?,
                        diabetes = ?, hypertension = ?, smoking_status = ?, family_history = ?,
                        diastolic_bp = ?, systolic_bp = ?, serum_creatinine = ?,
                        egfr = ?, bun = ?, urine_protein = ?, urine_albumin = ?,
                        acr = ?, serum_albumin_clinical = ?, bicarbonate = ?,
                        last_ckd_stage = ?, last_confidence = ?, last_risk_level = ?,
                        last_shap_features = ?, last_recommendation = ?, last_prediction_at = ?
                    WHERE id = ? AND username = ?
                ''', (
                    name, age, gender, height, weight,
                    diabetes, hypertension, smoking_status, family_history,
                    diastolic_bp, systolic_bp, serum_creatinine,
                    egfr, bun, urine_protein, urine_albumin,
                    acr, serum_albumin_clinical, bicarbonate,
                    last_ckd_stage, last_confidence, last_risk_level,
                    last_shap_features, last_recommendation, last_prediction_at,
                    target_id, username,
                ))
                log_audit_action(username, "Auto-Update Patient",
                                 f"Updated from prediction: {name} (ID: {target_id})")
                return target_id
            else:
                # INSERT new profile
                # Generate patient_id like CKD-0001
                c.execute("SELECT MAX(id) FROM patients WHERE username = ?", (username,))
                max_row = c.fetchone()
                next_num = (max_row[0] or 0) + 1
                auto_pid = f"CKD-{next_num:04d}"

                c.execute('''
                    INSERT INTO patients (
                        username, name, age, gender, height, weight,
                        diabetes, hypertension, smoking_status, family_history, created_at,
                        patient_id,
                        diastolic_bp, systolic_bp, serum_creatinine, egfr, bun,
                        urine_protein, urine_albumin, acr, serum_albumin_clinical, bicarbonate,
                        last_ckd_stage, last_confidence, last_risk_level,
                        last_shap_features, last_recommendation, last_prediction_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ''', (
                    username, name, age, gender, height, weight,
                    diabetes, hypertension, smoking_status, family_history, now,
                    auto_pid,
                    diastolic_bp, systolic_bp, serum_creatinine, egfr, bun,
                    urine_protein, urine_albumin, acr, serum_albumin_clinical, bicarbonate,
                    last_ckd_stage, last_confidence, last_risk_level,
                    last_shap_features, last_recommendation, last_prediction_at,
                ))
                new_id = c.lastrowid
                log_audit_action(username, "Auto-Create Patient",
                                 f"Created from prediction: {name} ({auto_pid})")
                return new_id
    finally:
        conn.close()


def get_patient_by_id(username, patient_db_id):
    """Return a single patient dict by primary key, or None if not found."""
    conn = get_conn()
    try:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            "SELECT * FROM patients WHERE id = ? AND username = ?",
            (patient_db_id, username),
        )
        row = c.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def get_patients(username):
    conn = get_conn()
    try:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM patients WHERE username = ? ORDER BY name ASC', (username,))
        rows = c.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

def update_patient(username, patient_id, name, age, gender, height, weight, diabetes, hypertension, smoking_status, family_history):
    conn = get_conn()
    try:
        with conn:
            c = conn.cursor()
            c.execute('''
                UPDATE patients SET
                    name = ?, age = ?, gender = ?, height = ?, weight = ?,
                    diabetes = ?, hypertension = ?, smoking_status = ?, family_history = ?
                WHERE id = ? AND username = ?
            ''', (name, age, gender, height, weight, diabetes, hypertension, smoking_status, family_history, patient_id, username))
    finally:
        conn.close()
    log_audit_action(username, "Update Patient", f"Updated patient: {name} (ID: {patient_id})")
    return True

def delete_patient(username, patient_id):
    conn = get_conn()
    try:
        with conn:
            c = conn.cursor()
            c.execute('DELETE FROM patients WHERE id = ? AND username = ?', (patient_id, username))
    finally:
        conn.close()
    log_audit_action(username, "Delete Patient", f"Deleted patient ID: {patient_id}")
    return True

# ---------------------------------------------------------------------------
# Appointment Reminders
# ---------------------------------------------------------------------------
def schedule_appointment(username, patient_name, appointment_date, notes):
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_conn()
    try:
        with conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO appointments (username, patient_name, appointment_date, notes, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, patient_name, appointment_date, notes, created_at))
            appointment_id = c.lastrowid
    finally:
        conn.close()
    log_audit_action(username, "Schedule Appointment", f"Scheduled appointment for {patient_name} on {appointment_date}")
    return appointment_id

def get_appointments(username):
    conn = get_conn()
    try:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM appointments WHERE username = ? ORDER BY appointment_date ASC', (username,))
        rows = c.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

def delete_appointment(username, appointment_id):
    conn = get_conn()
    try:
        with conn:
            c = conn.cursor()
            c.execute('DELETE FROM appointments WHERE id = ? AND username = ?', (appointment_id, username))
    finally:
        conn.close()
    log_audit_action(username, "Cancel Appointment", f"Cancelled appointment ID: {appointment_id}")
    return True

# ---------------------------------------------------------------------------
# Feedback Module
# ---------------------------------------------------------------------------
def submit_feedback(username, rating, comments, prediction_id=None):
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_conn()
    try:
        with conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO feedback (username, prediction_id, rating, comments, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, prediction_id, rating, comments, created_at))
    finally:
        conn.close()
    log_audit_action(username, "Submit Feedback", f"Rated prediction: {rating} stars")
    return True

def get_all_feedback():
    conn = get_conn()
    try:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM feedback ORDER BY created_at DESC')
        rows = c.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Audit Logs
# ---------------------------------------------------------------------------
def log_audit_action(username, action, details):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = get_conn()
        try:
            with conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO audit_logs (username, action, details, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (username, action, details, timestamp))
            return True
        finally:
            conn.close()
    except Exception:
        # Audit logging must never crash the main application
        return False

def get_audit_logs():
    conn = get_conn()
    try:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 500')
        rows = c.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Admin Panel Metrics
# ---------------------------------------------------------------------------
def get_admin_metrics():
    conn = get_conn()
    metrics = {}
    try:
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM users')
        metrics["total_users"] = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM predictions_v2')
        metrics["total_predictions"] = c.fetchone()[0]
        
        c.execute('SELECT username, email FROM users ORDER BY username ASC')
        metrics["users_list"] = [{"username": r[0], "email": r[1]} for r in c.fetchall()]
        return metrics
    finally:
        conn.close()

