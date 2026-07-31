# Chronic Kidney Disease Risk Prediction System

## 🫁 Overview
An AI-powered clinical decision support web application that predicts the stage of Chronic Kidney Disease (CKD) using 35 clinical biomarkers and a trained machine learning model. The system also includes user management, a medical assistant chatbot, PDF report generation, and explainable AI (SHAP) to assist both patients and clinicians.

---

## ✨ Features
- **Predictive ML Models**: Predict CKD stage using 35 biomarkers with top-performing classifiers.
- **Explainable AI (SHAP)**: Understand local predictions with SHAP explainer integration.
- **PDF Report Generation**: Generate and email detailed prediction reports and recommendations.
- **Medical Assistant Chatbot**: Interactive chatbot using an extensive offline CKD Knowledge Base.
- **User Authentication & Roles**: Registration, login, and tailored views for admins and standard users.
- **Patient & Appointment Management**: Track patient histories, schedule appointments, and manage feedback.
- **Admin Dashboard**: View system metrics, patient logs, and detailed audit trails.

---

## 📁 Project Structure

```
project/
│
├── app.py                          # Main Streamlit application
├── train_model.py                  # Model training script
├── auth.py                         # Authentication and database operations
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── users.db                        # SQLite database for users, patients, and predictions
│
├── model/
│     └── kidney_model.pkl          # Trained model bundle (generated)
│
├── dataset/
│     ├── Training_CKD_dataset.csv  # Training data (21,001 rows)
│     └── Testing_CKD_dataset.csv   # Testing data
│
├── notebooks/
│     ├── 01_Data_Cleaning.ipynb    # Data quality & cleaning
│     ├── 02_EDA.ipynb              # Exploratory Data Analysis
│     └── 03_Model_Training.ipynb   # Model training & evaluation
│
├── ckd_utils/
│     ├── __init__.py
│     ├── preprocessing.py          # Data loading, encoding, scaling
│     ├── prediction.py             # Model loading & inference
│     ├── knowledge_base.py         # Offline CKD knowledge base
│     ├── chatbot.py                # Chatbot logic
│     ├── shap_explainer.py         # SHAP explainability
│     └── report_generator.py       # PDF generation
│
├── tests/
│     ├── test_scrum_master.py      # Unit tests
│     ├── test_business_owner.py
│     ├── test_chatbot.py
│     ├── test_ckd.py
│     ├── test_product_owner.py
│     └── test_team_leader.py
│
└── assets/
      ├── confusion_matrix.png       # Generated after training
      └── model_comparison.png       # Generated after training
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```
This will:
- Load and preprocess the training & testing datasets
- Train 4 classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- Select the best model by weighted F1-score
- Save the model bundle to `model/kidney_model.pkl`
- Generate performance charts in `assets/`

### 3. Launch the Web App
```bash
streamlit run app.py
```

---

## 🎯 Target Classes

| Class | CKD Stage | eGFR Range |
|-------|-----------|------------|
| Healthy Kidney | — | ≥ 90 mL/min |
| Mild CKD (Stage 1–2) | Stage 1–2 | 60–89 mL/min |
| Moderate CKD (Stage 3) | Stage 3 | 30–59 mL/min |
| Severe CKD (Stage 4) | Stage 4 | 15–29 mL/min |
| Kidney Failure (Stage 5) | Stage 5 | < 15 mL/min |

---

## 🔬 Input Features (35 total)

| Category | Features |
|----------|---------|
| Demographics | Age, Gender, BMI |
| Vital Signs | Systolic BP, Diastolic BP, Heart Rate |
| Kidney Function | Serum Creatinine, BUN, eGFR, Urine Albumin, Urine Protein, ACR, Urine Specific Gravity |
| Electrolytes | Sodium, Potassium, Calcium, Phosphorus, Chloride, Bicarbonate |
| Blood Panel | Hemoglobin, RBC Count, WBC Count, Platelet Count, Packed Cell Volume |
| Glucose & Lipids | Blood Glucose Random, Fasting Glucose, HbA1c, Cholesterol, Triglycerides |
| Serum Proteins | Serum Albumin, Total Protein |
| Risk Factors | Diabetes, Hypertension, Smoking Status, Family History of Kidney Disease |

---

## 🤖 Machine Learning Models Trained

- **Logistic Regression** – Baseline linear classifier
- **Decision Tree** – Non-linear tree-based classifier
- **Random Forest** – Ensemble bagging classifier ⭐ (typically best)
- **Gradient Boosting** – Ensemble boosting classifier

The best model is selected based on **weighted F1-score** on the test set.

---

## 📊 Evaluation Metrics

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
- Confusion Matrix
- Full Classification Report (per-class)

---

## 🖥️ Web Application Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Overview, CKD stages, key stats, how it works |
| 🔑 Login/Register | Secure authentication for users and admins |
| 🔬 Prediction | Patient input form → CKD stage prediction + SHAP explainer + PDF Report |
| 📊 Analytics | Dataset summary, class distribution, feature charts, confusion matrix |
| 💬 Medical Assistant | Interactive AI Chatbot powered by an offline CKD Knowledge Base |
| 🗓️ Appointments | Schedule and manage appointments |
| 👨‍💼 Admin Dashboard | View audit logs, system metrics, manage users and patient records |
| ℹ️ About | CKD explanation, prediction workflow, algorithm details |

---

## ⚠️ Medical Disclaimer
This application is intended for **educational and research purposes only**.
It is **not a substitute** for professional medical advice, diagnosis, or treatment.
Always consult a qualified healthcare provider for medical decisions.
