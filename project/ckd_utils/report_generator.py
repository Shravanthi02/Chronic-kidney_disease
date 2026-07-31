from fpdf import FPDF
import datetime


# ---------------------------------------------------------------------------
# Helper: replace/strip characters unsupported by fpdf2 built-in fonts
# ---------------------------------------------------------------------------
def safe_text(text: str) -> str:
    """
    Replace common Unicode characters that helvetica (latin-1) cannot render,
    then drop any remaining characters outside the latin-1 range.
    """
    replacements = {
        "\u2013": "-",    # en-dash  (the culprit in "Stage 1-2")
        "\u2014": "-",    # em-dash
        "\u2192": "->",   # right arrow
        "\u2190": "<-",   # left arrow
        "\u2026": "...",  # ellipsis
        "\u00b2": "2",    # superscript 2
        "\u00b0": " deg", # degree sign
        "\u2265": ">=",   # greater-than-or-equal
        "\u2264": "<=",   # less-than-or-equal
        "\u00b1": "+/-",  # plus-minus
        "\u2018": "'",    # left single quotation mark
        "\u2019": "'",    # right single quotation mark
        "\u201c": '"',    # left double quotation mark
        "\u201d": '"',    # right double quotation mark
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Drop anything else outside latin-1
    return text.encode("latin-1", errors="ignore").decode("latin-1")


# ---------------------------------------------------------------------------
# Recommendation generator
# ---------------------------------------------------------------------------
def generate_detailed_recommendations(input_dict, stage_label):
    """
    Generate detailed health recommendations based on patient input and CKD stage.
    Categorizes recommendations into Diet Suggestions, Lifestyle Recommendations, 
    and Medical Consultation (including consulting a nephrologist for severe stages).
    """
    recommendations = []

    # Normalise stage label to handle en-dash vs hyphen variants from the model
    normalised_label = safe_text(stage_label)

    # 1. Base stage recommendation & Medical Consultation
    if "Healthy" in normalised_label or "Healthy Kidney" in stage_label:
        assessment_text = (
            "Great news! Your kidneys appear healthy. Maintain a balanced diet, stay hydrated, "
            "exercise regularly, and get annual check-ups to monitor your kidney health."
        )
        medical_text = (
            "No specialized medical intervention is required. Continue routine annual exams "
            "with your primary care physician to monitor blood pressure and basic kidney function tests."
        )
    elif "Stage 1-2" in normalised_label or "Stage 1" in normalised_label or "Stage 2" in normalised_label:
        assessment_text = (
            "Early-stage Chronic Kidney Disease (Stage 1-2) detected. Your kidneys are functioning well, "
            "but there are early signs of damage. Early intervention can stop or slow down progression."
        )
        medical_text = (
            "Consult your primary care physician. We recommend obtaining a baseline consultation "
            "with a nephrologist (kidney specialist) to establish a monitoring and management plan."
        )
    elif "Stage 3" in normalised_label:
        assessment_text = (
            "Moderate Chronic Kidney Disease (Stage 3) detected. Kidney function is moderately reduced. "
            "It is highly important to take active clinical measures to prevent further decline."
        )
        medical_text = (
            "**CRITICAL ACTION REQUIRED**: Schedule a consultation with a nephrologist (kidney specialist) "
            "immediately. Regular lab work (eGFR, serum creatinine, urine protein) must be monitored closely."
        )
    elif "Stage 4" in normalised_label:
        assessment_text = (
            "Severe Chronic Kidney Disease (Stage 4) detected. Kidney function is severely reduced. "
            "Close clinical management is vital to manage symptoms and prepare for advanced options."
        )
        medical_text = (
            "**CRITICAL ACTION REQUIRED**: Immediate and regular treatment under a nephrologist is required. "
            "You should begin discussions and planning for renal replacement options (dialysis or kidney transplant)."
        )
    elif "Stage 5" in normalised_label:
        assessment_text = (
            "Kidney Failure (Stage 5) detected. Kidney function is near or at failure. "
            "Immediate medical treatment is necessary to support life and manage symptoms."
        )
        medical_text = (
            "**EMERGENCY CLINICAL INTERVENTION REQUIRED**: Contact your nephrologist or go to a kidney care center "
            "immediately. Renal replacement therapy (dialysis or kidney transplant evaluation) is urgently needed."
        )
    else:
        assessment_text = "CKD clinical assessment complete. Please share these results with your healthcare provider."
        medical_text = "Consult a healthcare professional for diagnosis, assessment, and treatment planning."

    recommendations.append(("Overall Assessment", assessment_text))
    recommendations.append(("Medical Consultation", medical_text))

    # 2. Diet Suggestions based on stage
    if "Healthy" in normalised_label:
        diet_text = (
            "Follow a balanced diet rich in vegetables, fruits, whole grains, and lean proteins. "
            "Stay well-hydrated (approx. 2-2.5 liters of water daily). Avoid excessive consumption of processed foods and salt."
        )
    elif "Stage 1-2" in normalised_label:
        diet_text = (
            "Adopt a kidney-healthy diet. Limit sodium intake to under 2,000 mg per day. "
            "Ensure moderate, high-quality protein consumption and maintain healthy portion sizes. Stay hydrated."
        )
    else:  # Stages 3, 4, 5
        diet_text = (
            "**Renal Diet Restriction Required**: Strictly limit dietary sodium (less than 1,500 mg/day). "
            "Limit potassium (avoid bananas, oranges, potatoes) and phosphorus (limit dairy, nuts, cola). "
            "Adjust protein intake according to your nephrologist's guidelines, and consult a registered renal dietitian."
        )
    recommendations.append(("Dietary Suggestions", diet_text))

    # 3. Lifestyle Recommendations
    lifestyle_tips = []
    
    # Smoking
    if "Smoking_Status" in input_dict and input_dict.get("Smoking_Status", "No") == "Yes":
        lifestyle_tips.append("Quit smoking immediately, as smoking worsens blood flow to kidneys.")
    else:
        lifestyle_tips.append("Avoid smoking and exposure to secondhand smoke.")

    # BMI
    bmi = input_dict.get("BMI", 25.0)
    if bmi >= 30:
        lifestyle_tips.append("Aim for weight loss (target BMI < 25) through structured, low-impact exercise and calorie control.")
    elif bmi >= 25:
        lifestyle_tips.append("Maintain moderate weight control to reduce kidney workload.")
    else:
        lifestyle_tips.append("Maintain a stable healthy weight.")

    # Exercise
    lifestyle_tips.append("Engage in moderate physical activity (such as brisk walking) for 150 minutes per week.")
    lifestyle_tips.append("Avoid over-the-counter NSAID pain relievers (like ibuprofen, naproxen) which are toxic to kidneys.")

    recommendations.append(("Lifestyle Recommendations", " ".join(lifestyle_tips)))

    # 4. Blood Pressure Management
    sys_bp = input_dict.get("Systolic_BP")
    dia_bp = input_dict.get("Diastolic_BP")
    if sys_bp is not None and dia_bp is not None:
        if sys_bp >= 130 or dia_bp >= 80:
            recommendations.append(("Blood Pressure Management",
                f"Your recorded BP of {sys_bp}/{dia_bp} mmHg is elevated. High blood pressure accelerates kidney damage. "
                "Target a blood pressure below 130/80 mmHg. Adhere strictly to any prescribed antihypertensive medications."))
        else:
            recommendations.append(("Blood Pressure Management",
                f"Your blood pressure ({sys_bp}/{dia_bp} mmHg) is within the recommended target range (< 130/80 mmHg). "
                "Continue regular monitoring to maintain control."))

    # 5. Diabetes & Hypertension Indicators
    if input_dict.get("Diabetes") == "Yes":
        recommendations.append(("Glycemic Control",
            "Maintain strict glycemic control (target HbA1c < 7.0%). Monitor blood sugar levels daily and "
            "take diabetic medications as prescribed."))
    if input_dict.get("Hypertension") == "Yes" and not (sys_bp is not None and sys_bp >= 130):
        recommendations.append(("Hypertension Care",
            "Since you have a history of hypertension, check your blood pressure twice daily. "
            "Stick to a low-sodium diet and avoid high-stress triggers."))

    # 6. Urine Protein / ACR Indicators
    ua = input_dict.get("Urine_Albumin", 0)
    up = input_dict.get("Urine_Protein", 0)
    acr_val = input_dict.get("Albumin_Creatinine_Ratio", 0)
    if ua >= 30 or up >= 30 or acr_val >= 30:
        recommendations.append(("Proteinuria Management",
            "Elevated urine albumin/protein or ACR indicates kidney barrier breakdown. "
            "Your doctor may prescribe ACE inhibitors or ARBs, which protect kidneys and reduce protein leakage."))

    return recommendations


# ---------------------------------------------------------------------------
# PDF Report
# ---------------------------------------------------------------------------
import os

class PDFReport(FPDF):
    def __init__(self, logo_path=None, hospital_name="Metropolitan Kidney Care Center"):
        super().__init__()
        self.logo_path = logo_path
        self.hospital_name = hospital_name

    def header(self):
        # Draw hospital style banner
        self.set_fill_color(26, 54, 93)  # Dark Blue
        self.rect(0, 0, 210, 42, "F")

        # Check logo
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                self.image(self.logo_path, x=15, y=10, w=22)
            except Exception:
                pass
            self.set_xy(42, 12)
        else:
            self.set_xy(15, 12)

        # Title text
        self.set_text_color(255, 255, 255)
        self.set_font("helvetica", "B", 16)
        self.cell(0, 8, safe_text(self.hospital_name), new_x="LMARGIN", new_y="NEXT", align="L")
        
        if self.logo_path and os.path.exists(self.logo_path):
            self.set_x(42)
        else:
            self.set_x(15)
        self.set_font("helvetica", "B", 9)
        self.set_text_color(173, 181, 189)
        self.cell(0, 4, "CLINICAL DECISION SUPPORT SYSTEM - ASSESSMENT REPORT", new_x="LMARGIN", new_y="NEXT", align="L")
        
        self.set_y(48)

    def footer(self):
        self.set_y(-25)
        # Decorative divider line
        self.set_draw_color(226, 232, 240)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(2)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(113, 128, 150)
        self.cell(0, 6, "Metropolitan Kidney Care Center | CDSS Automated Report", align="L")
        self.set_y(-19)
        self.cell(0, 6, f"Page {self.page_no()}", align="R")


def create_pdf_report(input_dict, label, conf, recommendations, patient_name="N/A"):
    """
    Create a PDF report using fpdf2 and return the binary output to be downloaded.
    All text is passed through safe_text(). Includes patient name, formatted grids,
    and professional header with logo.
    """
    # Define logo path in assets
    logo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "logo.png")
    
    pdf = PDFReport(logo_path=logo_path)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    # Date and Report Meta
    pdf.set_font("helvetica", "", 9)
    pdf.set_text_color(113, 128, 150)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 6, f"Report Generated: {current_date}", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 1. Patient Profile & Clinical Markers Grid
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 8, "1. Patient Profile & Clinical Indicators", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(226, 232, 240)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)

    # Build demographic and clinical values
    gender_val = input_dict.get("Gender", 0)
    gender_str = "Male" if gender_val == 1 else "Female"
    bmi_val    = input_dict.get("BMI", 0.0)
    sys_bp     = input_dict.get("Systolic_BP", "N/A")
    dia_bp     = input_dict.get("Diastolic_BP", "N/A")
    bp_str     = f"{sys_bp}/{dia_bp} mmHg" if sys_bp != "N/A" else "N/A"

    # Draw grid table
    pdf.set_font("helvetica", "B", 10)
    pdf.set_text_color(45, 55, 72)
    pdf.set_fill_color(247, 250, 252)

    # Helper function to print grid row
    def draw_row(label1, val1, label2, val2):
        pdf.set_font("helvetica", "B", 9)
        pdf.cell(45, 8, safe_text(label1), border=1, fill=True)
        pdf.set_font("helvetica", "", 9)
        pdf.cell(45, 8, safe_text(str(val1)), border=1)
        pdf.set_font("helvetica", "B", 9)
        pdf.cell(45, 8, safe_text(label2), border=1, fill=True)
        pdf.set_font("helvetica", "", 9)
        pdf.cell(45, 8, safe_text(str(val2)), border=1, new_x="LMARGIN", new_y="NEXT")

    draw_row("Patient Name", patient_name, "Date of Report", datetime.datetime.now().strftime("%Y-%m-%d"))
    draw_row("Age", f"{input_dict.get('Age')} years", "Gender", gender_str)
    draw_row("Body Mass Index (BMI)", f"{bmi_val:.1f} kg/m2", "Blood Pressure", bp_str)
    draw_row("Urine Albumin", f"{input_dict.get('Urine_Albumin', 0)} mg/L", "Urine Protein", f"{input_dict.get('Urine_Protein', 0)} mg/dL")
    draw_row("Albumin-Creatinine Ratio", f"{input_dict.get('Albumin_Creatinine_Ratio', 0)} mg/g", "Diabetes Mellitus", input_dict.get("Diabetes", "No"))
    draw_row("Hypertension History", input_dict.get("Hypertension", "No"), "Family History of CKD", input_dict.get("Family_History_Kidney", "No"))

    pdf.ln(6)

    # 2. Prediction Result Banner
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 8, "2. Diagnostic Risk Assessment", new_x="LMARGIN", new_y="NEXT")
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    # Color match the banner depending on prediction
    safe_label = safe_text(str(label))
    if "Healthy" in safe_label:
        pdf.set_fill_color(240, 253, 244)  # Light Green
        pdf.set_text_color(22, 101, 52)    # Dark Green
        border_color = (187, 247, 208)
    elif "Stage 1" in safe_label or "Stage 2" in safe_label or "Mild" in safe_label:
        pdf.set_fill_color(254, 252, 232)  # Light Yellow
        pdf.set_text_color(133, 77, 14)    # Dark Yellow
        border_color = (254, 240, 138)
    elif "Stage 3" in safe_label:
        pdf.set_fill_color(255, 247, 237)  # Light Orange
        pdf.set_text_color(194, 65, 12)    # Dark Orange
        border_color = (0xff, 0xdd, 0xb3)
    else:
        pdf.set_fill_color(254, 242, 242)  # Light Red
        pdf.set_text_color(153, 27, 27)    # Dark Red
        border_color = (254, 226, 226)

    # Custom banner cell with border
    pdf.set_draw_color(*border_color)
    res_text = f"PREDICTED CKD RISK STATUS: {safe_label.upper()}"
    if conf is not None:
        res_text += f"   |   MODEL CONFIDENCE: {conf:.1f}%"

    pdf.set_font("helvetica", "B", 11)
    pdf.cell(0, 14, res_text, border=1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # 3. Recommendations (Numbered Sections)
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 8, "3. Personalized Care Guidelines", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(226, 232, 240)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)

    for title, desc in recommendations:
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(43, 108, 176)
        pdf.cell(0, 6, safe_text(str(title)), new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("helvetica", "", 9.5)
        pdf.set_text_color(74, 85, 104)
        pdf.multi_cell(0, 5, safe_text(str(desc)), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3.5)

    # 4. Disclaimer Box
    pdf.ln(4)
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.set_font("helvetica", "I", 8.5)
    pdf.set_text_color(100, 110, 120)
    
    disclaimer_text = (
        "Medical Disclaimer: This report is generated by an automated clinical decision support system utilizing "
        "machine learning algorithms. It is intended for educational and informational purposes only. "
        "It DOES NOT constitute a medical diagnosis, prognosis, or treatment plan. Always consult a qualified "
        "nephrologist or medical practitioner for formal diagnosis, professional medical advice, and before "
        "making any healthcare decisions."
    )
    pdf.multi_cell(0, 4.5, safe_text(disclaimer_text), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())