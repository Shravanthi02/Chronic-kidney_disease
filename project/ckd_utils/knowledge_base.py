# ckd_utils/knowledge_base.py
# ─────────────────────────────────────────────────────────────────────────────
# Offline CKD Knowledge Base for the Medical Assistant Chatbot.
# Contains 110 kidney-related FAQs, symptoms, diet, tests, treatments, and app usage details.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import re
import math
from typing import Optional

# Define the 110 QA entries
_KNOWLEDGE_BASE: list[dict] = [
    # ── Category 1: CKD Basics & Stages (1-15) ──
    {
        "keywords": ["ckd", "chronic", "kidney", "disease", "definition", "overview"],
        "question": "What is Chronic Kidney Disease (CKD)?",
        "answer": (
            "**Chronic Kidney Disease (CKD)** is a condition characterized by the gradual loss of kidney function over time. "
            "The kidneys filter waste and excess fluids from your blood, which are then excreted in urine. When kidney disease "
            "reaches an advanced stage, dangerous levels of fluid, electrolytes, and wastes can build up in your body."
        )
    },
    {
        "keywords": ["stages", "stage", "classify", "classification", "five"],
        "question": "What are the stages of CKD?",
        "answer": (
            "CKD is classified into **five stages** based on eGFR: "
            "Stage 1 (eGFR ≥ 90, normal/high function with kidney damage); "
            "Stage 2 (eGFR 60-89, mild decrease); "
            "Stage 3 (eGFR 30-59, moderate decrease); "
            "Stage 4 (eGFR 15-29, severe decrease); "
            "Stage 5 (eGFR < 15, kidney failure/end-stage renal disease)."
        )
    },
    {
        "keywords": ["stage1", "early damage", "mild damage"],
        "question": "What is Stage 1 CKD?",
        "answer": (
            "**Stage 1 CKD** means you have mild kidney damage, but your eGFR is at a normal level of 90 or higher. "
            "There are usually no symptoms. Management focuses on controlling blood pressure, blood sugar, and identifying the underlying cause."
        )
    },
    {
        "keywords": ["stage2", "mild loss"],
        "question": "What is Stage 2 CKD?",
        "answer": (
            "**Stage 2 CKD** means your eGFR has slightly decreased to between 60 and 89. Like Stage 1, there are "
            "usually no symptoms. Lifestyle changes, such as healthy eating, regular exercise, and blood pressure control, are key to preventing progression."
        )
    },
    {
        "keywords": ["stage3", "stage3a", "stage3b", "moderate loss"],
        "question": "What is Stage 3 CKD?",
        "answer": (
            "**Stage 3 CKD** is divided into 3a (eGFR 45-59) and 3b (eGFR 30-44). Kidney function is moderately reduced. "
            "You may begin to experience symptoms like swelling (edema), fatigue, changes in urination, or back pain. Consult a nephrologist."
        )
    },
    {
        "keywords": ["stage4", "severe loss", "pre-dialysis"],
        "question": "What is Stage 4 CKD?",
        "answer": (
            "**Stage 4 CKD** indicates severely reduced kidney function (eGFR 15-29). It is the final stage before kidney failure. "
            "Symptoms like anemia, bone disease, and fluid retention are common. Treatment focuses on preparing for dialysis or a kidney transplant."
        )
    },
    {
        "keywords": ["stage5", "failure", "esrd", "end stage"],
        "question": "What is Stage 5 CKD?",
        "answer": (
            "**Stage 5 CKD** is kidney failure (eGFR < 15), also called End-Stage Renal Disease (ESRD). "
            "The kidneys have lost nearly all filtering ability. Treatment options are dialysis or a kidney transplant to sustain life."
        )
    },
    {
        "keywords": ["egfr normal", "normal egfr", "healthy egfr"],
        "question": "What is a normal eGFR level?",
        "answer": (
            "A normal eGFR is **90 mL/min/1.73m² or higher**. An eGFR below 60 for three months or more "
            "indicates chronic kidney disease."
        )
    },
    {
        "keywords": ["reversibility", "cure ckd", "reversible ckd"],
        "question": "Can chronic kidney disease be cured or reversed?",
        "answer": (
            "CKD is generally **irreversible**, meaning damaged kidney tissue cannot be cured. However, "
            "early detection and proper management can slow down or halt its progression, preventing kidney failure."
        )
    },
    {
        "keywords": ["silent killer", "no symptoms early"],
        "question": "Why is CKD called a 'silent' disease?",
        "answer": (
            "CKD is called 'silent' because in its early stages (1 and 2), it rarely causes noticeable symptoms. "
            "Many people lose up to 90% of their kidney function before experiencing any symptoms."
        )
    },
    {
        "keywords": ["progression rate", "how fast progress"],
        "question": "How fast does CKD progress?",
        "answer": (
            "The rate of progression varies widely. For some, it takes decades; for others, it is much faster. "
            "Progression depends on factors like blood pressure control, diabetes management, diet, genetics, and lifestyle."
        )
    },
    {
        "keywords": ["gfr age", "egfr decline age"],
        "question": "Does kidney function naturally decline with age?",
        "answer": (
            "Yes, kidney function naturally declines as you age, usually starting around age 30-40. "
            "However, this normal age-related decline is much slower than the decline caused by chronic kidney disease."
        )
    },
    {
        "keywords": ["kidney function percent", "stage percentage"],
        "question": "How does eGFR relate to kidney function percentage?",
        "answer": (
            "As a general rule of thumb, eGFR is roughly equivalent to your remaining kidney function percentage. "
            "For example, an eGFR of 50 indicates approximately 50% kidney function."
        )
    },
    {
        "keywords": ["chronic vs acute", "aki vs ckd"],
        "question": "What is the difference between AKI and CKD?",
        "answer": (
            "**Acute Kidney Injury (AKI)** is a sudden, temporary episode of kidney failure that happens within hours or days. "
            "**Chronic Kidney Disease (CKD)** is a slow, progressive, and permanent decline of function over months or years."
        )
    },
    {
        "keywords": ["kidney location", "kidney function role"],
        "question": "What do the kidneys do in the body?",
        "answer": (
            "The kidneys filter waste products and extra water from the blood, regulate blood pressure, balance electrolytes "
            "(sodium, potassium, calcium), produce hormones for red blood cells (EPO), and activate Vitamin D for bone health."
        )
    },

    # ── Category 2: Symptoms & Signs (16-30) ──
    {
        "keywords": ["symptoms", "signs", "indications", "manifestations"],
        "question": "What are the common symptoms of CKD?",
        "answer": (
            "Common symptoms of advanced CKD include fatigue, swollen ankles or feet (edema), puffiness around the eyes, "
            "foamy or bubbly urine, difficulty sleeping, itchy skin, nausea, loss of appetite, muscle cramps, and shortness of breath."
        )
    },
    {
        "keywords": ["edema", "swollen feet", "swelling ankles"],
        "question": "Why do my legs and ankles swell in CKD?",
        "answer": (
            "Swelling (edema) occurs because damaged kidneys cannot filter out excess sodium and fluid. "
            "This fluid accumulates in tissues, most commonly in the lower extremities (legs, ankles, feet) due to gravity."
        )
    },
    {
        "keywords": ["foamy urine", "bubbles urine", "proteinuria bubbles"],
        "question": "Why is my urine foamy or bubbly?",
        "answer": (
            "Foamy or frothy urine is a sign of **proteinuria** (protein leaking into the urine). "
            "Healthy kidneys keep proteins in the blood. If the kidney filters are damaged, proteins like albumin leak into the urine."
        )
    },
    {
        "keywords": ["fatigue symptoms", "tired ckd", "weakness ckd"],
        "question": "Why does CKD cause extreme fatigue?",
        "answer": (
            "Fatigue in CKD is primarily caused by **anemia** (lack of red blood cells due to low EPO hormone production) "
            "and the buildup of toxic waste products in the blood (uremia)."
        )
    },
    {
        "keywords": ["itchy skin", "pruritus skin"],
        "question": "Why does kidney disease cause itchy skin?",
        "answer": (
            "Severe itching (pruritus) is caused by high levels of phosphorus in the blood (hyperphosphatemia) "
            "and the accumulation of uremic waste products in the skin tissue."
        )
    },
    {
        "keywords": ["taste changes", "metallic taste", "ammonia breath"],
        "question": "Why do I have a metallic taste in my mouth?",
        "answer": (
            "A metallic taste or ammonia-like breath (uremic fetor) is caused by the buildup of urea and other "
            "nitrogenous wastes in the blood, which breakdown into ammonia in saliva."
        )
    },
    {
        "keywords": ["nausea vomiting", "sick stomach"],
        "question": "Does CKD cause nausea and vomiting?",
        "answer": (
            "Yes, advanced CKD causes nausea and vomiting due to **uremia** (wastes accumulating in the body). "
            "This waste buildup irritates the digestive system and brain receptors."
        )
    },
    {
        "keywords": ["shortness of breath", "breathless fluid"],
        "question": "Why do I feel short of breath?",
        "answer": (
            "Shortness of breath in CKD can be caused by fluid building up in the lungs (pulmonary edema) "
            "or severe anemia (lack of oxygen-carrying red blood cells)."
        )
    },
    {
        "keywords": ["muscle cramps", "cramping legs"],
        "question": "Why do I get muscle cramps?",
        "answer": (
            "Muscle cramps, especially in the legs, are common in CKD due to electrolyte imbalances "
            "(low calcium, high phosphorus, or potassium shifts) and impaired nerve function."
        )
    },
    {
        "keywords": ["frequent urination", "nocturia night"],
        "question": "Why do I urinate more often at night?",
        "answer": (
            "Urinating more frequently, especially at night (nocturia), occurs because the kidneys lose their ability "
            "to concentrate urine, requiring more water to excrete the daily waste load."
        )
    },
    {
        "keywords": ["blood in urine", "hematuria color"],
        "question": "Why is there blood in my urine?",
        "answer": (
            "Blood in the urine (hematuria) occurs when the kidney filters (glomeruli) are damaged and leak red blood cells. "
            "It can also indicate kidney stones, infections, or cysts. See a doctor immediately."
        )
    },
    {
        "keywords": ["sleep issues", "insomnia sleep"],
        "question": "How does CKD affect sleep?",
        "answer": (
            "CKD causes sleep issues like insomnia, restless legs syndrome, and sleep apnea. "
            "These are triggered by uremic toxins, electrolyte shifts, and anxiety."
        )
    },
    {
        "keywords": ["dry skin", "flaky skin"],
        "question": "Why is my skin so dry in CKD?",
        "answer": (
            "Kidneys help balance minerals in the body. When they fail, mineral imbalances, dehydration, "
            "and decreased sweat gland activity lead to dry, flaky, and easily irritated skin."
        )
    },
    {
        "keywords": ["puffy eyes", "eye swelling"],
        "question": "Why are my eyes puffy in the morning?",
        "answer": (
            "Puffiness around the eyes is caused by large amounts of protein leaking into the urine, "
            "which reduces fluid retention capacity in blood vessels and leads to fluid pooling around tissues."
        )
    },
    {
        "keywords": ["brain fog", "confusion ckd"],
        "question": "Can CKD cause brain fog or confusion?",
        "answer": (
            "Yes, the accumulation of uremic toxins in the blood can affect the central nervous system, "
            "leading to difficulty concentrating, forgetfulness, brain fog, and in severe cases, confusion."
        )
    },

    # ── Category 3: Causes & Risk Factors (31-45) ──
    {
        "keywords": ["causes of ckd", "why ckd occurs", "underlying cause"],
        "question": "What are the primary causes of CKD?",
        "answer": (
            "The two leading causes of CKD are **Diabetes** (high blood sugar damages filters) and "
            "**Hypertension** (high blood pressure damages vessels). Other causes include glomerulonephritis, polycystic kidney disease, and autoimmune conditions."
        )
    },
    {
        "keywords": ["diabetes ckd", "diabetic nephropathy glucose"],
        "question": "How does diabetes damage the kidneys?",
        "answer": (
            "Chronically high blood sugar levels force the kidneys to filter too much blood, stressing the glomeruli. "
            "Over time, this extra work damages the filters, causing them to leak protein and lose function (diabetic nephropathy)."
        )
    },
    {
        "keywords": ["hypertension ckd", "high blood pressure damage"],
        "question": "How does high blood pressure damage kidneys?",
        "answer": (
            "High blood pressure stretches and damages the blood vessels throughout the body, including the delicate "
            "vessels in the kidneys. Damaged vessels can no longer deliver enough blood to kidney tissue, leading to scarring."
        )
    },
    {
        "keywords": ["risk factors ckd", "who is at risk"],
        "question": "What are the risk factors for CKD?",
        "answer": (
            "Key risk factors include having diabetes, high blood pressure, heart disease, a family history of kidney failure, "
            "being over 60 years old, obesity, smoking, and recurrent kidney infections."
        )
    },
    {
        "keywords": ["family history genetic", "hereditary ckd"],
        "question": "Does family history increase the risk of CKD?",
        "answer": (
            "Yes. If a close family member has kidney disease, you are at a significantly higher risk. "
            "This can be due to inherited genetic disorders (like Polycystic Kidney Disease) or shared genetic risk for diabetes and hypertension."
        )
    },
    {
        "keywords": ["obesity risk", "overweight kidney"],
        "question": "Does obesity increase the risk of kidney disease?",
        "answer": (
            "Yes, obesity increases the risk because it forces the kidneys to filter more blood to meet metabolic demands (hyperfiltration). "
            "It also increases the risk of developing diabetes and hypertension, the primary causes of CKD."
        )
    },
    {
        "keywords": ["glomerulonephritis cause", "inflammation glomeruli"],
        "question": "What is Glomerulonephritis?",
        "answer": (
            "It is an inflammation of the **glomeruli** (the tiny filtering units of the kidneys). "
            "It can be caused by immune system responses, infections, or toxic drugs, leading to blood and protein in the urine."
        )
    },
    {
        "keywords": ["polycystic kidney", "pkd cysts"],
        "question": "What is Polycystic Kidney Disease (PKD)?",
        "answer": (
            "PKD is a genetic disorder that causes numerous fluid-filled cysts to grow in the kidneys. "
            "These cysts enlarge over time, compressing and destroying the surrounding healthy kidney tissue."
        )
    },
    {
        "keywords": ["nsaid damage", "painkillers ibuprofen"],
        "question": "Can regular use of painkillers cause kidney damage?",
        "answer": (
            "Yes, long-term or high-dose use of Over-The-Counter painkillers, especially NSAIDs (like ibuprofen, naproxen), "
            "reduces blood flow to the kidneys and can lead to chronic kidney damage or acute kidney injury."
        )
    },
    {
        "keywords": ["lupus nephritis", "autoimmune kidney"],
        "question": "What is Lupus Nephritis?",
        "answer": (
            "Lupus nephritis is a kidney complication of Systemic Lupus Erythematosus (SLE), an autoimmune disease. "
            "The body's immune system attacks the kidneys, causing inflammation, swelling, and scarring of the filtering units."
        )
    },
    {
        "keywords": ["kidney stones damage", "obstruction stones"],
        "question": "Can kidney stones lead to CKD?",
        "answer": (
            "Yes, if kidney stones recur frequently and cause chronic urinary tract blockages (obstruction), "
            "the backpressure of urine can damage and scar the kidneys over time, contributing to CKD."
        )
    },
    {
        "keywords": ["recurrent uti", "kidney infection scarring"],
        "question": "Can urinary tract infections (UTIs) cause CKD?",
        "answer": (
            "A simple bladder infection doesn't cause kidney damage. However, recurrent, severe kidney infections "
            "(pyelonephritis) can cause permanent kidney scarring, leading to CKD."
        )
    },
    {
        "keywords": ["heavy metals toxin", "lead cadmium kidney"],
        "question": "Can environmental toxins cause CKD?",
        "answer": (
            "Yes, chronic exposure to heavy metals (like lead, cadmium, mercury) and certain industrial solvents "
            "is nephrotoxic and can lead to chronic interstitial nephritis and kidney damage."
        )
    },
    {
        "keywords": ["heart disease connection", "cardiovascular ckd"],
        "question": "How are heart disease and kidney disease connected?",
        "answer": (
            "They share risk factors like diabetes and high blood pressure. Additionally, damaged kidneys increase "
            "cardiovascular strain, and a failing heart reduces blood flow to the kidneys, compounding damage to both organs."
        )
    },
    {
        "keywords": ["smoking damage", "nicotine kidney"],
        "question": "Does smoking affect the kidneys?",
        "answer": (
            "Yes, smoking slows blood flow to important organs, including the kidneys. It also worsens high blood pressure "
            "and increases the risk of kidney cancer and diabetic nephropathy."
        )
    },

    # ── Category 4: Prevention & Lifestyle (46-60) ──
    {
        "keywords": ["prevention", "prevent ckd", "stop progression"],
        "question": "How can I prevent Chronic Kidney Disease?",
        "answer": (
            "Prevent CKD by managing diabetes and high blood pressure, eating a balanced diet low in sodium, exercising regularly, "
            "maintaining a healthy weight, quitting smoking, staying hydrated, and avoiding overuse of NSAIDs."
        )
    },
    {
        "keywords": ["hydration amount", "water kidney protection"],
        "question": "How does hydration protect the kidneys?",
        "answer": (
            "Drinking water helps the kidneys clear sodium, urea, and toxins from the blood. It also prevents kidney stones "
            "and UTIs. Aim for 8 glasses a day, unless you have advanced CKD requiring fluid restriction."
        )
    },
    {
        "keywords": ["bp control target", "target blood pressure"],
        "question": "What is the recommended blood pressure target for CKD?",
        "answer": (
            "For most people with CKD, the target blood pressure is **less than 130/80 mmHg** to slow kidney damage. "
            "Consult your doctor for your specific targets."
        )
    },
    {
        "keywords": ["quit smoking benefits", "smoking cessation"],
        "question": "Does quitting smoking help my kidneys?",
        "answer": (
            "Yes. Quitting smoking improves overall blood flow, lowers blood pressure, and significantly reduces the "
            "rate of decline in kidney function, especially in people with diabetes."
        )
    },
    {
        "keywords": ["weight loss benefit", "lose weight kidneys"],
        "question": "How does losing weight help my kidneys?",
        "answer": (
            "Losing weight reduces the workload (hyperfiltration) on the kidneys, helps lower blood pressure, "
            "improves blood sugar control, and reduces the risk of obesity-related glomerulopathy."
        )
    },
    {
        "keywords": ["sleep duration", "sleep hygiene kidneys"],
        "question": "Does sleep affect kidney health?",
        "answer": (
            "Yes, getting 7-8 hours of quality sleep nightly is important. Kidney function is regulated by the sleep-wake cycle, "
            "and chronic sleep deprivation is linked to faster kidney function decline."
        )
    },
    {
        "keywords": ["stress management", "meditation kidneys"],
        "question": "Can stress affect kidney function?",
        "answer": (
            "Yes, chronic stress increases blood pressure and blood sugar levels, both of which strain the kidneys. "
            "Mindfulness, meditation, and exercise help manage stress and protect kidneys."
        )
    },
    {
        "keywords": ["alcohol limit", "drinking kidneys"],
        "question": "How much alcohol is safe for kidneys?",
        "answer": (
            "Heavy drinking (more than 2 drinks/day for men, 1 for women) can cause high blood pressure and dehydration, "
            "harming the kidneys. Moderate drinking is generally safe if your doctor approves."
        )
    },
    {
        "keywords": ["annual screening", "how often test ckd"],
        "question": "Who should be screened annually for CKD?",
        "answer": (
            "Anyone with diabetes, high blood pressure, heart disease, or a family history of kidney failure "
            "should get annual blood (eGFR) and urine (ACR) tests to monitor kidney health."
        )
    },
    {
        "keywords": ["herbal remedies risk", "natural supplements danger"],
        "question": "Are natural or herbal supplements safe for kidneys?",
        "answer": (
            "Not necessarily. Some herbs are toxic to the kidneys (like aristolochic acid) or interact with medications. "
            "Always consult your doctor before starting any herbal supplement."
        )
    },
    {
        "keywords": ["flu vaccine ckd", "immunization ckd"],
        "question": "Should CKD patients get a flu vaccine?",
        "answer": (
            "Yes, people with CKD have a weakened immune system and are at higher risk for complications from the flu. "
            "An annual flu shot is strongly recommended."
        )
    },
    {
        "keywords": ["nsaid alternative", "tylenol acetaminophen"],
        "question": "What is a safe alternative to NSAIDs for pain relief?",
        "answer": (
            "**Acetaminophen (Tylenol)** is generally considered safer for the kidneys than NSAIDs (like ibuprofen). "
            "However, use it in moderation and discuss chronic pain management with your doctor."
        )
    },
    {
        "keywords": ["contrast dye warning", "ct scan precautions"],
        "question": "Are medical imaging contrast dyes safe for kidneys?",
        "answer": (
            "Contrast dyes (used in CT scans or MRIs) can cause acute kidney injury in patients with existing CKD. "
            "Ensure your doctor knows you have CKD so they can take preventative measures (like IV fluids)."
        )
    },
    {
        "keywords": ["dental health kidneys", "gum disease ckd"],
        "question": "Is dental health related to kidney health?",
        "answer": (
            "Yes, chronic gum disease (periodontitis) is a source of chronic inflammation, which is linked to "
            "heart disease and faster kidney function decline in people with CKD."
        )
    },
    {
        "keywords": ["salt substitutes warning", "potassium salt substitute"],
        "question": "Are potassium-based salt substitutes safe for CKD?",
        "answer": (
            "**No**, potassium-based salt substitutes (like NoSalt) are dangerous for people with moderate to advanced CKD "
            "because they can lead to life-threatening high potassium levels (hyperkalemia)."
        )
    },

    # ── Category 5: Diet Suggestions (61-77) ──
    {
        "keywords": ["diet", "nutrition", "renal diet", "foods to avoid", "kidney diet"],
        "question": "What is a renal (kidney-friendly) diet?",
        "answer": (
            "A renal diet is low in sodium, phosphorus, and potassium, and has controlled protein. "
            "It aims to reduce the buildup of waste products in the blood, lower blood pressure, and minimize kidney strain."
        )
    },
    {
        "keywords": ["avoid potassium", "potassium foods list"],
        "question": "Which high-potassium foods should I limit?",
        "answer": (
            "Limit high-potassium foods like bananas, oranges, potatoes, tomatoes, spinach, avocados, pumpkins, "
            "dried fruits, and dairy products if your blood potassium levels are elevated."
        )
    },
    {
        "keywords": ["avoid phosphorus", "phosphorus foods list"],
        "question": "Which high-phosphorus foods should I limit?",
        "answer": (
            "Limit dark colas, processed foods (which contain phosphate additives), dairy products (milk, cheese, yogurt), "
            "nuts, seeds, beans, and beer, as excess phosphorus weakens bones."
        )
    },
    {
        "keywords": ["sodium limit diet", "salt restriction mg"],
        "question": "How much sodium should I consume daily?",
        "answer": (
            "People with CKD should limit sodium to **less than 2,000 mg per day** (about 1 teaspoon of salt) "
            "to help control blood pressure and reduce fluid retention."
        )
    },
    {
        "keywords": ["low protein ckd", "protein restriction stages"],
        "question": "Why should I limit protein in early stages?",
        "answer": (
            "Limiting protein intake (0.6-0.8 g/kg of body weight) in Stages 3-4 reduces the build-up of nitrogenous wastes "
            "in the blood and lowers the filtering workload of the kidneys, slowing progression."
        )
    },
    {
        "keywords": ["dialysis protein", "high protein dialysis"],
        "question": "Why do dialysis patients need MORE protein?",
        "answer": (
            "Dialysis removes protein waste as well as healthy protein from the blood. Therefore, patients on dialysis "
            "need a high-protein diet (1.2 g/kg) to maintain muscle mass and strength."
        )
    },
    {
        "keywords": ["low potassium fruits", "safe fruits ckd"],
        "question": "What are some safe, low-potassium fruits?",
        "answer": (
            "Safe, low-potassium fruits include apples, berries (strawberries, blueberries, raspberries), "
            "grapes, pineapples, cranberries, plums, and pears."
        )
    },
    {
        "keywords": ["low potassium vegetables", "safe veggies ckd"],
        "question": "What are some safe, low-potassium vegetables?",
        "answer": (
            "Safe vegetables include cauliflower, cabbage, onions, garlic, cucumbers, radishes, celery, "
            "green beans, carrots, and bell peppers."
        )
    },
    {
        "keywords": ["healthy fats renal", "olive oil ckd"],
        "question": "What fats are recommended in a renal diet?",
        "answer": (
            "Healthy fats like **olive oil** and canola oil are highly recommended. Olive oil is rich in anti-inflammatory "
            "monounsaturated fatty acids and is low in sodium, potassium, and phosphorus."
        )
    },
    {
        "keywords": ["whole grains phosphorus", "white vs brown rice"],
        "question": "Should I choose white or brown grains in CKD?",
        "answer": (
            "In advanced CKD (Stages 4-5), white rice, white bread, and pasta are often preferred over brown rice and "
            "whole wheat because they contain significantly less phosphorus and potassium."
        )
    },
    {
        "keywords": ["meat alternatives", "plant protein kidneys"],
        "question": "Are plant proteins better for the kidneys than animal proteins?",
        "answer": (
            "Yes, research suggests plant proteins (tofu, beans in moderation) produce less acid and have lower bioavailability "
            "of phosphorus, putting less strain on the kidneys compared to red meat."
        )
    },
    {
        "keywords": ["processed meat danger", "sodium phosphate meat"],
        "question": "Why are processed meats particularly bad for CKD?",
        "answer": (
            "Processed meats (bacon, sausage, hot dogs) are extremely high in sodium and contain chemical phosphorus additives "
            "(sodium phosphate), which are absorbed 100% by the body."
        )
    },
    {
        "keywords": ["canned foods sodium", "rinse canned beans"],
        "question": "How can I reduce sodium in canned foods?",
        "answer": (
            "Choose 'no salt added' varieties. For canned beans or vegetables, draining and thoroughly rinsing them "
            "with water can remove up to 40% of the sodium."
        )
    },
    {
        "keywords": ["egg whites phosphorus", "egg yolks limit"],
        "question": "Are eggs good for kidney disease patients?",
        "answer": (
            "**Egg whites** are an excellent source of high-quality protein that is low in phosphorus. "
            "However, limit egg yolks, as they contain high amounts of phosphorus."
        )
    },
    {
        "keywords": ["fluid restriction tips", "thirst management"],
        "question": "How can I manage my fluid intake restrictions?",
        "answer": (
            "Manage thirst by chewing sugarless gum, rinsing your mouth with cold water without swallowing, "
            "sucking on ice chips (count as fluid), using small cups, and limiting sodium."
        )
    },
    {
        "keywords": ["foods avoid", "avoid foods", "avoid eating", "restrict", "bad foods", "limit foods", "patients avoid"],
        "question": "What foods should I avoid with CKD?",
        "answer": (
            "Limit high-sodium foods (processed meats, canned soups, fast food), high-potassium foods (bananas, oranges, potatoes, tomatoes), "
            "and high-phosphorus foods (dairy, dark colas, beer) to prevent waste buildup and protect kidney function."
        )
    },
    {
        "keywords": ["good foods", "foods eat", "eat what", "allowed foods", "kidney friendly food"],
        "question": "What foods are good for the kidneys?",
        "answer": (
            "Kidney-friendly foods include apples, berries (strawberries, blueberries), grapes, cauliflower, cabbage, "
            "garlic, olive oil, egg whites, and lean chicken breast. These are low in sodium, potassium, and phosphorus."
        )
    },

    # ── Category 6: Exercise Recommendations (78-85) ──
    {
        "keywords": ["exercise", "workout", "fitness", "walking", "activity"],
        "question": "Is exercise safe for CKD patients?",
        "answer": (
            "Yes, regular exercise is safe and highly recommended for CKD patients at all stages. "
            "It helps lower blood pressure, improve energy, manage weight, and control blood sugar."
        )
    },
    {
        "keywords": ["aerobic exercise ckd", "cardio kidneys"],
        "question": "What is the best type of exercise for CKD?",
        "answer": (
            "Low-impact aerobic exercises like walking, cycling (stationary or outdoor), swimming, "
            "and water aerobics are excellent for building cardiovascular health without straining the body."
        )
    },
    {
        "keywords": ["exercise frequency", "how often work out"],
        "question": "How often should a CKD patient exercise?",
        "answer": (
            "Aim for at least **30 minutes of moderate activity** (like brisk walking) "
            "on most days of the week, totaling 150 minutes weekly. You can break it into 10-minute sessions."
        )
    },
    {
        "keywords": ["strength training ckd", "lifting weights kidneys"],
        "question": "Is light resistance training safe?",
        "answer": (
            "Yes, light strength training using resistance bands or light weights 2-3 times a week helps "
            "prevent muscle wasting, which is common in chronic kidney disease."
        )
    },
    {
        "keywords": ["exercise warning signs", "stop exercising immediately"],
        "question": "When should I stop exercising?",
        "answer": (
            "Stop exercising immediately if you feel dizzy, short of breath, experience chest pain, "
            "an irregular heartbeat, muscle weakness, or extreme fatigue."
        )
    },
    {
        "keywords": ["yoga kidney health", "stretching flexibility"],
        "question": "Can I do yoga or stretching exercises?",
        "answer": (
            "Yes, yoga, Tai Chi, and stretching exercises are excellent for improving flexibility, strength, "
            "balance, and reducing stress, which benefits blood pressure control."
        )
    },
    {
        "keywords": ["exercise during dialysis", "dialysis workout"],
        "question": "Can you exercise while on dialysis?",
        "answer": (
            "Yes, many centers encourage stationary cycling during the first few hours of hemodialysis treatment, "
            "which improves blood flow and makes waste clearance more efficient."
        )
    },
    {
        "keywords": ["hydration exercise", "water during workout"],
        "question": "How should I hydrate during exercise?",
        "answer": (
            "Hydrate moderately. If you are on a fluid restriction, you must measure your exercise water intake "
            "within your daily limit. Otherwise, drink a small cup of water every 15 minutes of exercise."
        )
    },

    # ── Category 7: Medication & Precautions (86-94) ──
    {
        "keywords": ["medications", "drugs", "nsaids", "painkillers", "avoid meds"],
        "question": "What medication precautions should I take in CKD?",
        "answer": (
            "Avoid NSAIDs (like ibuprofen), read all labels for sodium/potassium, tell your doctor about all supplements, "
            "and ensure your doctor adjusts prescription dosages based on your current eGFR."
        )
    },
    {
        "keywords": ["ace inhibitors protective", "lisinopril protective"],
        "question": "Why are ACE inhibitors/ARBs prescribed in CKD?",
        "answer": (
            "ACE inhibitors (e.g., Lisinopril) and ARBs (e.g., Losartan) lower blood pressure and expand blood vessels "
            "in the kidneys, reducing pressure inside the filters and lowering protein leakage, which protects the kidneys."
        )
    },
    {
        "keywords": ["phosphate binders meals", "calcium acetate binders"],
        "question": "What are phosphate binders?",
        "answer": (
            "Phosphate binders are medications taken with meals or snacks. They bind to phosphorus in your food, "
            "preventing it from being absorbed into your bloodstream, thus protecting your bones and blood vessels."
        )
    },
    {
        "keywords": ["diuretics edema", "lasix furosemide"],
        "question": "Why are water pills (diuretics) prescribed?",
        "answer": (
            "Diuretics (like furosemide/Lasix) help the kidneys excrete extra sodium and water, reducing "
            "fluid retention (swelling) in the legs, lungs, and lowering blood pressure."
        )
    },
    {
        "keywords": ["epo injections anemia", "procrit aranesp"],
        "question": "How is anemia treated in CKD?",
        "answer": (
            "Anemia is treated with Erythropoietin-Stimulating Agents (ESAs) like Procrit or Aranesp, which are "
            "injections that mimic the natural hormone EPO to stimulate red blood cell production, along with iron supplements."
        )
    },
    {
        "keywords": ["avoid magnesium", "laxatives antacids warning"],
        "question": "Why should I avoid magnesium antacids?",
        "answer": (
            "Damaged kidneys cannot clear magnesium efficiently. Taking magnesium-based antacids (like Maalox) "
            "or laxatives can build up to toxic levels, causing muscle weakness and heart issues."
        )
    },
    {
        "keywords": ["metformin ckd", "diabetes drug gfr"],
        "question": "Is Metformin safe for patients with CKD?",
        "answer": (
            "Metformin is safe in early CKD but must be discontinued or dose-reduced if eGFR falls below 45 mL/min, "
            "and stopped completely if eGFR is below 30, due to the risk of lactic acidosis."
        )
    },
    {
        "keywords": ["dialysis", "hemodialysis", "peritoneal dialysis", "treatment"],
        "question": "What is dialysis?",
        "answer": (
            "**Dialysis** is a treatment that filters and purifies the blood using a machine (hemodialysis) "
            "or the lining of your abdomen (peritoneal dialysis). It is needed when kidneys fail (Stage 5 CKD) "
            "and can no longer perform their essential cleaning functions."
        )
    },
    {
        "keywords": ["transplant", "organ", "donor", "kidney transplant", "surgery"],
        "question": "What is a kidney transplant?",
        "answer": (
            "A **kidney transplant** is a surgical procedure to place a healthy kidney from a living or deceased donor "
            "into a person whose kidneys no longer function properly. It is the preferred treatment for kidney failure."
        )
    },

    # ── Category 8: Blood & Urine Tests (95-104) ──
    {
        "keywords": ["bun", "blood urea nitrogen", "urea levels"],
        "question": "What is Blood Urea Nitrogen (BUN)?",
        "answer": (
            "BUN measures the amount of nitrogen in your blood that comes from the waste product urea (created by protein breakdown). "
            "High levels indicate that the kidneys are not filtering waste effectively."
        )
    },
    {
        "keywords": ["egfr test", "glomerular filtration test"],
        "question": "What is eGFR?",
        "answer": (
            "eGFR (estimated Glomerular Filtration Rate) measures how well your kidneys filter waste. It is calculated "
            "from a blood creatinine test, age, and gender. Lower numbers mean reduced kidney function."
        )
    },
    {
        "keywords": ["creatinine test", "serum creatinine waste"],
        "question": "What is Serum Creatinine?",
        "answer": (
            "Creatinine is a waste product from muscle breakdown. Healthy kidneys filter it out of the blood. "
            "If your kidneys are damaged, creatinine builds up in the blood, causing serum creatinine levels to rise."
        )
    },
    {
        "keywords": ["acr test", "albumin creatinine ratio"],
        "question": "What is the Urine Albumin-Creatinine Ratio (ACR)?",
        "answer": (
            "Urine ACR measures the amount of albumin (protein) in your urine compared to creatinine. "
            "An ACR over 30 mg/g is an early sign of kidney damage (microalbuminuria)."
        )
    },
    {
        "keywords": ["urinalysis test", "dipstick protein"],
        "question": "What is a urinalysis?",
        "answer": (
            "A urinalysis is a urine test that checks for protein, blood, sugar, specific gravity, and signs of infection. "
            "It helps detect kidney disease, diabetes, or UTIs."
        )
    },
    {
        "keywords": ["bicarbonate levels", "co2 blood test"],
        "question": "What does bicarbonate measure in kidney tests?",
        "answer": (
            "Bicarbonate (or CO2) measures the acid-base balance in your blood. Kidneys balance acid. "
            "Low bicarbonate levels (< 22 mEq/L) indicate metabolic acidosis, common in advanced CKD."
        )
    },
    {
        "keywords": ["potassium blood test", "hyperkalemia test"],
        "question": "What is hyperkalemia?",
        "answer": (
            "Hyperkalemia is high potassium in the blood. Healthy kidneys filter potassium. When they fail, "
            "high potassium can build up, risking dangerous heart rhythm problems."
        )
    },
    {
        "keywords": ["calcium test", "phosphorus test"],
        "question": "What are calcium and phosphorus blood tests?",
        "answer": (
            "These tests check mineral balance. Advanced CKD causes high phosphorus and low calcium, "
            "which signals the body to pull calcium from bones, weakening them."
        )
    },
    {
        "keywords": ["hemoglobin test", "anemia test hgb"],
        "question": "What is the Hemoglobin (Hgb) blood test?",
        "answer": (
            "Hemoglobin measures the oxygen-carrying protein in red blood cells. Low levels indicate anemia, "
            "which is common in CKD due to reduced EPO hormone production by the kidneys."
        )
    },
    {
        "keywords": ["bmi", "body mass index", "obesity", "overweight"],
        "question": "What is BMI?",
        "answer": (
            "**BMI (Body Mass Index)** is calculated as weight (kg) divided by height squared (m²). "
            "A high BMI (obesity) is a risk factor for kidney disease as it causes the kidneys to filter "
            "more blood, increasing glomerular pressure and risk of CKD."
        )
    },

    # ── Category 9: Prediction System & FAQs (105-110) ──
    {
        "keywords": ["how use prediction", "run prediction"],
        "question": "How do I run a prediction?",
        "answer": (
            "Go to the **🔬 Prediction** tab in the sidebar, input all required patient data (age, BP, eGFR, creatinine, BUN, etc.), "
            "and click the **Run Risk Assessment** button to get the predicted CKD stage and confidence."
        )
    },
    {
        "keywords": ["explain prediction", "prediction explanation"],
        "question": "How do I get the chatbot to explain my prediction?",
        "answer": (
            "Run a prediction first. Then come to this page and ask: **'Explain my prediction'**. "
            "The chatbot will read your last prediction result and explain the stage and main contributing factors."
        )
    },
    {
        "keywords": ["how save patient", "add patient profile"],
        "question": "How do I save a patient profile?",
        "answer": (
            "Go to the **👤 Patients** page, select **Add / Edit Patient**, fill in their name, age, "
            "demographic details, height, and weight, and click **Save Patient Profile**."
        )
    },
    {
        "keywords": ["how schedule appointment", "reminders scheduling"],
        "question": "How do I schedule an appointment?",
        "answer": (
            "Go to the **📅 Reminders** page, select a patient, select the date and time of the appointment, "
            "write notes, and click **Confirm Schedule** to save the reminder."
        )
    },
    {
        "keywords": ["view history", "prediction history"],
        "question": "How do I see past prediction reports?",
        "answer": (
            "Go to the **📋 History** page to view a table of all past predictions, including inputs, "
            "predicted CKD stage, confidence scores, and timestamps."
        )
    },
    {
        "keywords": ["pdf download", "generate pdf report"],
        "question": "How do I download a PDF report?",
        "answer": (
            "On the **🔬 Prediction** page, after running a prediction, scroll to the bottom. Click **Generate PDF Report**, "
            "then click the download button to save the detailed PDF report."
        )
    }
]

class KnowledgeBase:
    """
    Offline CKD Knowledge Base.
    Features rule-based routing fallback, then keyword overlap matching over 110 specific entries.
    """

    # Stop words list (words that don't help disambiguate)
    _STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
        "this", "that", "these", "those", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "have", "has", "had",
        "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "up", "about", "than", "so", "no", "not", "as", "tell",
        "please", "explain", "describe", "know", "like", "want", "should",
        "how", "what", "why", "where", "when", "who", "does", "do", "some", "many",
        "ckd", "kidney", "renal", "disease", "patient", "patients"
    }

    def __init__(self) -> None:
        self._entries = _KNOWLEDGE_BASE
        self._index: list[tuple[set[str], dict]] = []
        self._build_index()

    def _build_index(self) -> None:
        for entry in self._entries:
            kw_set = set()
            # Add explicit keywords
            for kw in entry["keywords"]:
                for word in re.sub(r"[^\w\s]", " ", kw.lower()).split():
                    kw_set.add(word)
            
            # Also add tokens from the question for better semantic/keyword matching
            q_tokens = self._tokenize(entry["question"])
            kw_set.update(q_tokens)
            
            self._index.append((kw_set, entry))

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = set(text.split())
        tokens -= KnowledgeBase._STOP_WORDS
        tokens = {t for t in tokens if len(t) > 2}
        return tokens

    def _find_by_question(self, question_substring: str) -> Optional[str]:
        for entry in self._entries:
            if question_substring.lower() in entry["question"].lower():
                return entry["answer"]
        return None

    def search(self, query: str, threshold: float = 0.20) -> Optional[str]:
        if not query or not query.strip():
            return None

        # Cleaned lowercase query
        q_clean = re.sub(r"[^\w\s]", " ", query.lower()).strip()
        words = set(q_clean.split())

        # 1. Routing rules for high-priority specific keywords
        if "bmi" in words:
            return self._find_by_question("What is BMI?")

        if "prevent" in words or "prevention" in words:
            return self._find_by_question("How can I prevent Chronic Kidney Disease?")

        if "egfr" in words or "gfr" in words:
            if "normal" in words or "healthy" in words:
                return self._find_by_question("What is a normal eGFR level?")
            return self._find_by_question("What is eGFR?")

        if "creatinine" in words:
            return self._find_by_question("What is Serum Creatinine?")

        if "symptom" in words or "symptoms" in words or "sign" in words or "signs" in words:
            if "early" in words:
                return self._find_by_question("What are the early signs of CKD?")
            return self._find_by_question("What are the common symptoms of CKD?")

        if "diet" in words or "nutrition" in words:
            return self._find_by_question("What is a renal (kidney-friendly) diet?")

        if "food" in words or "foods" in words or "eat" in words:
            if "avoid" in words or "limit" in words or "restrict" in words or "bad" in words:
                return self._find_by_question("What foods should I avoid with CKD?")
            return self._find_by_question("What foods are good for the kidneys?")

        if "exercise" in words or "workout" in words or "fitness" in words or "active" in words:
            return self._find_by_question("Is exercise safe for CKD patients?")

        if "medication" in words or "medications" in words or "drug" in words or "drugs" in words or "painkiller" in words or "painkillers" in words or "nsaid" in words or "nsaids" in words:
            return self._find_by_question("What medication precautions should I take in CKD?")

        if "dialysis" in words or "hemodialysis" in words or "peritoneal" in words:
            return self._find_by_question("What is dialysis?")

        if "transplant" in words or "donor" in words:
            return self._find_by_question("What is a kidney transplant?")

        if "pdf" in words or "report" in words or "download" in words:
            return self._find_by_question("How do I download a PDF report?")

        if "history" in words:
            return self._find_by_question("Where can I view past predictions?")

        if "appointment" in words or "reminder" in words or "schedule" in words:
            return self._find_by_question("How do I schedule an appointment?")

        # 2. Token overlap standard search
        query_tokens = self._tokenize(query)
        if not query_tokens:
            # Fallback for generic queries that got completely tokenized out
            if "ckd" in words or "kidney" in words or "disease" in words:
                return self._entries[0]["answer"]
            return None

        best_score = 0.0
        best_entry: Optional[dict] = None

        for kw_set, entry in self._index:
            overlap = len(query_tokens & kw_set)
            if overlap == 0:
                continue

            score = overlap / math.sqrt(len(query_tokens) * len(kw_set))
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= threshold:
            return best_entry["answer"]

        return None

    def get_all_questions(self) -> list[str]:
        return [entry["question"] for entry in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

_kb_singleton: Optional[KnowledgeBase] = None

def get_knowledge_base() -> KnowledgeBase:
    global _kb_singleton
    if _kb_singleton is None:
        _kb_singleton = KnowledgeBase()
    return _kb_singleton
