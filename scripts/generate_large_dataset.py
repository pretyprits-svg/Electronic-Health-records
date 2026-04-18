"""
Generate Large Realistic Medical Dataset for Training
Creates 10,000+ synthetic medical records with realistic variations
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Medical templates for different conditions
DISEASE_TEMPLATES = {
    'Diabetes': [
        "Patient presents with elevated blood glucose levels at {glucose} mg/dL. HbA1c: {hba1c}%. {symptom}. Diagnosed with Type {type} Diabetes Mellitus. Started on {medication}.",
        "{age_desc} patient with history of diabetes. Current blood sugar: {glucose} mg/dL. {complication}. Medication adjusted to {medication}.",
        "Diabetic patient reports {symptom}. Fasting glucose: {glucose} mg/dL. {treatment} recommended.",
    ],
    'Hypertension': [
        "Blood pressure reading: {bp_sys}/{bp_dia} mmHg. Patient reports {symptom}. Diagnosed with {stage} hypertension. Prescribed {medication}.",
        "{age_desc} patient with elevated BP: {bp_sys}/{bp_dia}. {complication}. Started on {medication} therapy.",
        "Hypertensive crisis with BP {bp_sys}/{bp_dia}. {symptom}. Immediate treatment with {medication} initiated.",
    ],
    'Asthma': [
        "Patient experiencing {severity} asthma exacerbation. {symptom}. Peak flow: {peak_flow}% predicted. {treatment} administered.",
        "{age_desc} asthmatic with {symptom}. FEV1: {fev1}% predicted. Prescribed {medication}.",
        "Acute asthma attack. {symptom}. Oxygen saturation: {spo2}%. Nebulizer treatment with {medication} given.",
    ],
    'Cardiac': [
        "Patient presents with {symptom}. ECG shows {ecg_finding}. Troponin: {troponin}. Diagnosed with {diagnosis}. {treatment} initiated.",
        "{age_desc} patient with chest pain. {symptom}. Cardiac enzymes elevated. {diagnosis} confirmed. Started on {medication}.",
        "Acute coronary syndrome. {symptom}. ST-segment {st_change}. Emergency {treatment} performed.",
    ],
    'Respiratory': [
        "Patient with {symptom}. Chest X-ray shows {xray_finding}. Oxygen saturation: {spo2}%. Diagnosed with {diagnosis}. {treatment} started.",
        "{age_desc} patient presenting with {symptom}. Respiratory rate: {rr} breaths/min. {diagnosis}. Prescribed {medication}.",
        "Acute respiratory distress. {symptom}. ABG shows {abg_finding}. {treatment} initiated.",
    ],
    'Neurological': [
        "Patient reports {symptom}. Neurological exam shows {neuro_finding}. MRI reveals {mri_finding}. Diagnosed with {diagnosis}. {treatment} recommended.",
        "{age_desc} patient with {symptom}. CT scan shows {ct_finding}. {diagnosis} confirmed. Started on {medication}.",
        "Acute neurological event. {symptom}. NIH Stroke Scale: {nihss}. {treatment} administered within therapeutic window.",
    ],
    'Gastrointestinal': [
        "Patient complains of {symptom}. Endoscopy shows {endo_finding}. Diagnosed with {diagnosis}. Prescribed {medication}.",
        "{age_desc} patient with {symptom}. Abdominal exam reveals {exam_finding}. {diagnosis}. {treatment} initiated.",
        "Acute {symptom}. Labs show {lab_finding}. Diagnosed with {diagnosis}. {treatment} recommended.",
    ],
    'Orthopedic': [
        "Patient with {symptom} following {mechanism}. X-ray shows {xray_finding}. Diagnosed with {diagnosis}. {treatment} performed.",
        "{age_desc} patient presenting with {symptom}. Range of motion: {rom}. {diagnosis}. Referred for {treatment}.",
        "Traumatic injury. {symptom}. Imaging reveals {imaging_finding}. {diagnosis} confirmed. {treatment} scheduled.",
    ],
    'Renal': [
        "Patient with {symptom}. Creatinine: {creatinine} mg/dL. eGFR: {egfr} mL/min. Diagnosed with {diagnosis}. {treatment} initiated.",
        "{age_desc} patient with declining renal function. {symptom}. BUN: {bun} mg/dL. {diagnosis}. Started on {medication}.",
        "Acute kidney injury. {symptom}. Urine output: {urine_output} mL/hr. {diagnosis}. {treatment} recommended.",
    ],
    'Psychiatric': [
        "Patient reports {symptom}. PHQ-9 score: {phq9}. Diagnosed with {diagnosis}. {treatment} recommended.",
        "{age_desc} patient with {symptom}. Mental status exam shows {mse_finding}. {diagnosis}. Prescribed {medication}.",
        "Acute psychiatric crisis. {symptom}. Risk assessment: {risk}. {diagnosis}. {treatment} initiated.",
    ],
}

# Variable options for templates
VARIABLES = {
    'glucose': lambda: random.randint(80, 450),
    'hba1c': lambda: round(random.uniform(5.0, 12.0), 1),
    'type': lambda: random.choice(['1', '2']),
    'bp_sys': lambda: random.randint(110, 220),
    'bp_dia': lambda: random.randint(70, 140),
    'peak_flow': lambda: random.randint(40, 95),
    'fev1': lambda: random.randint(35, 90),
    'spo2': lambda: random.randint(85, 99),
    'troponin': lambda: round(random.uniform(0.01, 5.0), 2),
    'rr': lambda: random.randint(12, 35),
    'nihss': lambda: random.randint(0, 25),
    'creatinine': lambda: round(random.uniform(0.8, 8.0), 1),
    'egfr': lambda: random.randint(15, 120),
    'bun': lambda: random.randint(10, 80),
    'urine_output': lambda: random.randint(10, 100),
    'phq9': lambda: random.randint(0, 27),
    'age_desc': lambda: random.choice(['Young', 'Middle-aged', 'Elderly', 'Geriatric']),
}

# Symptoms, medications, etc. (abbreviated for space)
SYMPTOMS = {
    'Diabetes': ['polyuria', 'polydipsia', 'fatigue', 'blurred vision', 'weight loss'],
    'Hypertension': ['headache', 'dizziness', 'chest pain', 'shortness of breath'],
    'Asthma': ['wheezing', 'dyspnea', 'chest tightness', 'cough'],
    'Cardiac': ['chest pain', 'dyspnea', 'diaphoresis', 'nausea'],
    # ... add more
}

MEDICATIONS = {
    'Diabetes': ['Metformin', 'Insulin', 'Glipizide', 'Januvia'],
    'Hypertension': ['Lisinopril', 'Amlodipine', 'Losartan', 'Hydrochlorothiazide'],
    'Asthma': ['Albuterol', 'Fluticasone', 'Montelukast', 'Budesonide'],
    # ... add more
}

def generate_record(disease, index):
    """Generate a single medical record"""
    template = random.choice(DISEASE_TEMPLATES[disease])
    
    # Fill in variables
    record = template
    for var, func in VARIABLES.items():
        if f'{{{var}}}' in record:
            record = record.replace(f'{{{var}}}', str(func()))
    
    # Fill in symptoms
    if '{symptom}' in record:
        symptom = random.choice(SYMPTOMS.get(disease, ['symptoms']))
        record = record.replace('{symptom}', symptom)
    
    # Fill in medications
    if '{medication}' in record:
        medication = random.choice(MEDICATIONS.get(disease, ['medication']))
        record = record.replace('{medication}', medication)
    
    # Fill remaining placeholders with generic text
    record = record.replace('{stage}', random.choice(['Stage 1', 'Stage 2']))
    record = record.replace('{severity}', random.choice(['mild', 'moderate', 'severe']))
    record = record.replace('{diagnosis}', disease)
    record = record.replace('{treatment}', 'treatment')
    record = record.replace('{complication}', 'no complications noted')
    
    return record

def generate_dataset(num_records=10000):
    """Generate complete dataset"""
    diseases = list(DISEASE_TEMPLATES.keys())
    records = []
    
    for i in range(num_records):
        disease = random.choice(diseases)
        text = generate_record(disease, i)
        records.append({'text': text, 'label': disease, 'record_id': i})
    
    return pd.DataFrame(records)

if __name__ == "__main__":
    print("Generating large medical dataset...")
    df = generate_dataset(10000)
    
    df.to_csv('data/processed/large_medical_dataset.csv', index=False)
    print(f"✅ Generated {len(df)} records")
    print(f"✅ Saved to: data/processed/large_medical_dataset.csv")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())

