"""
Named Entity Recognition (NER) Module for Medical Text
Extracts medical entities: diseases, symptoms, medications, anatomy, procedures
"""

import re
from typing import List, Dict, Tuple
import nltk
from nltk import pos_tag, word_tokenize
from collections import Counter

class MedicalNER:
    """
    Medical Named Entity Recognition using pattern matching and medical dictionaries.
    """
    
    def __init__(self):
        """Initialize NER with medical entity dictionaries."""
        
        # Medical entity dictionaries
        self.diseases = {
            'diabetes', 'hypertension', 'pneumonia', 'asthma', 'copd',
            'myocardial infarction', 'stroke', 'cancer', 'arthritis',
            'appendicitis', 'thrombosis', 'infection', 'disease',
            'osteoporosis', 'gerd', 'uti', 'kidney disease', 'heart disease'
        }
        
        self.symptoms = {
            'pain', 'chest pain', 'headache', 'fever', 'cough', 'nausea',
            'vomiting', 'dizziness', 'fatigue', 'dyspnea', 'shortness of breath',
            'wheezing', 'swelling', 'bleeding', 'rash', 'weakness',
            'diaphoresis', 'abdominal pain', 'back pain'
        }
        
        self.medications = {
            'aspirin', 'metformin', 'lisinopril', 'atorvastatin', 'omeprazole',
            'levothyroxine', 'amlodipine', 'metoprolol', 'losartan',
            'gabapentin', 'hydrochlorothiazide', 'simvastatin', 'insulin',
            'warfarin', 'prednisone', 'epinephrine', 'anticoagulation'
        }
        
        self.anatomy = {
            'heart', 'lung', 'kidney', 'liver', 'brain', 'stomach',
            'chest', 'abdomen', 'leg', 'arm', 'head', 'neck', 'back',
            'femur', 'joint', 'extremity', 'artery', 'vein'
        }
        
        self.procedures = {
            'surgery', 'biopsy', 'mri', 'ct scan', 'x-ray', 'ultrasound',
            'ecg', 'ekg', 'blood test', 'appendectomy', 'catheterization',
            'intervention', 'examination'
        }
        
        self.lab_values = {
            'troponin', 'glucose', 'creatinine', 'hemoglobin', 'hba1c',
            'bp', 'blood pressure', 't-score', 'fev1'
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from text.
        
        Args:
            text (str): Input medical text
            
        Returns:
            dict: Dictionary of entity types and their occurrences
        """
        text_lower = text.lower()
        
        entities = {
            'diseases': [],
            'symptoms': [],
            'medications': [],
            'anatomy': [],
            'procedures': [],
            'lab_values': []
        }
        
        # Extract diseases
        for disease in self.diseases:
            if disease in text_lower:
                entities['diseases'].append(disease)
        
        # Extract symptoms
        for symptom in self.symptoms:
            if symptom in text_lower:
                entities['symptoms'].append(symptom)
        
        # Extract medications
        for med in self.medications:
            if med in text_lower:
                entities['medications'].append(med)
        
        # Extract anatomy
        for anat in self.anatomy:
            if anat in text_lower:
                entities['anatomy'].append(anat)
        
        # Extract procedures
        for proc in self.procedures:
            if proc in text_lower:
                entities['procedures'].append(proc)
        
        # Extract lab values
        for lab in self.lab_values:
            if lab in text_lower:
                entities['lab_values'].append(lab)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def get_entity_counts(self, texts: List[str]) -> Dict[str, Counter]:
        """
        Get entity counts across multiple texts.
        
        Args:
            texts (list): List of medical texts
            
        Returns:
            dict: Counters for each entity type
        """
        all_entities = {
            'diseases': Counter(),
            'symptoms': Counter(),
            'medications': Counter(),
            'anatomy': Counter(),
            'procedures': Counter(),
            'lab_values': Counter()
        }
        
        for text in texts:
            entities = self.extract_entities(text)
            for entity_type, entity_list in entities.items():
                all_entities[entity_type].update(entity_list)
        
        return all_entities
    
    def summarize_entities(self, texts: List[str], top_n: int = 10) -> Dict[str, List[Tuple[str, int]]]:
        """
        Get top N entities of each type.
        
        Args:
            texts (list): List of medical texts
            top_n (int): Number of top entities to return
            
        Returns:
            dict: Top entities for each type
        """
        counts = self.get_entity_counts(texts)
        
        summary = {}
        for entity_type, counter in counts.items():
            summary[entity_type] = counter.most_common(top_n)
        
        return summary


def extract_medical_entities(text: str) -> Dict[str, List[str]]:
    """
    Convenience function to extract entities from a single text.
    
    Args:
        text (str): Medical text
        
    Returns:
        dict: Extracted entities
    """
    ner = MedicalNER()
    return ner.extract_entities(text)


# Demo
if __name__ == "__main__":
    # Sample medical texts
    sample_texts = [
        "Patient with severe chest pain and dyspnea. History of hypertension and diabetes.",
        "Diagnosed with acute myocardial infarction. Elevated troponin levels.",
        "Patient on metformin for Type 2 Diabetes. Blood glucose well controlled.",
        "Severe headache and visual disturbances. MRI scheduled.",
        "Fractured left femur. Requires surgical intervention."
    ]
    
    print("="*70)
    print("MEDICAL NER DEMONSTRATION")
    print("="*70)
    
    ner = MedicalNER()
    
    # Extract from first text
    print(f"\nSample Text: {sample_texts[0]}")
    print("\nExtracted Entities:")
    entities = ner.extract_entities(sample_texts[0])
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"  {entity_type.upper()}: {', '.join(entity_list)}")
    
    # Summary across all texts
    print("\n" + "="*70)
    print("ENTITY SUMMARY ACROSS ALL TEXTS")
    print("="*70)
    summary = ner.summarize_entities(sample_texts, top_n=5)
    for entity_type, top_entities in summary.items():
        if top_entities:
            print(f"\n{entity_type.upper()}:")
            for entity, count in top_entities:
                print(f"  - {entity}: {count}")

