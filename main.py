"""
Medical NLP Project - Main Execution Script

This script demonstrates the complete NLP pipeline for medical text analysis.

Based on: "Natural language processing techniques applied to the electronic health record 
in clinical research and practice" by Clay et al. (2025)
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from preprocessing import MedicalTextPreprocessor, preprocess_text
from anonymization import MedicalTextAnonymizer, anonymize_text
from bow_model import BagOfWordsModel
from tfidf_model import TFIDFModel
from utils import (load_data, save_data, create_wordcloud, plot_class_distribution,
                  plot_text_length_distribution, compare_model_performance, ensure_dir)
from sklearn.model_selection import train_test_split


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(text)
    print("="*70 + "\n")


def demo_preprocessing():
    """Demonstrate text preprocessing."""
    print_header("1. TEXT PREPROCESSING DEMONSTRATION")
    
    sample_text = """
    Patient John Smith (DOB: 03/15/1975) presents with severe chest pain 
    and shortness of breath. He's been experiencing these symptoms for 24 hours.
    Blood pressure is elevated at 160/95 mmHg. Contact: (555) 123-4567.
    """
    
    print("Original Text:")
    print(sample_text)
    
    # Preprocessing
    preprocessor = MedicalTextPreprocessor()
    processed = preprocessor.preprocess(sample_text, return_tokens=False)
    
    print("\nPreprocessed Text:")
    print(processed)
    
    # Tokenization
    tokens = preprocessor.preprocess(sample_text, return_tokens=True)
    print(f"\nTokens ({len(tokens)}): {tokens[:15]}...")
    
    # POS Tagging
    pos_tags = preprocessor.pos_tagging(preprocessor.tokenize_words(sample_text))
    print(f"\nPOS Tags (first 10): {pos_tags[:10]}")


def demo_anonymization():
    """Demonstrate text anonymization."""
    print_header("2. TEXT ANONYMIZATION DEMONSTRATION")
    
    sample_text = """
    Patient: John Smith
    DOB: 03/15/1975
    MRN: 12345678
    Phone: (555) 123-4567
    Email: john.smith@email.com
    
    Dr. Sarah Johnson examined the 48-year-old patient on 12/10/2023.
    Patient reports chest pain and elevated blood pressure.
    """
    
    print("Original Text:")
    print(sample_text)
    
    # Anonymization with mask strategy
    anonymizer = MedicalTextAnonymizer(replacement_strategy='mask')
    anonymized = anonymizer.anonymize(sample_text)
    
    print("\nAnonymized Text:")
    print(anonymized)


def demo_bow_model():
    """Demonstrate Bag of Words model."""
    print_header("3. BAG OF WORDS (BoW) MODEL DEMONSTRATION")
    
    # Sample medical texts
    texts = [
        "Patient presents with severe chest pain and shortness of breath",
        "Patient has diabetes and hypertension, requires medication adjustment",
        "Acute myocardial infarction, emergency intervention required",
        "Routine checkup, patient is healthy and stable",
        "Patient reports headache and dizziness for past week",
        "Cardiovascular disease, patient needs cardiac catheterization",
        "Patient has normal blood pressure and heart rate",
        "Emergency admission for stroke symptoms",
        "Patient diagnosed with type 2 diabetes mellitus",
        "Chronic obstructive pulmonary disease with acute exacerbation"
    ]
    
    labels = [
        "Emergency", "Routine", "Emergency", "Routine", "Routine",
        "Emergency", "Routine", "Emergency", "Routine", "Emergency"
    ]
    
    # Preprocess
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.3, random_state=42
    )
    
    # Create and train BoW model
    bow_model = BagOfWordsModel(max_features=50)
    X_train_bow = bow_model.fit_transform(X_train)
    X_test_bow = bow_model.transform(X_test)
    
    print(f"BoW Feature Matrix Shape: {X_train_bow.shape}")
    print(f"Vocabulary Size: {len(bow_model.get_feature_names())}")
    print(f"Top 10 Features: {bow_model.get_feature_names()[:10]}")
    
    # Train classifier
    bow_model.train_classifier(X_train_bow, y_train, classifier_type='naive_bayes')
    
    # Evaluate
    results = bow_model.evaluate(X_test_bow, y_test)
    print(f"\nBoW Model Accuracy: {results['accuracy']:.2%}")
    
    return results['accuracy']


def demo_tfidf_model():
    """Demonstrate TF-IDF model."""
    print_header("4. TF-IDF MODEL DEMONSTRATION")
    
    # Sample medical texts
    texts = [
        "Patient presents with severe chest pain and shortness of breath",
        "Patient has diabetes and hypertension, requires medication adjustment",
        "Acute myocardial infarction, emergency intervention required",
        "Routine checkup, patient is healthy and stable",
        "Patient reports headache and dizziness for past week",
        "Cardiovascular disease, patient needs cardiac catheterization",
        "Patient has normal blood pressure and heart rate",
        "Emergency admission for stroke symptoms",
        "Patient diagnosed with type 2 diabetes mellitus",
        "Chronic obstructive pulmonary disease with acute exacerbation"
    ]
    
    labels = [
        "Emergency", "Routine", "Emergency", "Routine", "Routine",
        "Emergency", "Routine", "Emergency", "Routine", "Emergency"
    ]
    
    # Preprocess
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.3, random_state=42
    )
    
    # Create and train TF-IDF model
    tfidf_model = TFIDFModel(max_features=50)
    X_train_tfidf = tfidf_model.fit_transform(X_train)
    X_test_tfidf = tfidf_model.transform(X_test)
    
    print(f"TF-IDF Feature Matrix Shape: {X_train_tfidf.shape}")
    print(f"Vocabulary Size: {len(tfidf_model.get_feature_names())}")
    print(f"Top 10 Features: {tfidf_model.get_feature_names()[:10]}")

    # Train classifier
    tfidf_model.train_classifier(X_train_tfidf, y_train, classifier_type='svm')

    # Evaluate
    results = tfidf_model.evaluate(X_test_tfidf, y_test)
    print(f"\nTF-IDF Model Accuracy: {results['accuracy']:.2%}")

    return results['accuracy']


def main():
    """Main execution function."""
    print_header("MEDICAL NLP PROJECT - COMPLETE PIPELINE DEMONSTRATION")
    print("Based on: 'Natural language processing techniques applied to the")
    print("electronic health record in clinical research and practice'")
    print("by Clay et al. (2025)")

    # Ensure output directories exist
    ensure_dir('outputs/visualizations')
    ensure_dir('outputs/results')
    ensure_dir('models/saved_models')

    # Run demonstrations
    try:
        demo_preprocessing()
    except Exception as e:
        print(f"Error in preprocessing demo: {e}")

    try:
        demo_anonymization()
    except Exception as e:
        print(f"Error in anonymization demo: {e}")

    try:
        bow_accuracy = demo_bow_model()
    except Exception as e:
        print(f"Error in BoW demo: {e}")
        bow_accuracy = 0.0

    try:
        tfidf_accuracy = demo_tfidf_model()
    except Exception as e:
        print(f"Error in TF-IDF demo: {e}")
        tfidf_accuracy = 0.0

    # Compare models
    if bow_accuracy > 0 or tfidf_accuracy > 0:
        print_header("5. MODEL COMPARISON")
        results_dict = {
            'Bag of Words': bow_accuracy,
            'TF-IDF': tfidf_accuracy
        }

        print("Model Performance Summary:")
        for model, acc in results_dict.items():
            print(f"  {model}: {acc:.2%}")

        # Plot comparison
        try:
            compare_model_performance(results_dict,
                                    save_path='outputs/visualizations/model_comparison.png')
        except Exception as e:
            print(f"Could not create comparison plot: {e}")

    print_header("DEMONSTRATION COMPLETE")
    print("Next steps:")
    print("1. Explore Jupyter notebooks in the 'notebooks/' directory")
    print("2. Run: jupyter notebook notebooks/")
    print("3. Try with your own medical text data")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()


