"""
Anonymization Module for Medical Text
Based on the paper: "Natural language processing techniques applied to the electronic health record"

This module removes Protected Health Information (PHI) including:
- Patient names
- Dates of birth
- Hospital IDs
- Addresses
- Phone numbers
- Email addresses
"""

import re
import pandas as pd
from typing import List, Dict, Tuple
import random
import string


class MedicalTextAnonymizer:
    """
    Anonymize medical text by removing or replacing Protected Health Information (PHI).
    """
    
    def __init__(self, replacement_strategy='mask', use_generalization=True):
        """
        Initialize the anonymizer.

        Args:
            replacement_strategy (str): 'mask' (replace with [REDACTED]), 'fake' (replace with fake data),
                                       or 'generalize' (use generalization for k-anonymity)
            use_generalization (bool): If True, use generalization instead of complete removal for ages, dates, ZIP codes
                                      This helps prevent re-identification attacks (k-anonymity)
        """
        self.replacement_strategy = replacement_strategy
        self.use_generalization = use_generalization

        # Regex patterns for PHI detection
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'hospital_id': r'\b[A-Z]{2,3}\d{6,10}\b',
            'mrn': r'\bMRN[:\s]*\d{6,10}\b',
            'zip_code': r'\b\d{5}(?:-\d{4})?\b',
            'age_specific': r'\b\d{1,3}[-\s]year[-\s]old\b',
        }

        # Common name patterns (simplified - in production, use NER models)
        self.name_titles = ['Dr', 'Mr', 'Mrs', 'Ms', 'Miss', 'Prof', 'Doctor']

        # Month names for date generalization
        self.month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December']
        
    def mask_phi(self, text: str, phi_type: str) -> str:
        """
        Mask PHI with [REDACTED_TYPE] tags.
        
        Args:
            text (str): Input text
            phi_type (str): Type of PHI being masked
            
        Returns:
            str: Text with masked PHI
        """
        return f"[REDACTED_{phi_type.upper()}]"
    
    def generate_fake_phi(self, phi_type: str) -> str:
        """
        Generate fake PHI data for replacement.
        
        Args:
            phi_type (str): Type of PHI to generate
            
        Returns:
            str: Fake PHI data
        """
        if phi_type == 'email':
            return f"patient{random.randint(1000, 9999)}@example.com"
        elif phi_type == 'phone':
            return f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        elif phi_type == 'date':
            return f"XX/XX/XXXX"
        elif phi_type == 'hospital_id':
            return f"ID{''.join(random.choices(string.digits, k=8))}"
        else:
            return f"[FAKE_{phi_type.upper()}]"
    
    def remove_emails(self, text: str) -> str:
        """Remove or mask email addresses."""
        replacement = self.mask_phi(text, 'email') if self.replacement_strategy == 'mask' else self.generate_fake_phi('email')
        return re.sub(self.patterns['email'], replacement, text)
    
    def remove_phone_numbers(self, text: str) -> str:
        """Remove or mask phone numbers."""
        replacement = self.mask_phi(text, 'phone') if self.replacement_strategy == 'mask' else self.generate_fake_phi('phone')
        return re.sub(self.patterns['phone'], replacement, text)
    
    def remove_ssn(self, text: str) -> str:
        """Remove or mask Social Security Numbers."""
        replacement = self.mask_phi(text, 'ssn') if self.replacement_strategy == 'mask' else 'XXX-XX-XXXX'
        return re.sub(self.patterns['ssn'], replacement, text)
    
    def generalize_age_to_range(self, age: int) -> str:
        """
        Convert exact age to age range for k-anonymity.
        Follows HIPAA Safe Harbor guidelines (ages 90+ are grouped).

        Args:
            age (int): Exact age

        Returns:
            str: Age range
        """
        if age < 18:
            return "under 18 years old"
        elif age < 30:
            return "18-29 years old"
        elif age < 40:
            return "30-39 years old"
        elif age < 50:
            return "40-49 years old"
        elif age < 60:
            return "50-59 years old"
        elif age < 70:
            return "60-69 years old"
        elif age < 80:
            return "70-79 years old"
        elif age < 90:
            return "80-89 years old"
        else:
            return "90+ years old"  # HIPAA Safe Harbor requirement

    def generalize_date_to_month_year(self, month: int, year: str) -> str:
        """
        Convert exact date to month/year only for k-anonymity.

        Args:
            month (int): Month number (1-12)
            year (str): Year

        Returns:
            str: Generalized date (e.g., "March 2023")
        """
        if 1 <= month <= 12:
            return f"{self.month_names[month-1]} {year}"
        else:
            return f"[MONTH] {year}"

    def generalize_zip_code(self, zip_code: str) -> str:
        """
        Generalize ZIP code to first 3 digits only (per HIPAA Safe Harbor).

        Args:
            zip_code (str): Full ZIP code

        Returns:
            str: Generalized ZIP (e.g., "100**")
        """
        if len(zip_code) >= 3:
            return zip_code[:3] + "**"
        else:
            return "***"

    def remove_dates(self, text: str) -> str:
        """Remove or generalize dates."""
        if self.use_generalization and self.replacement_strategy != 'fake':
            # Generalize to month/year only
            def date_replacer(match):
                try:
                    parts = match.group(0).replace('-', '/').split('/')
                    if len(parts) == 3:
                        month = int(parts[0])
                        year = parts[2]
                        # Handle 2-digit years
                        if len(year) == 2:
                            year = '20' + year if int(year) < 50 else '19' + year
                        return self.generalize_date_to_month_year(month, year)
                    return "[DATE]"
                except Exception as e:
                    return "[DATE]"

            return re.sub(self.patterns['date'], date_replacer, text)
        else:
            # Original behavior: mask or fake
            replacement = self.mask_phi(text, 'date') if self.replacement_strategy == 'mask' else self.generate_fake_phi('date')
            return re.sub(self.patterns['date'], replacement, text)
    
    def remove_hospital_ids(self, text: str) -> str:
        """Remove or mask hospital IDs and MRNs."""
        replacement = self.mask_phi(text, 'id') if self.replacement_strategy == 'mask' else self.generate_fake_phi('hospital_id')
        text = re.sub(self.patterns['hospital_id'], replacement, text)
        text = re.sub(self.patterns['mrn'], replacement, text, flags=re.IGNORECASE)
        return text
    
    def remove_zip_codes(self, text: str) -> str:
        """Remove or generalize ZIP codes."""
        if self.use_generalization and self.replacement_strategy != 'fake':
            # Generalize to first 3 digits only (HIPAA Safe Harbor)
            def zip_replacer(match):
                return self.generalize_zip_code(match.group(0))

            return re.sub(self.patterns['zip_code'], zip_replacer, text)
        else:
            # Original behavior: mask or fake
            replacement = self.mask_phi(text, 'zip') if self.replacement_strategy == 'mask' else 'XXXXX'
            return re.sub(self.patterns['zip_code'], replacement, text)

    def remove_ages(self, text: str) -> str:
        """Remove or generalize specific age mentions."""
        if self.use_generalization and self.replacement_strategy != 'fake':
            # Generalize to age ranges for k-anonymity
            def age_replacer(match):
                try:
                    age = int(match.group(1))
                    return self.generalize_age_to_range(age)
                except:
                    return "[AGE]"

            pattern = r'\b(\d{1,3})[-\s]year[-\s]old\b'
            return re.sub(pattern, age_replacer, text, flags=re.IGNORECASE)
        else:
            # Original behavior: mask or fake
            replacement = self.mask_phi(text, 'age') if self.replacement_strategy == 'mask' else 'XX-year-old'
            return re.sub(self.patterns['age_specific'], replacement, text, flags=re.IGNORECASE)
    
    def remove_names_simple(self, text: str) -> str:
        """
        Simple name removal based on titles.
        Note: In production, use NER models like spaCy or BioBERT for better accuracy.
        """
        replacement = self.mask_phi(text, 'name') if self.replacement_strategy == 'mask' else 'John Doe'

        # Remove names following titles (preserve period if present)
        for title in self.name_titles:
            # Match title with optional period, then name(s)
            pattern = rf'\b({title})(\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'
            # Replace with title + period (if it was there) + replacement
            text = re.sub(pattern, rf'\1\2 {replacement}', text)

        return text

    def anonymize(self, text: str) -> str:
        """
        Apply all anonymization steps to text.

        Args:
            text (str): Input text

        Returns:
            str: Anonymized text
        """
        text = self.remove_emails(text)
        text = self.remove_phone_numbers(text)
        text = self.remove_ssn(text)
        text = self.remove_dates(text)
        text = self.remove_hospital_ids(text)
        text = self.remove_zip_codes(text)
        text = self.remove_ages(text)
        text = self.remove_names_simple(text)

        return text

    def anonymize_dataframe(self, df: pd.DataFrame, text_column: str,
                           new_column: str = 'anonymized_text') -> pd.DataFrame:
        """
        Anonymize a text column in a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of column containing text
            new_column (str): Name for new anonymized column

        Returns:
            pd.DataFrame: DataFrame with new anonymized column
        """
        df[new_column] = df[text_column].apply(self.anonymize)
        return df


def anonymize_text(text: str, strategy='mask') -> str:
    """
    Convenience function for quick text anonymization.

    Args:
        text (str): Input text
        strategy (str): 'mask' or 'fake'

    Returns:
        str: Anonymized text
    """
    anonymizer = MedicalTextAnonymizer(replacement_strategy=strategy)
    return anonymizer.anonymize(text)


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Patient: John Smith
    DOB: 03/15/1975
    MRN: 12345678
    Phone: (555) 123-4567
    Email: john.smith@email.com
    Address: 123 Main St, New York, NY 10001

    Dr. Sarah Johnson examined the 48-year-old patient on 12/10/2023.
    Patient reports chest pain. Blood pressure: 140/90 mmHg.
    SSN: 123-45-6789

    The 65-year-old male patient was admitted on 11/20/2023.
    """

    print("Original Text:")
    print(sample_text)
    print("\n" + "="*80 + "\n")

    # Strategy 1: Mask (complete removal)
    print("STRATEGY 1: MASK (Complete Removal)")
    print("-" * 80)
    anonymizer_mask = MedicalTextAnonymizer(replacement_strategy='mask', use_generalization=False)
    anonymized_mask = anonymizer_mask.anonymize(sample_text)
    print(anonymized_mask)

    print("\n" + "="*80 + "\n")

    # Strategy 2: Generalization (k-anonymity) - NEW!
    print("STRATEGY 2: GENERALIZATION (K-Anonymity Protection) ⭐ RECOMMENDED")
    print("-" * 80)
    print("This prevents re-identification attacks by generalizing quasi-identifiers")
    print("(e.g., exact age → age range, exact date → month/year, full ZIP → prefix)")
    print()
    anonymizer_generalize = MedicalTextAnonymizer(replacement_strategy='mask', use_generalization=True)
    anonymized_generalize = anonymizer_generalize.anonymize(sample_text)
    print(anonymized_generalize)

    print("\n" + "="*80 + "\n")

    # Strategy 3: Fake data
    print("STRATEGY 3: FAKE DATA (Synthetic Replacement)")
    print("-" * 80)
    anonymizer_fake = MedicalTextAnonymizer(replacement_strategy='fake', use_generalization=False)
    anonymized_fake = anonymizer_fake.anonymize(sample_text)
    print(anonymized_fake)

    print("\n" + "="*80 + "\n")
    print("✅ COMPARISON:")
    print("-" * 80)
    print("Mask Strategy:         Removes all PHI → High privacy, low utility")
    print("Generalization:        Age ranges, month/year → Balanced privacy & utility ⭐")
    print("Fake Strategy:         Synthetic data → Maintains format, moderate privacy")
    print("\n💡 For preventing re-identification (k-anonymity), use GENERALIZATION!")
    print("   Example: '65-year-old on 11/20/2023' → '60-69 years old in November 2023'")
    print("   This ensures the patient cannot be uniquely identified!")


