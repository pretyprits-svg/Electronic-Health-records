"""
Text Preprocessing Module for Medical NLP
Based on the paper: "Natural language processing techniques applied to the electronic health record"

This module implements essential preprocessing steps:
1. Sentence segmentation
2. Word tokenization
3. Removal of contractions
4. Case elimination
5. Removal of stopwords
6. Stemming and Lemmatization
7. Part-of-Speech (POS) tagging
"""

import re
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
import contractions
import pandas as pd
from typing import List, Dict, Union

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class MedicalTextPreprocessor:
    """
    A comprehensive text preprocessor for medical documents.
    """
    
    def __init__(self, remove_stopwords=True, use_lemmatization=True):
        """
        Initialize the preprocessor.
        
        Args:
            remove_stopwords (bool): Whether to remove stopwords
            use_lemmatization (bool): Use lemmatization (True) or stemming (False)
        """
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into individual sentences.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of sentences
        """
        return sent_tokenize(text)
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into individual words.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of tokens
        """
        return word_tokenize(text)
    
    def expand_contractions(self, text: str) -> str:
        """
        Expand contractions (e.g., "don't" -> "do not").
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with expanded contractions
        """
        return contractions.fix(text)
    
    def to_lowercase(self, text: str) -> str:
        """
        Convert text to lowercase.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Lowercase text
        """
        return text.lower()
    
    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text without punctuation
        """
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_stopwords_func(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Tokens without stopwords
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming to tokens.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to tokens.

        Args:
            tokens (List[str]): List of tokens

        Returns:
            List[str]: Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def pos_tagging(self, tokens: List[str]) -> List[tuple]:
        """
        Apply Part-of-Speech tagging to tokens.

        Args:
            tokens (List[str]): List of tokens

        Returns:
            List[tuple]: List of (token, POS_tag) tuples
        """
        return pos_tag(tokens)

    def preprocess(self, text: str, return_tokens=True) -> Union[List[str], str]:
        """
        Complete preprocessing pipeline.

        Args:
            text (str): Input text
            return_tokens (bool): Return tokens (True) or joined string (False)

        Returns:
            Union[List[str], str]: Preprocessed tokens or text
        """
        # Step 1: Expand contractions
        text = self.expand_contractions(text)

        # Step 2: Convert to lowercase
        text = self.to_lowercase(text)

        # Step 3: Remove punctuation
        text = self.remove_punctuation(text)

        # Step 4: Tokenize
        tokens = self.tokenize_words(text)

        # Step 5: Remove stopwords (optional)
        if self.remove_stopwords:
            tokens = self.remove_stopwords_func(tokens)

        # Step 6: Stemming or Lemmatization
        if self.use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        else:
            tokens = self.stem_tokens(tokens)

        # Return tokens or joined string
        if return_tokens:
            return tokens
        else:
            return ' '.join(tokens)

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str,
                            new_column: str = 'processed_text') -> pd.DataFrame:
        """
        Preprocess a text column in a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of column containing text
            new_column (str): Name for new preprocessed column

        Returns:
            pd.DataFrame: DataFrame with new preprocessed column
        """
        df[new_column] = df[text_column].apply(lambda x: self.preprocess(x, return_tokens=False))
        return df


def preprocess_text(text: str, remove_stopwords=True, use_lemmatization=True) -> str:
    """
    Convenience function for quick text preprocessing.

    Args:
        text (str): Input text
        remove_stopwords (bool): Whether to remove stopwords
        use_lemmatization (bool): Use lemmatization (True) or stemming (False)

    Returns:
        str: Preprocessed text
    """
    preprocessor = MedicalTextPreprocessor(remove_stopwords, use_lemmatization)
    return preprocessor.preprocess(text, return_tokens=False)


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Patient presents with severe chest pain and shortness of breath.
    He's been experiencing these symptoms for the past 24 hours.
    Blood pressure is elevated at 160/95 mmHg.
    """

    print("Original Text:")
    print(sample_text)
    print("\n" + "="*50 + "\n")

    preprocessor = MedicalTextPreprocessor()

    # Demonstrate each step
    print("1. Sentence Segmentation:")
    sentences = preprocessor.segment_sentences(sample_text)
    for i, sent in enumerate(sentences, 1):
        print(f"   {i}. {sent}")

    print("\n2. Expanded Contractions:")
    expanded = preprocessor.expand_contractions(sample_text)
    print(f"   {expanded}")

    print("\n3. Tokenization:")
    tokens = preprocessor.tokenize_words(sample_text)
    print(f"   {tokens[:20]}...")

    print("\n4. POS Tagging:")
    pos_tags = preprocessor.pos_tagging(tokens[:10])
    print(f"   {pos_tags}")

    print("\n5. Full Preprocessing:")
    processed = preprocessor.preprocess(sample_text, return_tokens=False)
    print(f"   {processed}")


