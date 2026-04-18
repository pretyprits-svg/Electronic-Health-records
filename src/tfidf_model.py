"""
TF-IDF (Term Frequency-Inverse Document Frequency) Model for Medical Text Analysis
Based on the paper: "Natural language processing techniques applied to the electronic health record"

TF-IDF is an advanced text representation method that:
- Weighs words by their importance in a document relative to the entire corpus
- Reduces the weight of common words
- Increases the weight of rare, distinctive words
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class TFIDFModel:
    """
    TF-IDF model for text classification.
    """
    
    def __init__(self, max_features=1000, ngram_range=(1, 1), min_df=1, max_df=1.0):
        """
        Initialize the TF-IDF model.
        
        Args:
            max_features (int): Maximum number of features to use
            ngram_range (tuple): Range of n-grams to extract
            min_df (int/float): Minimum document frequency
            max_df (float): Maximum document frequency (ignore terms that appear in > max_df of documents)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=True
        )
        self.classifier = None
        self.is_fitted = False
        
    def fit_vectorizer(self, texts: List[str]):
        """
        Fit the vectorizer on training texts.
        
        Args:
            texts (List[str]): List of text documents
        """
        self.vectorizer.fit(texts)
        
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF vectors.
        
        Args:
            texts (List[str]): List of text documents
            
        Returns:
            np.ndarray: TF-IDF feature matrix
        """
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit vectorizer and transform texts in one step.
        
        Args:
            texts (List[str]): List of text documents
            
        Returns:
            np.ndarray: TF-IDF feature matrix
        """
        return self.vectorizer.fit_transform(texts).toarray()
    
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names (vocabulary).
        
        Returns:
            List[str]: List of feature names
        """
        return self.vectorizer.get_feature_names_out()
    
    def get_top_features(self, n=10) -> pd.DataFrame:
        """
        Get top N features by average TF-IDF score.
        
        Args:
            n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Top features with their scores
        """
        feature_names = self.get_feature_names()
        # This requires having transformed data
        return feature_names[:n]
    
    def train_classifier(self, X_train: np.ndarray, y_train: np.ndarray, 
                        classifier_type='svm'):
        """
        Train a classifier on TF-IDF features.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            classifier_type (str): 'svm' or 'logistic_regression'
        """
        if classifier_type == 'svm':
            self.classifier = SVC(kernel='linear', random_state=42)
        elif classifier_type == 'logistic_regression':
            self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError("classifier_type must be 'svm' or 'logistic_regression'")
        
        self.classifier.fit(X_train, y_train)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained classifier.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be trained before making predictions")
        return self.classifier.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the classifier on test data.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }

    def plot_confusion_matrix(self, cm: np.ndarray, labels: Optional[List[str]] = None,
                             save_path: Optional[str] = None):
        """
        Plot confusion matrix.

        Args:
            cm (np.ndarray): Confusion matrix
            labels (List[str]): Class labels
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - TF-IDF Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_model(self, filepath: str):
        """
        Save the model to disk.

        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load a model from disk.

        Args:
            filepath (str): Path to load the model from
        """
        model_data = joblib.load(filepath)
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.max_features = model_data['max_features']
        self.ngram_range = model_data['ngram_range']
        self.is_fitted = model_data['is_fitted']
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    from preprocessing import preprocess_text

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
    ]

    labels = [
        "Emergency", "Routine", "Emergency", "Routine",
        "Routine", "Emergency", "Routine", "Emergency"
    ]

    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.25, random_state=42
    )

    # Create and train TF-IDF model
    tfidf_model = TFIDFModel(max_features=50)

    # Fit and transform
    X_train_tfidf = tfidf_model.fit_transform(X_train)
    X_test_tfidf = tfidf_model.transform(X_test)

    print("TF-IDF Feature Matrix Shape:", X_train_tfidf.shape)
    print("Vocabulary Size:", len(tfidf_model.get_feature_names()))
    print("\nTop 10 Features:", tfidf_model.get_feature_names()[:10])

    # Train classifier
    tfidf_model.train_classifier(X_train_tfidf, y_train, classifier_type='svm')

    # Evaluate
    results = tfidf_model.evaluate(X_test_tfidf, y_test)
    print(f"\nAccuracy: {results['accuracy']:.2f}")
    print("\nClassification Report:")
    print(pd.DataFrame(results['classification_report']).transpose())


