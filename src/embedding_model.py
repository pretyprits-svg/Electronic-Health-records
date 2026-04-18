"""
Word Embeddings Model for Medical NLP
Uses Word2Vec to create dense vector representations of medical terms
"""

import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import List, Optional
import joblib


class MedicalEmbeddingModel(BaseEstimator, TransformerMixin):
    """
    Word2Vec-based embedding model for medical text classification.
    """
    
    def __init__(self, vector_size: int = 100, window: int = 5, 
                 min_count: int = 1, workers: int = 4, epochs: int = 10):
        """
        Initialize the embedding model.
        
        Args:
            vector_size (int): Dimensionality of word vectors
            window (int): Context window size
            min_count (int): Minimum word frequency
            workers (int): Number of worker threads
            epochs (int): Number of training epochs
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model = None
        self.classifier = None
        self.is_fitted = False
    
    def fit(self, X: List[str], y: Optional[List] = None):
        """
        Train Word2Vec model on medical texts.
        
        Args:
            X (list): List of preprocessed texts (space-separated words)
            y (list, optional): Labels (not used for Word2Vec training)
            
        Returns:
            self
        """
        # Convert texts to list of word lists
        sentences = [text.split() for text in X]
        
        # Train Word2Vec
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            sg=1  # Skip-gram model
        )
        
        self.is_fitted = True
        return self
    
    def transform(self, X: List[str]) -> np.ndarray:
        """
        Transform texts to document vectors (average of word vectors).
        
        Args:
            X (list): List of preprocessed texts
            
        Returns:
            np.ndarray: Document vectors
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")
        
        doc_vectors = []
        
        for text in X:
            words = text.split()
            # Get vectors for words in vocabulary
            word_vectors = [
                self.model.wv[word] for word in words 
                if word in self.model.wv
            ]
            
            if word_vectors:
                # Average word vectors to get document vector
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                # If no words in vocabulary, use zero vector
                doc_vector = np.zeros(self.vector_size)
            
            doc_vectors.append(doc_vector)
        
        return np.array(doc_vectors)
    
    def fit_transform(self, X: List[str], y: Optional[List] = None) -> np.ndarray:
        """
        Fit model and transform texts.
        
        Args:
            X (list): List of preprocessed texts
            y (list, optional): Labels
            
        Returns:
            np.ndarray: Document vectors
        """
        self.fit(X, y)
        return self.transform(X)
    
    def train_classifier(self, X_train: np.ndarray, y_train: np.ndarray,
                        classifier_type: str = 'logistic_regression'):
        """
        Train a classifier on embedding features.
        
        Args:
            X_train (np.ndarray): Training features (document vectors)
            y_train (np.ndarray): Training labels
            classifier_type (str): 'logistic_regression' or 'svm'
        """
        if classifier_type == 'logistic_regression':
            self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        elif classifier_type == 'svm':
            self.classifier = SVC(kernel='rbf', random_state=42)
        else:
            raise ValueError("classifier_type must be 'logistic_regression' or 'svm'")
        
        self.classifier.fit(X_train, y_train)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict labels for test data.
        
        Args:
            X_test (np.ndarray): Test features
            
        Returns:
            np.ndarray: Predicted labels
        """
        if self.classifier is None:
            raise ValueError("Classifier must be trained before prediction")
        
        return self.classifier.predict(X_test)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate classifier on test data.
        
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
    
    def get_similar_words(self, word: str, top_n: int = 10) -> List[tuple]:
        """
        Get most similar words to a given word.
        
        Args:
            word (str): Query word
            top_n (int): Number of similar words to return
            
        Returns:
            list: List of (word, similarity) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if word in self.model.wv:
            return self.model.wv.most_similar(word, topn=top_n)
        else:
            return []
    
    def save_model(self, filepath: str):
        """Save the model to disk."""
        model_data = {
            'word2vec_model': self.model,
            'classifier': self.classifier,
            'vector_size': self.vector_size,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load model from disk."""
        model_data = joblib.load(filepath)
        instance = cls(vector_size=model_data['vector_size'])
        instance.model = model_data['word2vec_model']
        instance.classifier = model_data['classifier']
        instance.is_fitted = model_data['is_fitted']
        return instance

