"""
Model Comparison Module
Compares all NLP models: BoW, TF-IDF, Word2Vec, BioBERT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from typing import List, Dict
import time

from preprocessing import preprocess_text
from bow_model import BagOfWordsModel
from tfidf_model import TFIDFModel
from embedding_model import MedicalEmbeddingModel


class ModelComparator:
    """
    Compare multiple NLP models on the same dataset.
    """
    
    def __init__(self):
        self.results = {}
        self.models = {}
    
    def compare_all_models(self, X_train: List[str], X_test: List[str],
                          y_train: List[str], y_test: List[str],
                          include_biobert: bool = False) -> pd.DataFrame:
        """
        Train and evaluate all models.
        
        Args:
            X_train (list): Training texts (preprocessed)
            X_test (list): Test texts (preprocessed)
            y_train (list): Training labels
            y_test (list): Test labels
            include_biobert (bool): Whether to include BioBERT (slow)
            
        Returns:
            pd.DataFrame: Comparison results
        """
        print("="*70)
        print("MODEL COMPARISON - TRAINING ALL MODELS")
        print("="*70)
        
        # 1. Bag of Words
        print("\n1️⃣  Training Bag of Words...")
        start_time = time.time()
        bow_model = BagOfWordsModel(max_features=50)
        X_train_bow = bow_model.fit_transform(X_train)
        X_test_bow = bow_model.transform(X_test)
        bow_model.train_classifier(X_train_bow, y_train, classifier_type='naive_bayes')
        bow_results = bow_model.evaluate(X_test_bow, y_test)
        bow_time = time.time() - start_time
        
        self.models['BoW'] = bow_model
        self.results['BoW'] = {
            'accuracy': bow_results['accuracy'],
            'training_time': bow_time,
            'model_type': 'Statistical'
        }
        print(f"   ✅ BoW Accuracy: {bow_results['accuracy']:.2%} (Time: {bow_time:.2f}s)")
        
        # 2. TF-IDF
        print("\n2️⃣  Training TF-IDF...")
        start_time = time.time()
        tfidf_model = TFIDFModel(max_features=50)
        X_train_tfidf = tfidf_model.fit_transform(X_train)
        X_test_tfidf = tfidf_model.transform(X_test)
        tfidf_model.train_classifier(X_train_tfidf, y_train, classifier_type='svm')
        tfidf_results = tfidf_model.evaluate(X_test_tfidf, y_test)
        tfidf_time = time.time() - start_time
        
        self.models['TF-IDF'] = tfidf_model
        self.results['TF-IDF'] = {
            'accuracy': tfidf_results['accuracy'],
            'training_time': tfidf_time,
            'model_type': 'Statistical'
        }
        print(f"   ✅ TF-IDF Accuracy: {tfidf_results['accuracy']:.2%} (Time: {tfidf_time:.2f}s)")
        
        # 3. Word2Vec Embeddings
        print("\n3️⃣  Training Word2Vec Embeddings...")
        start_time = time.time()
        embedding_model = MedicalEmbeddingModel(vector_size=100, epochs=10)
        X_train_emb = embedding_model.fit_transform(X_train)
        X_test_emb = embedding_model.transform(X_test)
        embedding_model.train_classifier(X_train_emb, y_train, classifier_type='logistic_regression')
        emb_results = embedding_model.evaluate(X_test_emb, y_test)
        emb_time = time.time() - start_time
        
        self.models['Word2Vec'] = embedding_model
        self.results['Word2Vec'] = {
            'accuracy': emb_results['accuracy'],
            'training_time': emb_time,
            'model_type': 'Neural Embedding'
        }
        print(f"   ✅ Word2Vec Accuracy: {emb_results['accuracy']:.2%} (Time: {emb_time:.2f}s)")
        
        # 4. BioBERT (optional - slow)
        if include_biobert:
            try:
                from biobert_model import BioBERTClassifier
                print("\n4️⃣  Training BioBERT (this may take a while)...")
                start_time = time.time()
                biobert_model = BioBERTClassifier(max_length=64)
                biobert_model.fit(X_train, y_train, epochs=2, batch_size=4)
                biobert_results = biobert_model.evaluate(X_test, y_test)
                biobert_time = time.time() - start_time
                
                self.models['BioBERT'] = biobert_model
                self.results['BioBERT'] = {
                    'accuracy': biobert_results['accuracy'],
                    'training_time': biobert_time,
                    'model_type': 'Deep Learning'
                }
                print(f"   ✅ BioBERT Accuracy: {biobert_results['accuracy']:.2%} (Time: {biobert_time:.2f}s)")
            except Exception as e:
                print(f"   ⚠️  BioBERT training failed: {e}")
                print(f"   Skipping BioBERT...")
        
        # Create comparison DataFrame
        df_results = pd.DataFrame(self.results).T
        df_results = df_results.sort_values('accuracy', ascending=False)
        
        return df_results
    
    def plot_comparison(self, save_path: str = None):
        """
        Create visualization comparing all models.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        if not self.results:
            print("⚠️  No results to plot. Run compare_all_models() first.")
            return
        
        df = pd.DataFrame(self.results).T
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Accuracy comparison
        ax1 = axes[0]
        models = df.index
        accuracies = df['accuracy'] * 100
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(models)]
        
        bars = ax1.barh(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 100)
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax1.text(acc + 1, i, f'{acc:.1f}%', va='center', fontweight='bold')
        
        # Plot 2: Training time comparison
        ax2 = axes[1]
        times = df['training_time']
        bars2 = ax2.barh(models, times, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (bar, t) in enumerate(zip(bars2, times)):
            ax2.text(t + 0.1, i, f'{t:.2f}s', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        
        plt.show()
    
    def print_summary(self):
        """Print summary of all model results."""
        if not self.results:
            print("⚠️  No results available.")
            return
        
        df = pd.DataFrame(self.results).T
        df = df.sort_values('accuracy', ascending=False)
        
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)
        print(df.to_string())
        print("="*70)
        
        best_model = df.index[0]
        best_accuracy = df.loc[best_model, 'accuracy']
        print(f"\n🏆 Best Model: {best_model} ({best_accuracy:.2%} accuracy)")

