"""
Utility Functions for Medical NLP Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from typing import List, Dict, Optional
import os


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    return pd.read_csv(filepath)


def save_data(df: pd.DataFrame, filepath: str):
    """
    Save DataFrame to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filepath (str): Path to save file
    """
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def create_wordcloud(text: str, title: str = "Word Cloud", 
                     save_path: Optional[str] = None):
    """
    Create and display a word cloud from text.
    
    Args:
        text (str): Input text
        title (str): Title for the plot
        save_path (str): Path to save the plot
    """
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         colormap='viridis').generate(text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_class_distribution(labels: List[str], title: str = "Class Distribution",
                           save_path: Optional[str] = None):
    """
    Plot the distribution of classes.
    
    Args:
        labels (List[str]): List of class labels
        title (str): Title for the plot
        save_path (str): Path to save the plot
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    plt.bar(unique, counts, color='skyblue', edgecolor='black')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    for i, (label, count) in enumerate(zip(unique, counts)):
        plt.text(i, count + 0.5, str(count), ha='center', va='bottom')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_text_length_distribution(texts: List[str], title: str = "Text Length Distribution",
                                  save_path: Optional[str] = None):
    """
    Plot the distribution of text lengths.
    
    Args:
        texts (List[str]): List of text documents
        title (str): Title for the plot
        save_path (str): Path to save the plot
    """
    lengths = [len(text.split()) for text in texts]
    
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, color='lightcoral', edgecolor='black')
    plt.xlabel('Number of Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14)
    plt.axvline(np.mean(lengths), color='red', linestyle='--', 
                label=f'Mean: {np.mean(lengths):.1f}')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def compare_model_performance(results_dict: Dict[str, float], 
                             title: str = "Model Performance Comparison",
                             save_path: Optional[str] = None):
    """
    Compare performance of multiple models.
    
    Args:
        results_dict (Dict[str, float]): Dictionary of model names and accuracies
        title (str): Title for the plot
        save_path (str): Path to save the plot
    """
    models = list(results_dict.keys())
    accuracies = list(results_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.ylim(0, 1.0)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_sample_data(df: pd.DataFrame, n: int = 5):
    """
    Print sample rows from DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to sample
        n (int): Number of rows to display
    """
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumn Names: {df.columns.tolist()}")
    print(f"\nFirst {n} rows:")
    print(df.head(n))
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nMissing Values:")
    print(df.isnull().sum())


def ensure_dir(directory: str):
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

