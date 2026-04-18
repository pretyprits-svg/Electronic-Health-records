"""
Privacy-Utility Tradeoff Analysis for K-Anonymity
Analyzes the impact of anonymization on model performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from collections import Counter
import re


class PrivacyUtilityAnalyzer:
    """
    Analyze privacy-utility tradeoffs in medical text anonymization.
    """
    
    def __init__(self):
        self.results = {}
    
    def calculate_k_anonymity(self, df: pd.DataFrame, 
                             quasi_identifiers: List[str]) -> int:
        """
        Calculate the k-anonymity level of a dataset.
        
        Args:
            df (DataFrame): Dataset
            quasi_identifiers (list): List of quasi-identifier columns
            
        Returns:
            int: k-anonymity level (minimum group size)
        """
        # Group by quasi-identifiers
        groups = df.groupby(quasi_identifiers).size()
        
        # k is the minimum group size
        k = groups.min()
        
        return k
    
    def measure_information_loss(self, original_text: str, 
                                 anonymized_text: str) -> Dict:
        """
        Measure information loss from anonymization.
        
        Args:
            original_text (str): Original text
            anonymized_text (str): Anonymized text
            
        Returns:
            dict: Information loss metrics
        """
        # Token-level metrics
        original_tokens = original_text.split()
        anonymized_tokens = anonymized_text.split()
        
        # Character-level metrics
        original_chars = len(original_text)
        anonymized_chars = len(anonymized_text)
        
        # Count changes
        tokens_changed = sum(1 for o, a in zip(original_tokens, anonymized_tokens) if o != a)
        tokens_removed = len(original_tokens) - len(anonymized_tokens)
        
        # Calculate metrics
        token_change_rate = tokens_changed / len(original_tokens) if original_tokens else 0
        token_removal_rate = abs(tokens_removed) / len(original_tokens) if original_tokens else 0
        char_change_rate = abs(original_chars - anonymized_chars) / original_chars if original_chars else 0
        
        # Semantic similarity (simple word overlap)
        original_set = set(original_tokens)
        anonymized_set = set(anonymized_tokens)
        jaccard_similarity = len(original_set & anonymized_set) / len(original_set | anonymized_set) if original_set | anonymized_set else 0
        
        return {
            'token_change_rate': token_change_rate,
            'token_removal_rate': token_removal_rate,
            'char_change_rate': char_change_rate,
            'jaccard_similarity': jaccard_similarity,
            'original_length': len(original_tokens),
            'anonymized_length': len(anonymized_tokens)
        }
    
    def analyze_privacy_utility_tradeoff(self, 
                                        original_texts: List[str],
                                        anonymized_texts_dict: Dict[str, List[str]],
                                        original_accuracy: float,
                                        anonymized_accuracies: Dict[str, float]) -> pd.DataFrame:
        """
        Analyze privacy-utility tradeoff across different anonymization levels.
        
        Args:
            original_texts (list): Original texts
            anonymized_texts_dict (dict): {k_value: anonymized_texts}
            original_accuracy (float): Model accuracy on original data
            anonymized_accuracies (dict): {k_value: accuracy on anonymized data}
            
        Returns:
            DataFrame: Tradeoff analysis results
        """
        results = []
        
        for k_value, anonymized_texts in anonymized_texts_dict.items():
            # Calculate average information loss
            info_losses = [
                self.measure_information_loss(orig, anon)
                for orig, anon in zip(original_texts, anonymized_texts)
            ]
            
            avg_token_change = np.mean([il['token_change_rate'] for il in info_losses])
            avg_jaccard = np.mean([il['jaccard_similarity'] for il in info_losses])
            
            # Get accuracy
            accuracy = anonymized_accuracies.get(k_value, 0.0)
            accuracy_drop = original_accuracy - accuracy
            
            results.append({
                'k_value': k_value,
                'privacy_level': f'k={k_value}',
                'accuracy': accuracy,
                'accuracy_drop': accuracy_drop,
                'accuracy_retention': accuracy / original_accuracy if original_accuracy > 0 else 0,
                'avg_token_change_rate': avg_token_change,
                'avg_jaccard_similarity': avg_jaccard,
                'information_loss': 1 - avg_jaccard
            })
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('k_value')
        
        self.results['tradeoff_analysis'] = df_results
        return df_results
    
    def plot_privacy_utility_curve(self, tradeoff_df: pd.DataFrame, 
                                   save_path: str = None):
        """
        Plot privacy-utility tradeoff curve.
        
        Args:
            tradeoff_df (DataFrame): Tradeoff analysis results
            save_path (str): Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Accuracy vs K-value
        ax1 = axes[0, 0]
        ax1.plot(tradeoff_df['k_value'], tradeoff_df['accuracy'] * 100, 
                marker='o', linewidth=2, markersize=8, color='#2ecc71')
        ax1.set_xlabel('K-Anonymity Level (k)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Model Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Accuracy vs Privacy Level', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Add value labels
        for _, row in tradeoff_df.iterrows():
            ax1.annotate(f"{row['accuracy']*100:.1f}%", 
                        (row['k_value'], row['accuracy']*100),
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        # Plot 2: Information Loss vs K-value
        ax2 = axes[0, 1]
        ax2.plot(tradeoff_df['k_value'], tradeoff_df['information_loss'] * 100,
                marker='s', linewidth=2, markersize=8, color='#e74c3c')
        ax2.set_xlabel('K-Anonymity Level (k)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Information Loss (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Information Loss vs Privacy Level', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Plot 3: Privacy-Utility Tradeoff (Pareto Curve)
        ax3 = axes[1, 0]
        ax3.scatter(tradeoff_df['information_loss'] * 100, 
                   tradeoff_df['accuracy'] * 100,
                   s=200, c=tradeoff_df['k_value'], cmap='viridis', 
                   edgecolors='black', linewidth=2)
        ax3.plot(tradeoff_df['information_loss'] * 100, 
                tradeoff_df['accuracy'] * 100,
                linestyle='--', alpha=0.5, color='gray')
        ax3.set_xlabel('Information Loss (%)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Model Accuracy (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Privacy-Utility Tradeoff Curve', fontsize=14, fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # Add k-value labels
        for _, row in tradeoff_df.iterrows():
            ax3.annotate(f"k={row['k_value']}", 
                        (row['information_loss']*100, row['accuracy']*100),
                        textcoords="offset points", xytext=(10,5), ha='left')
        
        # Plot 4: Accuracy Retention
        ax4 = axes[1, 1]
        bars = ax4.bar(tradeoff_df['k_value'].astype(str), 
                      tradeoff_df['accuracy_retention'] * 100,
                      color='#3498db', edgecolor='black', linewidth=1.5)
        ax4.set_xlabel('K-Anonymity Level (k)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Accuracy Retention (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Accuracy Retention vs Privacy Level', fontsize=14, fontweight='bold')
        ax4.axhline(y=100, color='red', linestyle='--', label='Original Accuracy')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Privacy-utility curve saved to {save_path}")
        
        plt.show()
    
    def generate_privacy_report(self, tradeoff_df: pd.DataFrame) -> str:
        """
        Generate comprehensive privacy analysis report.
        
        Args:
            tradeoff_df (DataFrame): Tradeoff analysis results
            
        Returns:
            str: Formatted report
        """
        report = []
        report.append("="*70)
        report.append("PRIVACY-UTILITY TRADEOFF ANALYSIS REPORT")
        report.append("="*70)
        report.append("")
        
        for _, row in tradeoff_df.iterrows():
            report.append(f"K-Anonymity Level: k={row['k_value']}")
            report.append(f"  Privacy: {row['k_value']}-anonymous (higher k = more privacy)")
            report.append(f"  Accuracy: {row['accuracy']*100:.2f}%")
            report.append(f"  Accuracy Drop: {row['accuracy_drop']*100:.2f}%")
            report.append(f"  Information Loss: {row['information_loss']*100:.2f}%")
            report.append(f"  Jaccard Similarity: {row['avg_jaccard_similarity']:.3f}")
            report.append("")
        
        report.append("="*70)
        report.append("RECOMMENDATIONS:")
        report.append("="*70)
        
        # Find optimal k (best accuracy-privacy balance)
        optimal_idx = (tradeoff_df['accuracy_retention'] * (1 - tradeoff_df['information_loss'])).idxmax()
        optimal_k = tradeoff_df.loc[optimal_idx, 'k_value']
        
        report.append(f"Optimal K-value: k={optimal_k}")
        report.append(f"  - Balances privacy and utility")
        report.append(f"  - Accuracy: {tradeoff_df.loc[optimal_idx, 'accuracy']*100:.2f}%")
        report.append(f"  - Information Loss: {tradeoff_df.loc[optimal_idx, 'information_loss']*100:.2f}%")
        report.append("="*70)
        
        return "\n".join(report)

