"""
Advanced Evaluation Metrics for Medical NLP Models
Includes: ROC-AUC, Precision-Recall, Cross-Validation, Statistical Tests
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import label_binarize
from scipy import stats
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class AdvancedEvaluator:
    """
    Comprehensive evaluation suite for medical NLP models.
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize evaluator.
        
        Args:
            class_names (list): List of class names
        """
        self.class_names = class_names
        self.results = {}
    
    def evaluate_comprehensive(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_proba: np.ndarray = None, 
                               model_name: str = "Model") -> Dict:
        """
        Comprehensive evaluation with all metrics.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            y_proba (array): Prediction probabilities (for ROC-AUC)
            model_name (str): Name of the model
            
        Returns:
            dict: All evaluation metrics
        """
        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
        }
        
        # Per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        results['per_class_metrics'] = report
        
        # ROC-AUC (if probabilities provided)
        if y_proba is not None:
            try:
                # Multi-class ROC-AUC
                classes = np.unique(y_true)
                y_true_bin = label_binarize(y_true, classes=classes)
                
                if len(classes) == 2:
                    # Binary classification
                    results['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class
                    results['roc_auc_macro'] = roc_auc_score(y_true_bin, y_proba, 
                                                             average='macro', multi_class='ovr')
                    results['roc_auc_weighted'] = roc_auc_score(y_true_bin, y_proba, 
                                                                average='weighted', multi_class='ovr')
            except Exception as e:
                print(f"⚠️  Could not calculate ROC-AUC: {e}")
        
        self.results[model_name] = results
        return results
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                            cv: int = 5, scoring: str = 'accuracy') -> Dict:
        """
        Perform k-fold cross-validation.
        
        Args:
            model: Sklearn-compatible model
            X (array): Features
            y (array): Labels
            cv (int): Number of folds
            scoring (str): Scoring metric
            
        Returns:
            dict: Cross-validation results
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
        
        return {
            'cv_scores': scores,
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_min': scores.min(),
            'cv_max': scores.max()
        }
    
    def mcnemar_test(self, y_true: np.ndarray, y_pred1: np.ndarray, 
                     y_pred2: np.ndarray) -> Dict:
        """
        McNemar's test for comparing two models.
        
        Args:
            y_true (array): True labels
            y_pred1 (array): Predictions from model 1
            y_pred2 (array): Predictions from model 2
            
        Returns:
            dict: Test results
        """
        # Create contingency table
        correct1 = (y_pred1 == y_true)
        correct2 = (y_pred2 == y_true)
        
        n00 = np.sum(~correct1 & ~correct2)  # Both wrong
        n01 = np.sum(~correct1 & correct2)   # Model 1 wrong, Model 2 correct
        n10 = np.sum(correct1 & ~correct2)   # Model 1 correct, Model 2 wrong
        n11 = np.sum(correct1 & correct2)    # Both correct
        
        # McNemar's test statistic
        if n01 + n10 == 0:
            p_value = 1.0
        else:
            statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
            p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        return {
            'contingency_table': [[n11, n10], [n01, n00]],
            'n_both_correct': n11,
            'n_both_wrong': n00,
            'n_only_model1_correct': n10,
            'n_only_model2_correct': n01,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def plot_roc_curves(self, y_true: np.ndarray, y_proba_dict: Dict[str, np.ndarray],
                       save_path: str = None):
        """
        Plot ROC curves for multiple models.
        
        Args:
            y_true (array): True labels
            y_proba_dict (dict): Dictionary of {model_name: probabilities}
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(y_proba_dict)))
        
        for (model_name, y_proba), color in zip(y_proba_dict.items(), colors):
            if len(classes) == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, lw=2, 
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
            else:
                # Multi-class: plot macro-average
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(len(classes)):
                    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Compute macro-average
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(len(classes)):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= len(classes)
                
                macro_auc = auc(all_fpr, mean_tpr)
                plt.plot(all_fpr, mean_tpr, color=color, lw=2,
                        label=f'{model_name} (Macro AUC = {macro_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ ROC curves saved to {save_path}")
        
        plt.show()

