"""
Professional Visualization Module for Medical NLP Project
Creates publication-quality charts and graphs for M.Tech evaluation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from typing import List, Dict, Optional
import matplotlib.patches as mpatches

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


class MedicalNLPVisualizer:
    """
    Professional visualization class for medical NLP project.
    """
    
    def __init__(self, style='professional'):
        """Initialize visualizer with style settings."""
        self.style = style
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#06A77D',
            'warning': '#F18F01',
            'danger': '#C73E1D',
            'info': '#6A4C93'
        }
    
    def plot_pipeline_flowchart(self, save_path: Optional[str] = None):
        """
        Create a visual flowchart of the NLP pipeline.
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')
        
        # Define pipeline stages
        stages = [
            "Raw Medical Text",
            "Preprocessing",
            "Anonymization",
            "Feature Extraction",
            "Classification",
            "Results & Insights"
        ]
        
        colors = ['#E8F4F8', '#B8E6F0', '#88D8E8', '#58CAE0', '#28BCD8', '#00AED0']
        
        y_positions = np.linspace(0.9, 0.1, len(stages))
        
        for i, (stage, color, y_pos) in enumerate(zip(stages, colors, y_positions)):
            # Draw box
            box = mpatches.FancyBboxPatch(
                (0.2, y_pos - 0.05), 0.6, 0.08,
                boxstyle="round,pad=0.01",
                facecolor=color,
                edgecolor='#333',
                linewidth=2
            )
            ax.add_patch(box)
            
            # Add text
            ax.text(0.5, y_pos, stage, 
                   ha='center', va='center',
                   fontsize=14, fontweight='bold',
                   color='#333')
            
            # Add arrow
            if i < len(stages) - 1:
                ax.arrow(0.5, y_pos - 0.06, 0, -0.06,
                        head_width=0.03, head_length=0.02,
                        fc='#666', ec='#666', linewidth=2)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Medical NLP Pipeline Architecture', 
                    fontsize=18, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def plot_model_comparison(self, results: Dict[str, float], 
                             save_path: Optional[str] = None):
        """
        Create professional model comparison chart.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        models = list(results.keys())
        accuracies = list(results.values())
        
        # Bar chart
        bars = ax1.bar(models, accuracies, 
                      color=[self.colors['primary'], self.colors['secondary'], 
                             self.colors['success']],
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax1.set_ylabel('Accuracy (%)', fontweight='bold')
        ax1.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(models), endpoint=False).tolist()
        accuracies_radar = accuracies + [accuracies[0]]
        angles += angles[:1]
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, accuracies_radar, 'o-', linewidth=2, 
                color=self.colors['primary'], label='Accuracy')
        ax2.fill(angles, accuracies_radar, alpha=0.25, color=self.colors['primary'])
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(models, fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.set_title('Model Performance Radar', fontweight='bold', 
                     fontsize=14, pad=20)
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    def plot_preprocessing_steps(self, original_text: str, processed_text: str,
                                save_path: Optional[str] = None):
        """
        Visualize the preprocessing transformation.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        # Original text
        ax1.text(0.5, 0.5, original_text,
                ha='center', va='center', wrap=True,
                fontsize=11, bbox=dict(boxstyle='round',
                facecolor='#FFE5E5', edgecolor='#C73E1D', linewidth=2))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('Original Medical Text', fontweight='bold',
                     fontsize=14, color='#C73E1D')

        # Processed text
        ax2.text(0.5, 0.5, processed_text,
                ha='center', va='center', wrap=True,
                fontsize=11, bbox=dict(boxstyle='round',
                facecolor='#E5FFE5', edgecolor='#06A77D', linewidth=2))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Preprocessed Text', fontweight='bold',
                     fontsize=14, color='#06A77D')

        plt.suptitle('Text Preprocessing Transformation',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    def plot_word_frequency(self, word_freq: Dict[str, int], top_n: int = 20,
                           title: str = "Top Medical Terms",
                           save_path: Optional[str] = None):
        """
        Plot word frequency distribution.
        """
        # Sort and get top N
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        words, freqs = zip(*sorted_words)

        fig, ax = plt.subplots(figsize=(14, 8))

        # Create horizontal bar chart
        bars = ax.barh(range(len(words)), freqs,
                      color=plt.cm.viridis(np.linspace(0.3, 0.9, len(words))),
                      edgecolor='black', linewidth=1)

        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontweight='bold')
        ax.set_xlabel('Frequency', fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=16, pad=15)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, freq) in enumerate(zip(bars, freqs)):
            ax.text(freq + max(freqs)*0.01, i, str(freq),
                   va='center', fontweight='bold', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    def create_medical_wordcloud(self, text: str,
                                title: str = "Medical Terms Word Cloud",
                                save_path: Optional[str] = None):
        """
        Create a professional word cloud for medical text.
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        wordcloud = WordCloud(
            width=1200, height=600,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10,
            contour_width=2,
            contour_color='steelblue'
        ).generate(text)

        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontweight='bold', fontsize=18, pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    def plot_metrics_dashboard(self, metrics: Dict[str, Dict[str, float]],
                              save_path: Optional[str] = None):
        """
        Create a comprehensive metrics dashboard.
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle('Model Performance Dashboard',
                    fontsize=20, fontweight='bold', y=0.98)

        # Extract metrics
        models = list(metrics.keys())
        accuracy = [metrics[m]['accuracy'] for m in models]
        precision = [metrics[m]['precision'] for m in models]
        recall = [metrics[m]['recall'] for m in models]
        f1 = [metrics[m]['f1_score'] for m in models]

        # Plot 1: Accuracy comparison
        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(len(models))
        width = 0.2

        ax1.bar(x - 1.5*width, accuracy, width, label='Accuracy', color=self.colors['primary'])
        ax1.bar(x - 0.5*width, precision, width, label='Precision', color=self.colors['secondary'])
        ax1.bar(x + 0.5*width, recall, width, label='Recall', color=self.colors['success'])
        ax1.bar(x + 1.5*width, f1, width, label='F1-Score', color=self.colors['warning'])

        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('Performance Metrics Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, model in enumerate(models):
            for j, (metric, offset) in enumerate([
                (accuracy[i], -1.5*width),
                (precision[i], -0.5*width),
                (recall[i], 0.5*width),
                (f1[i], 1.5*width)
            ]):
                ax1.text(i + offset, metric + 0.02, f'{metric:.2f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()


if __name__ == "__main__":
    # Demo visualizations
    viz = MedicalNLPVisualizer()

    # Example: Model comparison
    results = {
        'Bag of Words': 78.5,
        'TF-IDF': 83.2,
        'Word Embeddings': 87.6
    }

    viz.plot_model_comparison(results, save_path='../outputs/visualizations/model_comparison.png')

    print("Visualizations created successfully!")

    
    def plot_confusion_matrix_enhanced(self, cm: np.ndarray, 
                                      labels: List[str],
                                      title: str = "Confusion Matrix",
                                      save_path: Optional[str] = None):
        """
        Create enhanced confusion matrix visualization.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'}, ax=ax,
                   linewidths=1, linecolor='gray')
        
        # Add percentage annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)',
                             ha="center", va="center", color="darkblue", 
                             fontsize=9, style='italic')
        
        ax.set_title(title, fontweight='bold', fontsize=16, pad=15)
        ax.set_ylabel('True Label', fontweight='bold', fontsize=12)
        ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

