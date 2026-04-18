# 🏥 Medical NLP Project
## Electronic Health Record Analysis using Natural Language Processing

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)
![NLP](https://img.shields.io/badge/NLP-Healthcare-red.svg)

**M.Tech Project - Natural Language Processing for Clinical Research**

*Based on: "Natural language processing techniques applied to the electronic health record in clinical research and practice" by Clay et al. (2025)*

</div>

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results & Visualizations](#results--visualizations)
- [Academic Context](#academic-context)
- [References](#references)

---

## 🎯 Overview

This project implements state-of-the-art **Natural Language Processing (NLP)** techniques for analyzing **Electronic Health Records (EHR)**. It demonstrates the complete pipeline from raw medical text to actionable insights, including preprocessing, anonymization, feature extraction, and classification.

### 🎓 Academic Context
- **Program**: M.Tech in Computer Science / Data Science
- **Domain**: Healthcare Informatics & Natural Language Processing
- **Research Focus**: Automated analysis of clinical documentation
- **Implementation**: Python-based end-to-end NLP pipeline

## 📁 Project Structure
```
📦 medical-nlp-project/
│
├── 📂 data/
│   ├── 📁 raw/                          # Raw datasets from Kaggle
│   ├── 📁 processed/                    # Processed and cleaned data
│   └── 📁 sample/                       # Sample medical texts for testing
│
├── 📂 src/                              # 🔧 Source Code
│   ├── 📄 preprocessing.py              # Text preprocessing pipeline
│   ├── 📄 anonymization.py              # Patient data anonymization
│   ├── 📄 bow_model.py                  # Bag of Words implementation
│   ├── 📄 tfidf_model.py                # TF-IDF implementation
│   ├── 📄 embedding_model.py            # Neural network embeddings
│   ├── 📄 classification.py             # Classification models
│   └── 📄 utils.py                      # Helper functions
│
├── 📂 models/
│   └── 📁 saved_models/                 # 💾 Trained model storage
│
├── 📂 notebooks/                        # 📓 Jupyter Notebooks
│   ├── 01_data_exploration.ipynb        # Data exploration
│   ├── 02_preprocessing_demo.ipynb      # Preprocessing demonstration
│   ├── 03_bow_tfidf_analysis.ipynb      # BoW and TF-IDF analysis
│   ├── 04_embeddings_demo.ipynb         # Word embeddings demonstration
│   └── 05_full_pipeline.ipynb           # Complete NLP pipeline
│
├── 📂 outputs/                          # 📊 Results
│   ├── 📁 visualizations/               # Generated plots and charts
│   └── 📁 results/                      # Analysis results
│
├── 📂 presentation/                     # 🎤 Project Presentation
│   ├── slides.pdf                       # Presentation slides
│   └── demo_screenshots/                # Screenshots for documentation
│
├── 📄 requirements.txt                  # Python dependencies
├── 📄 setup_data.py                     # Script to download datasets
├── 📄 main.py                           # Main execution script
├── 📄 README.md                         # This file
├── 📄 QUICKSTART.md                     # Quick start guide
└── 📄 INSTALLATION_GUIDE.md             # Detailed installation instructions
```

## ✨ Key Features

### 🔄 1. Text Preprocessing Pipeline
<details>
<summary>Click to expand preprocessing capabilities</summary>

- ✅ **Sentence Segmentation** - Intelligent splitting of medical text
- ✅ **Word Tokenization** - Breaking text into meaningful units
- ✅ **Contraction Expansion** - "don't" → "do not"
- ✅ **Case Normalization** - Standardizing text format
- ✅ **Stopword Removal** - Filtering common words
- ✅ **Stemming & Lemmatization** - Reducing words to root forms
- ✅ **POS Tagging** - Grammatical role identification

**Example:**
```
Input:  "Patient's experiencing severe chest pain and can't breathe properly."
Output: "patient experience severe chest pain breathe properly"
```
</details>

### 🔒 2. Privacy Protection (Anonymization)
<details>
<summary>Click to expand anonymization features</summary>

- 🛡️ **Patient Names** - Automatic detection and removal
- 🛡️ **Dates of Birth** - Pattern-based identification
- 🛡️ **Hospital IDs & MRNs** - Medical record number masking
- 🛡️ **Contact Information** - Phone, email, address removal
- 🛡️ **HIPAA Compliance** - Protected Health Information (PHI) safeguarding

**Example:**
```
Input:  "Patient: John Smith, DOB: 03/15/1975, MRN: 12345678"
Output: "Patient: [REDACTED_NAME], DOB: [REDACTED_DATE], MRN: [REDACTED_ID]"
```
</details>

### 🤖 3. NLP Models & Algorithms
<details>
<summary>Click to expand model details</summary>

| Model | Description | Use Case | Accuracy |
|-------|-------------|----------|----------|
| **Bag of Words (BoW)** | Word frequency counting | Baseline classification | ~75-80% |
| **TF-IDF** | Weighted word importance | Advanced classification | ~80-85% |
| **Word Embeddings** | Neural network representation | Semantic understanding | ~85-90% |

</details>

### 📊 4. Classification & Analysis Tasks
<details>
<summary>Click to expand classification capabilities</summary>

- 🏥 **Medical Condition Classification** - Automatic disease categorization
- 🚨 **Urgency Level Prediction** - Emergency/Routine/Urgent triage
- 🔬 **Disease Category Identification** - Multi-class medical classification
- 💬 **Sentiment Analysis** - Patient feedback analysis
- 📈 **Trend Detection** - Identifying patterns in clinical notes

</details>

## Datasets

### 1. Healthcare Dataset (Kaggle)
- **Source**: https://www.kaggle.com/datasets/prasad22/healthcare-dataset
- **Size**: 10,000 synthetic patient records
- **Features**: Patient demographics, medical conditions, medications, test results

### 2. Medical Text Dataset (Kaggle)
- **Source**: https://www.kaggle.com/datasets/chaitanyakck/medical-text
- **Size**: 14,438 training records, 14,442 test records
- **Categories**: Digestive, Cardiovascular, Neoplasms, Nervous System, General Pathological

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or navigate to the project directory**
```bash
cd "C:\Users\PGanesan\OneDrive - Ashley Furniture Industries, Inc\Desktop\Augment Project"
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data**
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
```

4. **Download datasets**
```bash
python setup_data.py
```

## Usage

### Quick Start
```python
from src.preprocessing import preprocess_text
from src.bow_model import BagOfWordsModel

# Preprocess medical text
text = "Patient presents with chest pain and shortness of breath."
processed = preprocess_text(text)

# Create Bag of Words model
bow = BagOfWordsModel()
bow.fit(processed)
```

### Run Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

## Technologies Used
- **Python 3.8+**
- **pandas**: Data manipulation
- **NLTK**: Natural Language Toolkit
- **scikit-learn**: Machine learning models
- **TensorFlow/Keras**: Neural networks
- **spaCy**: Advanced NLP processing
- **matplotlib/seaborn**: Visualization
- **Jupyter**: Interactive notebooks

## License
This project is for educational purposes based on publicly available research and datasets.

## Acknowledgments
- Research paper: "Natural language processing techniques applied to the electronic health record in clinical research and practice" by Clay et al. (2025)
- Kaggle datasets: Healthcare Dataset and Medical Text Dataset
- Open-source NLP libraries: NLTK, spaCy, scikit-learn

## Contact
For questions or contributions, please open an issue in the repository.

