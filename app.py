"""
Interactive Medical NLP Dashboard
Upload CSV data, process it, and view results in real-time
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
sys.path.append('src')

from preprocessing import MedicalTextPreprocessor, preprocess_text
from anonymization import MedicalTextAnonymizer, anonymize_text
from bow_model import BagOfWordsModel
from tfidf_model import TFIDFModel
from embedding_model import MedicalEmbeddingModel
from utils import ensure_dir
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import io

# Page configuration
st.set_page_config(
    page_title="Medical NLP Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #667eea;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #764ba2;
        margin-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'bow_model' not in st.session_state:
    st.session_state.bow_model = None
if 'tfidf_model' not in st.session_state:
    st.session_state.tfidf_model = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None

# Header
st.markdown('<h1 class="main-header">🏥 Medical NLP Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Upload medical data, process with NLP, and analyze results</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/medical-heart.png", width=100)
    st.title("📋 Navigation")
    
    page = st.radio(
        "Select Page:",
        ["🏠 Home", "📤 Upload Data", "🔄 Process Data", "📊 View Results", "🧠 Word Embeddings", "💾 Download Results"]
    )
    
    st.markdown("---")
    st.markdown("### 📚 About")
    st.info("""
    This dashboard implements NLP techniques for Electronic Health Records based on Clay et al. (2025).
    
    **Features:**
    - Text Preprocessing
    - PHI Anonymization
    - Bag of Words
    - TF-IDF Analysis
    """)

# HOME PAGE
if page == "🏠 Home":
    st.markdown('<h2 class="sub-header">Welcome to Medical NLP Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📤</h3>
            <h4>Upload Data</h4>
            <p>CSV files from Kaggle</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔄</h3>
            <h4>Process</h4>
            <p>NLP Pipeline</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊</h3>
            <h4>Analyze</h4>
            <p>View Results</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>💾</h3>
            <h4>Download</h4>
            <p>Export Results</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    1. **Upload Data**: Go to 'Upload Data' and upload your CSV file
    2. **Select Text Column**: Choose which column contains medical text
    3. **Process**: Click 'Process Data' to run the NLP pipeline
    4. **View Results**: See preprocessing, anonymization, and model results
    5. **Download**: Export processed data as CSV
    """)
    
    st.markdown("### 📊 Sample Data Format")
    sample_df = pd.DataFrame({
        'Patient_ID': [1, 2, 3],
        'Clinical_Notes': [
            'Patient presents with chest pain and shortness of breath',
            'Diabetes mellitus type 2, requires insulin adjustment',
            'Routine checkup, patient is healthy and stable'
        ],
        'Diagnosis': ['Cardiac', 'Endocrine', 'Routine']
    })
    st.dataframe(sample_df, use_container_width=True)

# UPLOAD DATA PAGE
elif page == "📤 Upload Data":
    st.markdown('<h2 class="sub-header">Upload Your Medical Data</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing medical text data"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.original_data = df
            
            st.success(f"✅ File uploaded successfully! {len(df)} rows loaded.")
            
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("### 📊 Data Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            st.markdown("### 🔍 Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(col_info, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    else:
        st.info("👆 Please upload a CSV file to get started")

# PROCESS DATA PAGE
elif page == "🔄 Process Data":
    st.markdown('<h2 class="sub-header">Process Your Data</h2>', unsafe_allow_html=True)
    
    if st.session_state.original_data is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.original_data
        
        st.markdown("### ⚙️ Processing Configuration")
        
        # Select text column
        text_column = st.selectbox(
            "Select the column containing medical text:",
            options=df.columns.tolist(),
            help="Choose the column that contains the clinical notes or medical text"
        )

        # Processing options
        col1, col2 = st.columns(2)

        with col1:
            remove_stopwords = st.checkbox("Remove Stopwords", value=True)
            use_lemmatization = st.checkbox("Use Lemmatization", value=True)

        with col2:
            anonymize_data = st.checkbox("Anonymize PHI/PII", value=True)
            run_models = st.checkbox("Run BoW & TF-IDF Models", value=True)

        # Label column (optional)
        label_column = st.selectbox(
            "Select label column (optional, for classification):",
            options=['None'] + df.columns.tolist(),
            help="Choose a column for classification labels"
        )

        st.markdown("---")

        # Process button
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            with st.spinner("Processing data... This may take a few minutes..."):
                try:
                    # Initialize preprocessor and anonymizer
                    preprocessor = MedicalTextPreprocessor(
                        remove_stopwords=remove_stopwords,
                        use_lemmatization=use_lemmatization
                    )
                    anonymizer = MedicalTextAnonymizer(replacement_strategy='mask')

                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Step 1: Preprocessing
                    status_text.text("Step 1/4: Preprocessing text...")
                    df['preprocessed_text'] = df[text_column].apply(
                        lambda x: preprocessor.preprocess(str(x), return_tokens=False) if pd.notna(x) else ""
                    )
                    progress_bar.progress(25)

                    # Step 2: Anonymization
                    if anonymize_data:
                        status_text.text("Step 2/4: Anonymizing PHI/PII...")
                        df['anonymized_text'] = df[text_column].apply(
                            lambda x: anonymizer.anonymize(str(x)) if pd.notna(x) else ""
                        )
                    progress_bar.progress(50)

                    # Step 3: Run models
                    if run_models and label_column != 'None':
                        status_text.text("Step 3/4: Training BoW model...")

                        # Prepare data
                        X = df['preprocessed_text'].tolist()
                        y = df[label_column].tolist()

                        # BoW Model
                        bow_model = BagOfWordsModel()
                        bow_model.fit(X, y)
                        st.session_state.bow_model = bow_model

                        progress_bar.progress(75)

                        status_text.text("Step 4/4: Training TF-IDF model...")

                        # TF-IDF Model
                        tfidf_model = TFIDFModel()
                        tfidf_model.fit(X, y)
                        st.session_state.tfidf_model = tfidf_model

                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")

                    # Save processed data
                    st.session_state.processed_data = df

                    st.success("🎉 Data processed successfully!")

                    # Show sample results
                    st.markdown("### 📋 Sample Results")
                    st.dataframe(df.head(5), use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# VIEW RESULTS PAGE
elif page == "📊 View Results":
    st.markdown('<h2 class="sub-header">Analysis Results</h2>', unsafe_allow_html=True)

    if st.session_state.processed_data is None:
        st.warning("⚠️ Please process data first!")
    else:
        df = st.session_state.processed_data

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔄 Preprocessing", "🔒 Anonymization", "🤖 Models"])

        with tab1:
            st.markdown("### 📊 Data Overview")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Processed Records", df['preprocessed_text'].notna().sum())
            with col3:
                if 'anonymized_text' in df.columns:
                    st.metric("Anonymized Records", df['anonymized_text'].notna().sum())
                else:
                    st.metric("Anonymized Records", "N/A")
            with col4:
                avg_length = df['preprocessed_text'].str.len().mean()
                st.metric("Avg Text Length", f"{avg_length:.0f}")

            st.markdown("### 📋 Processed Data")
            st.dataframe(df, use_container_width=True)

        with tab2:
            st.markdown("### 🔄 Preprocessing Results")

            # Select a row to view
            row_idx = st.number_input("Select row to view:", min_value=0, max_value=len(df)-1, value=0)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Original Text")
                original_col = [col for col in df.columns if 'text' in col.lower() or 'note' in col.lower()][0]
                st.markdown(f'<div class="warning-box">{df.iloc[row_idx][original_col]}</div>', unsafe_allow_html=True)

            with col2:
                st.markdown("#### Preprocessed Text")
                st.markdown(f'<div class="success-box">{df.iloc[row_idx]["preprocessed_text"]}</div>', unsafe_allow_html=True)

            # Word cloud
            st.markdown("### ☁️ Word Cloud")
            try:
                text = ' '.join(df['preprocessed_text'].dropna().tolist())
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating word cloud: {str(e)}")

        with tab3:
            st.markdown("### 🔒 Anonymization Results")

            if 'anonymized_text' in df.columns:
                row_idx = st.number_input("Select row to view anonymization:", min_value=0, max_value=len(df)-1, value=0, key='anon_row')

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Before Anonymization")
                    original_col = [col for col in df.columns if 'text' in col.lower() or 'note' in col.lower()][0]
                    st.markdown(f'<div class="warning-box">{df.iloc[row_idx][original_col]}</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown("#### After Anonymization")
                    st.markdown(f'<div class="success-box">{df.iloc[row_idx]["anonymized_text"]}</div>', unsafe_allow_html=True)

                # Count redactions
                redaction_count = df['anonymized_text'].str.count(r'\[REDACTED_').sum()
                st.metric("Total PHI/PII Redactions", int(redaction_count))
            else:
                st.info("Anonymization was not performed on this dataset")

        with tab4:
            st.markdown("### 🤖 Model Results")

            if st.session_state.bow_model is not None:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Bag of Words Model")
                    bow_model = st.session_state.bow_model

                    st.metric("Vocabulary Size", len(bow_model.vectorizer.vocabulary_))
                    st.metric("Feature Matrix Shape", str(bow_model.X.shape))

                    if hasattr(bow_model, 'accuracy'):
                        st.metric("Accuracy", f"{bow_model.accuracy*100:.2f}%")

                    # Top features
                    st.markdown("**Top 10 Features:**")
                    feature_names = bow_model.vectorizer.get_feature_names_out()
                    st.write(feature_names[:10].tolist())

                with col2:
                    st.markdown("#### TF-IDF Model")
                    tfidf_model = st.session_state.tfidf_model

                    st.metric("Vocabulary Size", len(tfidf_model.vectorizer.vocabulary_))
                    st.metric("Feature Matrix Shape", str(tfidf_model.X.shape))

                    if hasattr(tfidf_model, 'accuracy'):
                        st.metric("Accuracy", f"{tfidf_model.accuracy*100:.2f}%")

                    # Top features
                    st.markdown("**Top 10 Features:**")
                    feature_names = tfidf_model.vectorizer.get_feature_names_out()
                    st.write(feature_names[:10].tolist())
            else:
                st.info("Models were not trained. Please process data with labels to train models.")

# WORD EMBEDDINGS PAGE
elif page == "🧠 Word Embeddings":
    st.markdown('<h2 class="sub-header">Word2Vec Embeddings Explorer</h2>', unsafe_allow_html=True)

    st.markdown("""
    ### 🧠 What are Word Embeddings?

    Word embeddings convert words into dense numerical vectors that capture semantic meaning.
    Similar words have similar vectors, allowing the model to understand relationships between medical terms.

    **Example:** "chest pain" and "cardiac discomfort" would have similar vectors even though they use different words.
    """)

    if st.session_state.processed_data is None:
        st.warning("⚠️ Please process data first to train Word2Vec model!")
    else:
        df = st.session_state.processed_data

        # Check if we have preprocessed text
        if 'preprocessed' in df.columns:
            texts = df['preprocessed'].tolist()

            st.markdown("### 🔄 Train Word2Vec Model")

            col1, col2, col3 = st.columns(3)
            with col1:
                vector_size = st.slider("Vector Size", 50, 200, 100, 50)
            with col2:
                window_size = st.slider("Window Size", 3, 10, 5)
            with col3:
                epochs = st.slider("Epochs", 5, 50, 20, 5)

            if st.button("🚀 Train Word2Vec Model", type="primary", use_container_width=True):
                with st.spinner("Training Word2Vec model..."):
                    try:
                        embedding_model = MedicalEmbeddingModel(
                            vector_size=vector_size,
                            window=window_size,
                            epochs=epochs
                        )
                        doc_vectors = embedding_model.fit_transform(texts)
                        st.session_state.embedding_model = embedding_model

                        st.success(f"✅ Model trained successfully! Document vectors shape: {doc_vectors.shape}")
                    except Exception as e:
                        st.error(f"❌ Error training model: {e}")

            # Word Similarity Explorer
            if st.session_state.embedding_model is not None:
                st.markdown("---")
                st.markdown("### 🔍 Word Similarity Explorer")

                embedding_model = st.session_state.embedding_model

                # Get all words in vocabulary
                vocab = list(embedding_model.model.wv.key_to_index.keys())

                st.info(f"📚 Vocabulary size: {len(vocab)} unique words")

                col1, col2 = st.columns([2, 1])

                with col1:
                    search_word = st.text_input(
                        "Enter a medical term to find similar words:",
                        placeholder="e.g., pain, diabetes, chest, patient"
                    )

                with col2:
                    top_n = st.slider("Number of similar words", 3, 20, 10)

                if search_word:
                    similar_words = embedding_model.get_similar_words(search_word.lower(), top_n=top_n)

                    if similar_words:
                        st.markdown(f"#### Words similar to **'{search_word}'**:")

                        # Create a dataframe for better display
                        similar_df = pd.DataFrame(similar_words, columns=['Word', 'Similarity Score'])
                        similar_df['Similarity Score'] = similar_df['Similarity Score'].round(3)

                        # Display as a nice table
                        st.dataframe(similar_df, use_container_width=True, hide_index=True)

                        # Visualize similarity scores
                        fig, ax = plt.subplots(figsize=(10, 6))
                        words = [w[0] for w in similar_words]
                        scores = [w[1] for w in similar_words]

                        bars = ax.barh(words, scores, color='#667eea')
                        ax.set_xlabel('Similarity Score', fontsize=12)
                        ax.set_title(f'Words Similar to "{search_word}"', fontsize=14, fontweight='bold')
                        ax.set_xlim(0, 1)

                        # Add value labels
                        for i, (bar, score) in enumerate(zip(bars, scores)):
                            ax.text(score + 0.01, i, f'{score:.3f}', va='center')

                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.warning(f"⚠️ Word '{search_word}' not found in vocabulary. Try another word.")

                        # Show some example words
                        st.markdown("**Try these words from the vocabulary:**")
                        sample_words = vocab[:20] if len(vocab) > 20 else vocab
                        st.write(", ".join(sample_words))

                # Vocabulary Browser
                st.markdown("---")
                st.markdown("### 📖 Vocabulary Browser")

                with st.expander("View all words in vocabulary"):
                    # Show vocabulary in columns
                    vocab_sorted = sorted(vocab)
                    num_cols = 5
                    cols = st.columns(num_cols)

                    for i, word in enumerate(vocab_sorted):
                        cols[i % num_cols].write(f"• {word}")

                # Model Statistics
                st.markdown("---")
                st.markdown("### 📊 Model Statistics")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Vocabulary Size", len(vocab))

                with col2:
                    st.metric("Vector Dimensions", embedding_model.vector_size)

                with col3:
                    st.metric("Training Epochs", embedding_model.epochs)
        else:
            st.warning("⚠️ No preprocessed text found. Please process data first!")

# DOWNLOAD RESULTS PAGE
elif page == "💾 Download Results":
    st.markdown('<h2 class="sub-header">Download Processed Data</h2>', unsafe_allow_html=True)

    if st.session_state.processed_data is None:
        st.warning("⚠️ Please process data first!")
    else:
        df = st.session_state.processed_data

        st.markdown("### 📥 Download Options")

        # Select columns to download
        columns_to_download = st.multiselect(
            "Select columns to include in download:",
            options=df.columns.tolist(),
            default=df.columns.tolist()
        )

        if columns_to_download:
            download_df = df[columns_to_download]

            # Preview
            st.markdown("### 👀 Preview")
            st.dataframe(download_df.head(10), use_container_width=True)

            # Convert to CSV
            csv = download_df.to_csv(index=False)

            # Download button
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="processed_medical_data.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

            st.success(f"✅ Ready to download {len(download_df)} rows with {len(columns_to_download)} columns")
        else:
            st.warning("Please select at least one column to download")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🏥 Medical NLP Dashboard | Based on Clay et al. (2025)</p>
    <p>M.Tech Project - Natural Language Processing for Electronic Health Records</p>
</div>
""", unsafe_allow_html=True)

