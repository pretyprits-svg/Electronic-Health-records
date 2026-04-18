"""
BioBERT Deep Learning Model for Medical Text Classification
Uses pre-trained BioBERT transformer model for state-of-the-art medical NLP
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class BioBERTClassifier:
    """
    BioBERT-based classifier for medical text.
    Uses pre-trained BioBERT model fine-tuned for classification.
    """
    
    def __init__(self, model_name: str = "dmis-lab/biobert-v1.1", 
                 max_length: int = 128, device: str = None):
        """
        Initialize BioBERT classifier.
        
        Args:
            model_name (str): Pre-trained model name from HuggingFace
            max_length (int): Maximum sequence length
            device (str): 'cuda' or 'cpu' (auto-detected if None)
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"🔧 Using device: {self.device}")
        
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"✅ Loaded tokenizer: {model_name}")
        except Exception as e:
            print(f"⚠️  Could not load {model_name}, using BERT base instead")
            self.model_name = "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.num_labels = None
        self.is_fitted = False
    
    def encode_texts(self, texts: List[str]) -> Dict:
        """
        Tokenize and encode texts for BERT.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            dict: Encoded inputs
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return encodings
    
    def fit(self, X_train: List[str], y_train: List[str], 
            X_val: Optional[List[str]] = None, y_val: Optional[List[str]] = None,
            epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-5):
        """
        Fine-tune BioBERT on training data.
        
        Args:
            X_train (list): Training texts
            y_train (list): Training labels
            X_val (list, optional): Validation texts
            y_val (list, optional): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
        """
        print(f"🔄 Fine-tuning BioBERT...")
        print(f"   Training samples: {len(X_train)}")
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.num_labels = len(self.label_encoder.classes_)
        
        print(f"   Number of classes: {self.num_labels}")
        print(f"   Classes: {list(self.label_encoder.classes_)}")
        
        # Load model for classification
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            ).to(self.device)
            print(f"✅ Loaded model: {self.model_name}")
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print(f"   Using simplified training approach...")
            self._simple_fit(X_train, y_train_encoded, epochs, batch_size, learning_rate)
            return self
        
        # Encode texts
        train_encodings = self.encode_texts(X_train)
        
        # Create dataset
        train_dataset = MedicalDataset(train_encodings, y_train_encoded)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./models/biobert_checkpoints',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_strategy='no',
            report_to='none'
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset
        )
        
        # Train
        print(f"🚀 Starting training for {epochs} epochs...")
        trainer.train()
        
        self.is_fitted = True
        print("✅ Training complete!")
        
        return self
    
    def _simple_fit(self, X_train, y_train, epochs, batch_size, learning_rate):
        """Simplified training without Trainer API."""
        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        
        # Load base model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        ).to(self.device)
        
        # Prepare data
        encodings = self.encode_texts(X_train)
        dataset = MedicalDataset(encodings, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"   Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        self.is_fitted = True

    def predict(self, X_test: List[str]) -> np.ndarray:
        """
        Predict labels for test data.

        Args:
            X_test (list): Test texts

        Returns:
            np.ndarray: Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()

        # Encode texts
        encodings = self.encode_texts(X_test)

        # Predict
        with torch.no_grad():
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)

            outputs = self.model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

        # Decode labels
        predictions_np = predictions.cpu().numpy()
        predicted_labels = self.label_encoder.inverse_transform(predictions_np)

        return predicted_labels

    def predict_proba(self, X_test: List[str]) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X_test (list): Test texts

        Returns:
            np.ndarray: Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()

        encodings = self.encode_texts(X_test)

        with torch.no_grad():
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)

            outputs = self.model(input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        return probabilities.cpu().numpy()

    def evaluate(self, X_test: List[str], y_test: List[str]) -> Dict:
        """
        Evaluate model on test data.

        Args:
            X_test (list): Test texts
            y_test (list): True labels

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

    def save_model(self, filepath: str):
        """Save model to disk."""
        if self.model is not None:
            self.model.save_pretrained(filepath)
            self.tokenizer.save_pretrained(filepath)
            print(f"✅ Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from disk."""
        self.model = AutoModelForSequenceClassification.from_pretrained(filepath).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(filepath)
        self.is_fitted = True
        print(f"✅ Model loaded from {filepath}")


class MedicalDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for medical text classification."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Demo
if __name__ == "__main__":
    print("="*70)
    print("BioBERT MEDICAL TEXT CLASSIFIER - DEMO")
    print("="*70)

    # Sample data
    texts = [
        "patient severe chest pain dyspnea",
        "acute myocardial infarction elevated troponin",
        "type diabetes mellitus high glucose",
        "patient persistent headache visual disturbance",
        "fractured femur surgical intervention"
    ] * 5

    labels = ['Cardiac', 'Cardiac', 'Endocrine', 'Neurological', 'Orthopedic'] * 5

    print(f"\n📊 Sample data: {len(texts)} texts")
    print(f"📊 Classes: {set(labels)}")

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )

    # Initialize and train
    print("\n🔧 Initializing BioBERT...")
    classifier = BioBERTClassifier(max_length=64)

    print("\n🚀 Training model...")
    classifier.fit(X_train, y_train, epochs=2, batch_size=4)

    # Evaluate
    print("\n📊 Evaluating...")
    results = classifier.evaluate(X_test, y_test)
    print(f"\n✅ Accuracy: {results['accuracy']:.2%}")

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
