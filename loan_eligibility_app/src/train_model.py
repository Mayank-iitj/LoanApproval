"""
Model training and evaluation for Loan Eligibility Prediction
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class LoanEligibilityModel:
    """
    Loan Eligibility Prediction Model with training and evaluation
    """
    
    def __init__(self, model_type='RandomForest', random_state=42):
        """
        Initialize the model
        
        Args:
            model_type: Type of model ('LogisticRegression' or 'RandomForest')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.preprocessor = None
        self.metrics = {}
        
        # Initialize model based on type
        if model_type == 'LogisticRegression':
            self.model = LogisticRegression(max_iter=1000, random_state=random_state)
        elif model_type == 'RandomForest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Initialized {model_type} model")
    
    def train(self, X_train, y_train, preprocessor):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training target
            preprocessor: Fitted preprocessor
        """
        self.preprocessor = preprocessor
        
        # Transform training data
        X_train_transformed = preprocessor.transform(X_train)
        
        # Train model
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train_transformed, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_transformed, y_train, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.metrics['cv_mean'] = cv_scores.mean()
        self.metrics['cv_std'] = cv_scores.std()
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Transform test data
        X_test_transformed = self.preprocessor.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_transformed)
        y_pred_proba = self.model.predict_proba(X_test_transformed)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.metrics.update(metrics)
        
        # Log metrics
        logger.info(f"\n{self.model_type} Evaluation Metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions (0 or 1)
        """
        X_transformed = self.preprocessor.transform(X)
        return self.model.predict(X_transformed)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Features to predict
            
        Returns:
            Probability of positive class
        """
        X_transformed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_transformed)[:, 1]
    
    def save(self, model_path: Path, preprocessor_path: Path):
        """
        Save model and preprocessor
        
        Args:
            model_path: Path to save model
            preprocessor_path: Path to save preprocessor
        """
        joblib.dump(self.model, model_path)
        joblib.dump(self.preprocessor, preprocessor_path)
        logger.info(f"Saved model to {model_path}")
        logger.info(f"Saved preprocessor to {preprocessor_path}")
    
    @staticmethod
    def load(model_path: Path, preprocessor_path: Path):
        """
        Load model and preprocessor
        
        Args:
            model_path: Path to model file
            preprocessor_path: Path to preprocessor file
            
        Returns:
            Loaded model instance
        """
        model_instance = LoanEligibilityModel()
        model_instance.model = joblib.load(model_path)
        model_instance.preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Loaded model from {model_path}")
        return model_instance


def train_and_compare_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train and compare multiple models
    
    Args:
        X_train, X_test: Train and test features
        y_train, y_test: Train and test targets
        preprocessor: Fitted preprocessor
        
    Returns:
        Dictionary with trained models and their metrics
    """
    models = {}
    
    # Train Logistic Regression
    logger.info("\n" + "="*50)
    logger.info("Training Logistic Regression")
    logger.info("="*50)
    lr_model = LoanEligibilityModel('LogisticRegression')
    lr_model.train(X_train, y_train, preprocessor)
    lr_metrics = lr_model.evaluate(X_test, y_test)
    models['Logistic Regression'] = {
        'model': lr_model,
        'metrics': lr_metrics
    }
    
    # Train Random Forest
    logger.info("\n" + "="*50)
    logger.info("Training Random Forest")
    logger.info("="*50)
    rf_model = LoanEligibilityModel('RandomForest')
    rf_model.train(X_train, y_train, preprocessor)
    rf_metrics = rf_model.evaluate(X_test, y_test)
    models['Random Forest'] = {
        'model': rf_model,
        'metrics': rf_metrics
    }
    
    # Compare models
    logger.info("\n" + "="*50)
    logger.info("Model Comparison")
    logger.info("="*50)
    
    comparison_df = pd.DataFrame({
        'Logistic Regression': {
            'Accuracy': lr_metrics['accuracy'],
            'Precision': lr_metrics['precision'],
            'Recall': lr_metrics['recall'],
            'F1-Score': lr_metrics['f1_score'],
            'ROC AUC': lr_metrics['roc_auc']
        },
        'Random Forest': {
            'Accuracy': rf_metrics['accuracy'],
            'Precision': rf_metrics['precision'],
            'Recall': rf_metrics['recall'],
            'F1-Score': rf_metrics['f1_score'],
            'ROC AUC': rf_metrics['roc_auc']
        }
    }).T
    
    logger.info(f"\n{comparison_df}")
    
    # Select best model based on F1-score
    best_model_name = 'Random Forest' if rf_metrics['f1_score'] > lr_metrics['f1_score'] else 'Logistic Regression'
    logger.info(f"\nBest model: {best_model_name} (F1-Score: {models[best_model_name]['metrics']['f1_score']:.4f})")
    
    return models, best_model_name


def plot_confusion_matrix(cm, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    return plt.gcf()


def plot_roc_curve(y_test, y_pred_proba, save_path=None):
    """
    Plot ROC curve
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        save_path: Optional path to save plot
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curve to {save_path}")
    
    return plt.gcf()


def plot_precision_recall_curve(y_test, y_pred_proba, save_path=None):
    """
    Plot Precision-Recall curve
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        save_path: Optional path to save plot
    """
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#2ca02c', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Precision-Recall curve to {save_path}")
    
    return plt.gcf()


if __name__ == "__main__":
    # Test training pipeline
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from src.data_loader import generate_synthetic_loan_data, save_loan_data
    from src.preprocessing import create_preprocessing_pipeline, prepare_features
    import config
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate data
    logger.info("Generating synthetic data...")
    df = generate_synthetic_loan_data(n_samples=1000)
    save_loan_data(df, config.LOAN_DATA_PATH)
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Create and fit preprocessor
    preprocessor = create_preprocessing_pipeline(
        config.NUMERIC_FEATURES,
        config.CATEGORICAL_FEATURES
    )
    preprocessor.fit(X_train)
    
    # Train and compare models
    models, best_model_name = train_and_compare_models(
        X_train, X_test, y_train, y_test, preprocessor
    )
    
    # Save best model
    best_model = models[best_model_name]['model']
    best_model.save(config.MODEL_PATH, config.PREPROCESSOR_PATH)
    
    logger.info("\nTraining completed successfully!")
