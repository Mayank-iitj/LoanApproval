"""
SHAP-based explainability for Loan Eligibility Prediction
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class LoanExplainer:
    """
    SHAP-based explainer for loan eligibility predictions
    """
    
    def __init__(self, model, preprocessor, X_train_sample=None):
        """
        Initialize the explainer
        
        Args:
            model: Trained model
            preprocessor: Fitted preprocessor
            X_train_sample: Sample of training data for background (optional)
        """
        self.model = model
        self.preprocessor = preprocessor
        self.explainer = None
        self.X_train_sample = X_train_sample
        
        # Initialize SHAP explainer
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer based on model type"""
        try:
            # For tree-based models, use TreeExplainer
            if hasattr(self.model, 'estimators_'):
                logger.info("Initializing TreeExplainer for Random Forest")
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # For linear models, use LinearExplainer or KernelExplainer
                logger.info("Initializing Explainer for Logistic Regression")
                if self.X_train_sample is not None:
                    X_background = self.preprocessor.transform(self.X_train_sample)
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba,
                        X_background[:100]  # Use subset for speed
                    )
                else:
                    # Use a simple approach
                    self.explainer = shap.Explainer(self.model.predict_proba, self.model)
            
            logger.info("SHAP explainer initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.explainer = None
    
    def explain_prediction(self, X: pd.DataFrame, feature_names: List[str] = None) -> Dict:
        """
        Get SHAP explanation for a prediction
        
        Args:
            X: Input features (single row DataFrame)
            feature_names: List of feature names
            
        Returns:
            Dictionary with SHAP values and explanations
        """
        if self.explainer is None:
            logger.warning("Explainer not initialized, returning empty explanation")
            return {}
        
        try:
            # Transform input
            X_transformed = self.preprocessor.transform(X)
            
            # Get SHAP values
            if hasattr(self.model, 'estimators_'):
                # Tree explainer
                shap_values = self.explainer.shap_values(X_transformed)
                # For binary classification, take positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                # Kernel explainer
                shap_values = self.explainer.shap_values(X_transformed)
                if len(shap_values.shape) > 2:
                    shap_values = shap_values[:, :, 1]
            
            # Get feature names if not provided
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(X_transformed.shape[1])]
            
            # Create explanation dictionary
            explanation = {
                'shap_values': shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                'feature_names': feature_names,
                'base_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return {}
    
    def get_feature_importance(self, X: pd.DataFrame, feature_names: List[str] = None, top_n: int = 10) -> pd.DataFrame:
        """
        Get global feature importance using SHAP
        
        Args:
            X: Input features
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.explainer is None:
            logger.warning("Explainer not initialized")
            return pd.DataFrame()
        
        try:
            # Transform input
            X_transformed = self.preprocessor.transform(X)
            
            # Get SHAP values
            if hasattr(self.model, 'estimators_'):
                shap_values = self.explainer.shap_values(X_transformed)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                shap_values = self.explainer.shap_values(X_transformed)
                if len(shap_values.shape) > 2:
                    shap_values = shap_values[:, :, 1]
            
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            # Get feature names
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(len(mean_shap))]
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_shap
            }).sort_values('importance', ascending=False).head(top_n)
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return pd.DataFrame()
    
    def plot_waterfall(self, explanation: Dict, max_display: int = 10, show: bool = True):
        """
        Create SHAP waterfall plot for a single prediction
        
        Args:
            explanation: Explanation dictionary from explain_prediction
            max_display: Maximum number of features to display
            show: Whether to show the plot
            
        Returns:
            Matplotlib figure
        """
        if not explanation:
            logger.warning("No explanation provided")
            return None
        
        try:
            shap_values = explanation['shap_values']
            feature_names = explanation['feature_names']
            base_value = explanation.get('base_value', 0)
            
            # Get top features by absolute SHAP value
            abs_shap = np.abs(shap_values)
            top_indices = np.argsort(abs_shap)[-max_display:][::-1]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort by SHAP value
            sorted_indices = sorted(top_indices, key=lambda i: shap_values[i])
            
            y_pos = np.arange(len(sorted_indices))
            values = [shap_values[i] for i in sorted_indices]
            names = [feature_names[i] if i < len(feature_names) else f"Feature_{i}" 
                    for i in sorted_indices]
            
            # Color bars based on positive/negative contribution
            colors = ['#ff4444' if v < 0 else '#44ff44' for v in values]
            
            ax.barh(y_pos, values, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names)
            ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
            ax.set_title('Feature Contributions to Prediction', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            
            if show:
                plt.show()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating waterfall plot: {e}")
            return None
    
    def plot_summary(self, X: pd.DataFrame, feature_names: List[str] = None, 
                     max_display: int = 10, show: bool = True):
        """
        Create SHAP summary plot
        
        Args:
            X: Input features
            feature_names: List of feature names
            max_display: Maximum number of features to display
            show: Whether to show the plot
            
        Returns:
            Matplotlib figure
        """
        if self.explainer is None:
            logger.warning("Explainer not initialized")
            return None
        
        try:
            # Transform input
            X_transformed = self.preprocessor.transform(X)
            
            # Get SHAP values
            if hasattr(self.model, 'estimators_'):
                shap_values = self.explainer.shap_values(X_transformed)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                shap_values = self.explainer.shap_values(X_transformed)
                if len(shap_values.shape) > 2:
                    shap_values = shap_values[:, :, 1]
            
            # Create summary plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Calculate mean absolute SHAP values for sorting
            mean_shap = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(mean_shap)[-max_display:][::-1]
            
            # Get feature names
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(X_transformed.shape[1])]
            
            # Plot
            for idx in top_indices:
                feature_shap = shap_values[:, idx]
                ax.scatter(feature_shap, [feature_names[idx]] * len(feature_shap), 
                          alpha=0.5, s=20)
            
            ax.set_xlabel('SHAP Value', fontsize=12)
            ax.set_title('SHAP Summary Plot', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            
            if show:
                plt.show()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating summary plot: {e}")
            return None


def get_simple_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Get simple feature importance for models that support it
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    try:
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_[0])
        else:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Test explainer
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from src.data_loader import load_loan_data
    from src.preprocessing import prepare_features
    from src.predict import load_model_and_preprocessor
    import config
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data and model
    df = load_loan_data(config.LOAN_DATA_PATH)
    X, y = prepare_features(df)
    model, preprocessor = load_model_and_preprocessor(config.MODEL_PATH, config.PREPROCESSOR_PATH)
    
    # Create explainer
    explainer = LoanExplainer(model, preprocessor, X.head(100))
    
    # Test single explanation
    test_sample = X.head(1)
    explanation = explainer.explain_prediction(test_sample, config.ALL_FEATURES)
    
    if explanation:
        print("SHAP Explanation generated successfully")
        explainer.plot_waterfall(explanation, show=False)
        print("Waterfall plot created")
