"""
Prediction utilities for Loan Eligibility Prediction
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Union

logger = logging.getLogger(__name__)


def load_model_and_preprocessor(model_path: Path, preprocessor_path: Path) -> Tuple:
    """
    Load trained model and preprocessor
    
    Args:
        model_path: Path to model file
        preprocessor_path: Path to preprocessor file
        
    Returns:
        Tuple of (model, preprocessor)
    """
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        logger.info("Successfully loaded model and preprocessor")
        return model, preprocessor
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def predict_single(input_data: Dict, model, preprocessor, feature_columns: list) -> Dict:
    """
    Make prediction for a single applicant
    
    Args:
        input_data: Dictionary with applicant information
        model: Trained model
        preprocessor: Fitted preprocessor
        feature_columns: List of feature column names
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Create DataFrame from input
        df = pd.DataFrame([input_data], columns=feature_columns)
        
        # Convert Credit_History if it's a string
        if 'Credit_History' in df.columns:
            if df['Credit_History'].dtype == 'object':
                df['Credit_History'] = df['Credit_History'].map({'Yes': 1, 'No': 0})
        
        # Transform data
        X_transformed = preprocessor.transform(df)
        
        # Make prediction
        prediction = model.predict(X_transformed)[0]
        probability = model.predict_proba(X_transformed)[0]
        
        # Prepare result
        result = {
            'prediction': 'Approved' if prediction == 1 else 'Rejected',
            'prediction_binary': int(prediction),
            'probability_approved': float(probability[1]),
            'probability_rejected': float(probability[0]),
            'confidence': float(max(probability))
        }
        
        logger.info(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2%})")
        
        return result
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise


def predict_batch(df: pd.DataFrame, model, preprocessor) -> pd.DataFrame:
    """
    Make predictions for multiple applicants
    
    Args:
        df: DataFrame with applicant information
        model: Trained model
        preprocessor: Fitted preprocessor
        
    Returns:
        DataFrame with predictions added
    """
    try:
        # Make a copy to avoid modifying original
        result_df = df.copy()
        
        # Convert Credit_History if it's a string
        if 'Credit_History' in result_df.columns:
            if result_df['Credit_History'].dtype == 'object':
                result_df['Credit_History'] = result_df['Credit_History'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
        
        # Transform data
        X_transformed = preprocessor.transform(result_df)
        
        # Make predictions
        predictions = model.predict(X_transformed)
        probabilities = model.predict_proba(X_transformed)[:, 1]
        
        # Add predictions to DataFrame
        result_df['Predicted_Loan_Status'] = ['Y' if p == 1 else 'N' for p in predictions]
        result_df['Approval_Probability'] = probabilities
        
        logger.info(f"Made predictions for {len(result_df)} applicants")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error making batch predictions: {str(e)}")
        raise


def interpret_prediction(result: Dict, input_data: Dict) -> str:
    """
    Generate human-readable interpretation of prediction
    
    Args:
        result: Prediction result dictionary
        input_data: Input data dictionary
        
    Returns:
        Interpretation string
    """
    prediction = result['prediction']
    confidence = result['confidence']
    
    # Base interpretation
    if prediction == 'Approved':
        interpretation = f"✅ **Loan Application APPROVED** with {confidence:.1%} confidence.\n\n"
    else:
        interpretation = f"❌ **Loan Application REJECTED** with {confidence:.1%} confidence.\n\n"
    
    # Add factors
    interpretation += "**Key Factors:**\n"
    
    # Credit history
    credit_history = input_data.get('Credit_History', 'Unknown')
    if credit_history in ['Yes', 1, '1']:
        interpretation += "- ✓ Good credit history\n"
    else:
        interpretation += "- ✗ No credit history or poor credit\n"
    
    # Income
    total_income = input_data.get('ApplicantIncome', 0) + input_data.get('CoapplicantIncome', 0)
    loan_amount = input_data.get('LoanAmount', 0) * 1000  # Convert to actual amount
    
    if loan_amount > 0:
        income_ratio = total_income / loan_amount
        if income_ratio > 3:
            interpretation += f"- ✓ Strong income-to-loan ratio ({income_ratio:.1f}x)\n"
        elif income_ratio > 2:
            interpretation += f"- ~ Moderate income-to-loan ratio ({income_ratio:.1f}x)\n"
        else:
            interpretation += f"- ✗ Low income-to-loan ratio ({income_ratio:.1f}x)\n"
    
    # Education
    education = input_data.get('Education', 'Unknown')
    if education == 'Graduate':
        interpretation += "- ✓ Graduate education\n"
    
    # Property area
    property_area = input_data.get('Property_Area', 'Unknown')
    interpretation += f"- Property in {property_area} area\n"
    
    return interpretation


def validate_prediction_input(input_data: Dict, required_features: list) -> Tuple[bool, str]:
    """
    Validate input data before prediction
    
    Args:
        input_data: Dictionary with input values
        required_features: List of required feature names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    errors = []
    
    # Check for missing features
    missing = set(required_features) - set(input_data.keys())
    if missing:
        errors.append(f"Missing required fields: {', '.join(missing)}")
    
    # Validate numeric fields
    numeric_fields = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    for field in numeric_fields:
        if field in input_data:
            try:
                value = float(input_data[field])
                if value < 0:
                    errors.append(f"{field} cannot be negative")
            except (ValueError, TypeError):
                errors.append(f"{field} must be a number")
    
    # Validate specific ranges
    if 'ApplicantIncome' in input_data:
        if float(input_data['ApplicantIncome']) < 500:
            errors.append("Applicant income seems too low (minimum 500)")
    
    if 'LoanAmount' in input_data:
        if float(input_data['LoanAmount']) < 10:
            errors.append("Loan amount seems too low (minimum 10)")
    
    if 'Loan_Amount_Term' in input_data:
        term = float(input_data['Loan_Amount_Term'])
        if term < 12 or term > 480:
            errors.append("Loan term must be between 12 and 480 months")
    
    if errors:
        return False, " | ".join(errors)
    
    return True, "Validation passed"


if __name__ == "__main__":
    # Test prediction
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    import config
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load model
    model, preprocessor = load_model_and_preprocessor(
        config.MODEL_PATH,
        config.PREPROCESSOR_PATH
    )
    
    # Test single prediction
    test_input = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '1',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 2000,
        'LoanAmount': 150,
        'Loan_Amount_Term': 360,
        'Credit_History': 'Yes',
        'Property_Area': 'Urban'
    }
    
    # Validate input
    is_valid, message = validate_prediction_input(test_input, config.ALL_FEATURES)
    print(f"Validation: {is_valid} - {message}")
    
    if is_valid:
        # Make prediction
        result = predict_single(test_input, model, preprocessor, config.ALL_FEATURES)
        print(f"\nPrediction Result:")
        print(f"  Decision: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Approval Probability: {result['probability_approved']:.2%}")
        
        # Interpretation
        interpretation = interpret_prediction(result, test_input)
        print(f"\n{interpretation}")
