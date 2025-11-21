"""
Preprocessing pipelines for Loan Eligibility Prediction
Handles missing values, encoding, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)


def create_preprocessing_pipeline(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for loan data
    
    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        
    Returns:
        ColumnTransformer with preprocessing steps
    """
    # Numeric pipeline: impute with median, then scale
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: impute with most frequent, then one-hot encode
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop'
    )
    
    logger.info(f"Created preprocessing pipeline with {len(numeric_features)} numeric and {len(categorical_features)} categorical features")
    
    return preprocessor


def prepare_features(df: pd.DataFrame, target_column: str = 'Loan_Status') -> tuple:
    """
    Prepare features and target from DataFrame
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Convert target to binary (Y=1, N=0)
    y = (y == 'Y').astype(int)
    
    logger.info(f"Prepared features: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y


def get_feature_names(preprocessor: ColumnTransformer, numeric_features: list, categorical_features: list) -> list:
    """
    Get feature names after preprocessing
    
    Args:
        preprocessor: Fitted ColumnTransformer
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        
    Returns:
        List of feature names after transformation
    """
    feature_names = []
    
    # Add numeric feature names
    feature_names.extend(numeric_features)
    
    # Add categorical feature names (after one-hot encoding)
    try:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = cat_encoder.get_feature_names_out(categorical_features)
        feature_names.extend(cat_features)
    except Exception as e:
        logger.warning(f"Could not extract categorical feature names: {e}")
        feature_names.extend([f"cat_{i}" for i in range(len(categorical_features))])
    
    return feature_names


def validate_input_data(input_dict: dict, numeric_features: list, categorical_features: list, 
                        feature_ranges: dict = None) -> tuple:
    """
    Validate input data for prediction
    
    Args:
        input_dict: Dictionary with input values
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        feature_ranges: Optional dictionary with valid ranges for numeric features
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    errors = []
    
    # Check all required features are present
    required_features = numeric_features + categorical_features
    missing_features = set(required_features) - set(input_dict.keys())
    if missing_features:
        errors.append(f"Missing required features: {missing_features}")
    
    # Validate numeric features
    for feature in numeric_features:
        if feature in input_dict:
            value = input_dict[feature]
            
            # Check if numeric
            try:
                value = float(value)
            except (ValueError, TypeError):
                errors.append(f"{feature} must be a number")
                continue
            
            # Check if negative
            if value < 0:
                errors.append(f"{feature} cannot be negative")
            
            # Check range if provided
            if feature_ranges and feature in feature_ranges:
                min_val, max_val = feature_ranges[feature]
                if not (min_val <= value <= max_val):
                    errors.append(f"{feature} must be between {min_val} and {max_val}")
    
    # Validate categorical features
    for feature in categorical_features:
        if feature in input_dict:
            value = input_dict[feature]
            if value is None or value == '':
                errors.append(f"{feature} is required")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, "Validation passed"


def convert_credit_history(value: str) -> int:
    """
    Convert credit history from Yes/No to 1/0
    
    Args:
        value: 'Yes' or 'No'
        
    Returns:
        1 for Yes, 0 for No
    """
    if isinstance(value, str):
        return 1 if value.lower() == 'yes' else 0
    return int(value)


def prepare_single_input(input_dict: dict, numeric_features: list, categorical_features: list) -> pd.DataFrame:
    """
    Prepare a single input dictionary for prediction
    
    Args:
        input_dict: Dictionary with input values
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        
    Returns:
        DataFrame with single row ready for prediction
    """
    # Create a copy to avoid modifying original
    data = input_dict.copy()
    
    # Convert Credit_History if present
    if 'Credit_History' in data:
        data['Credit_History'] = convert_credit_history(data['Credit_History'])
    
    # Create DataFrame with correct column order
    all_features = numeric_features + categorical_features
    df = pd.DataFrame([data], columns=all_features)
    
    # Convert numeric columns to proper type
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


if __name__ == "__main__":
    # Test preprocessing pipeline
    from data_loader import generate_synthetic_loan_data
    import config
    
    # Generate sample data
    df = generate_synthetic_loan_data(100)
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Create and fit preprocessor
    preprocessor = create_preprocessing_pipeline(
        config.NUMERIC_FEATURES,
        config.CATEGORICAL_FEATURES
    )
    
    X_transformed = preprocessor.fit_transform(X)
    
    print(f"\nOriginal shape: {X.shape}")
    print(f"Transformed shape: {X_transformed.shape}")
    print(f"Feature names: {get_feature_names(preprocessor, config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES)}")
