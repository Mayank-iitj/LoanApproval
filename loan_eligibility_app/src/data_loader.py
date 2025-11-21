"""
Data loading and validation utilities for Loan Eligibility Prediction
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_loan_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic loan application data with realistic distributions
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic loan data
    """
    np.random.seed(random_state)
    
    logger.info(f"Generating {n_samples} synthetic loan records...")
    
    # Generate applicant demographics
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.8, 0.2])
    married = np.random.choice(['Yes', 'No'], n_samples, p=[0.65, 0.35])
    dependents = np.random.choice(['0', '1', '2', '3+'], n_samples, p=[0.4, 0.25, 0.20, 0.15])
    education = np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.75, 0.25])
    self_employed = np.random.choice(['No', 'Yes'], n_samples, p=[0.85, 0.15])
    
    # Generate income data (realistic distributions)
    applicant_income = np.random.gamma(shape=2, scale=2000, size=n_samples).astype(int)
    applicant_income = np.clip(applicant_income, 1000, 50000)
    
    # Coapplicant income (many zeros, some with income)
    has_coapplicant = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    coapplicant_income = np.where(
        has_coapplicant,
        np.random.gamma(shape=1.5, scale=1500, size=n_samples).astype(int),
        0
    )
    coapplicant_income = np.clip(coapplicant_income, 0, 30000)
    
    # Loan amount (correlated with income)
    total_income = applicant_income + coapplicant_income
    loan_amount = (total_income * np.random.uniform(2, 5, n_samples) / 1000).astype(int)
    loan_amount = np.clip(loan_amount, 50, 700)
    
    # Loan term (mostly 360 months, some variations)
    loan_term = np.random.choice([120, 180, 240, 300, 360, 480], n_samples, 
                                  p=[0.05, 0.05, 0.10, 0.10, 0.65, 0.05])
    
    # Credit history (1 = good, 0 = bad)
    credit_history = np.random.choice([1, 0], n_samples, p=[0.85, 0.15])
    
    # Property area
    property_area = np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples, 
                                      p=[0.35, 0.40, 0.25])
    
    # Generate loan status based on realistic approval logic
    # Credit history is the strongest predictor
    approval_prob = np.where(credit_history == 1, 0.80, 0.30)
    
    # Adjust based on income
    income_ratio = total_income / loan_amount
    approval_prob = np.where(income_ratio > 3, approval_prob * 1.1, approval_prob * 0.9)
    
    # Adjust based on education
    approval_prob = np.where(education == 'Graduate', approval_prob * 1.05, approval_prob * 0.95)
    
    # Clip probabilities
    approval_prob = np.clip(approval_prob, 0, 1)
    
    # Generate final approval status
    loan_status = np.where(np.random.random(n_samples) < approval_prob, 'Y', 'N')
    
    # Introduce some missing values (realistic scenario)
    def add_missing_values(arr, missing_rate=0.05):
        mask = np.random.random(len(arr)) < missing_rate
        # Convert to object dtype to allow None values
        arr_copy = np.array(arr, dtype=object)
        arr_copy[mask] = None
        return arr_copy
    
    # Create DataFrame
    df = pd.DataFrame({
        'Gender': add_missing_values(gender, 0.03),
        'Married': add_missing_values(married, 0.02),
        'Dependents': add_missing_values(dependents, 0.03),
        'Education': education,  # No missing values
        'Self_Employed': add_missing_values(self_employed, 0.05),
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': add_missing_values(loan_amount, 0.04),
        'Loan_Amount_Term': add_missing_values(loan_term, 0.03),
        'Credit_History': add_missing_values(credit_history, 0.05),
        'Property_Area': property_area,
        'Loan_Status': loan_status
    })
    
    logger.info(f"Generated dataset with {len(df)} records")
    logger.info(f"Approval rate: {(df['Loan_Status'] == 'Y').mean():.2%}")
    
    return df


def load_loan_data(file_path: Path) -> pd.DataFrame:
    """
    Load loan data from CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with loan data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records from {file_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def validate_data(df: pd.DataFrame, required_columns: list) -> Tuple[bool, str]:
    """
    Validate that DataFrame has required columns and proper data types
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check for empty DataFrame
    if len(df) == 0:
        return False, "DataFrame is empty"
    
    return True, "Data validation passed"


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for the dataset
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_records': len(df),
        'num_features': len(df.columns) - 1,  # Exclude target
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
    }
    
    # Add target distribution if available
    if 'Loan_Status' in df.columns:
        summary['approval_rate'] = (df['Loan_Status'] == 'Y').mean()
        summary['approved_count'] = (df['Loan_Status'] == 'Y').sum()
        summary['rejected_count'] = (df['Loan_Status'] == 'N').sum()
    
    return summary


def save_loan_data(df: pd.DataFrame, file_path: Path) -> None:
    """
    Save loan data to CSV file
    
    Args:
        df: DataFrame to save
        file_path: Path where to save the CSV
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {len(df)} records to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise


if __name__ == "__main__":
    # Generate and save sample data
    from pathlib import Path
    
    # Generate data
    df = generate_synthetic_loan_data(n_samples=1000)
    
    # Save to file
    output_path = Path(__file__).parent.parent / "data" / "loan_data.csv"
    save_loan_data(df, output_path)
    
    # Print summary
    summary = get_data_summary(df)
    print("\nDataset Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
