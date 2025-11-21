"""
Configuration file for Loan Eligibility Prediction App
Contains all constants, paths, and configuration parameters
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Directory paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
SRC_DIR = BASE_DIR / "src"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, SRC_DIR]:
    directory.mkdir(exist_ok=True)

# File paths
LOAN_DATA_PATH = DATA_DIR / "loan_data.csv"
MODEL_PATH = MODELS_DIR / "model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
LOG_FILE_PATH = LOGS_DIR / "app.log"

# Feature definitions
NUMERIC_FEATURES = [
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term'
]

CATEGORICAL_FEATURES = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'Credit_History',
    'Property_Area'
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN = 'Loan_Status'

# Feature value options for UI
FEATURE_OPTIONS = {
    'Gender': ['Male', 'Female'],
    'Married': ['Yes', 'No'],
    'Dependents': ['0', '1', '2', '3+'],
    'Education': ['Graduate', 'Not Graduate'],
    'Self_Employed': ['Yes', 'No'],
    'Credit_History': ['Yes', 'No'],
    'Property_Area': ['Urban', 'Semiurban', 'Rural']
}

# Feature ranges for validation
FEATURE_RANGES = {
    'ApplicantIncome': (0, 100000),
    'CoapplicantIncome': (0, 50000),
    'LoanAmount': (0, 1000),
    'Loan_Amount_Term': (12, 480)
}

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Model configurations
MODELS_CONFIG = {
    'Logistic Regression': {
        'name': 'LogisticRegression',
        'params': {
            'max_iter': 1000,
            'random_state': RANDOM_STATE
        }
    },
    'Random Forest': {
        'name': 'RandomForestClassifier',
        'params': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': RANDOM_STATE
        }
    }
}

# UI Configuration
APP_TITLE = "🏦 Loan Eligibility Prediction System"
APP_ICON = "🏦"

# Color scheme (professional bank theme)
COLORS = {
    'primary': '#1f77b4',      # Professional blue
    'secondary': '#2ca02c',    # Success green
    'danger': '#d62728',       # Rejection red
    'warning': '#ff7f0e',      # Warning orange
    'info': '#17becf',         # Info cyan
    'background': '#f8f9fa',   # Light gray background
    'text': '#212529'          # Dark text
}

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# Dataset generation parameters
DATASET_SIZE = 1000
APPROVAL_RATE_WITH_CREDIT = 0.80
APPROVAL_RATE_WITHOUT_CREDIT = 0.30

# Streamlit page configuration
PAGE_CONFIG = {
    'page_title': 'Loan Eligibility Predictor',
    'page_icon': '🏦',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Custom CSS for premium UI
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Success/Failure badges */
    .approved-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 1.2em;
        border: 2px solid #28a745;
    }
    
    .rejected-badge {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 1.2em;
        border: 2px solid #dc3545;
    }
    
    /* Header styling */
    h1 {
        color: #1f77b4;
        font-weight: 600;
    }
    
    h2 {
        color: #2c3e50;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: 500;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #155a8a;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2c3e50;
    }
    
    /* Info box */
    .info-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-left: 4px solid #1f77b4;
        border-radius: 4px;
        margin: 10px 0;
    }
    
    /* Warning box */
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-left: 4px solid #ff7f0e;
        border-radius: 4px;
        margin: 10px 0;
    }
    
    /* Watermark styling */
    .watermark {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 12px 20px;
        border-radius: 25px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        animation: fadeIn 1s ease-in;
    }
    
    .watermark:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .watermark a {
        color: white;
        text-decoration: none;
        font-weight: 600;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .watermark a:hover {
        color: #fff;
    }
    
    .watermark-icon {
        font-size: 18px;
        animation: pulse 2s infinite;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.1);
        }
    }
</style>
"""
