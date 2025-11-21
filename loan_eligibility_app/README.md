# 🏦 Loan Eligibility Prediction System

A production-ready **Streamlit web application** for predicting loan eligibility using Machine Learning. This end-to-end ML system includes data exploration, model training, real-time predictions, explainability (SHAP), and batch processing capabilities.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Features

### Core Functionality
- ✅ **Real-time Loan Predictions** - Instant eligibility decisions with confidence scores
- ✅ **Batch Processing** - Upload CSV files for multiple predictions
- ✅ **Model Explainability** - SHAP-based explanations for transparency
- ✅ **Data Visualization** - Interactive charts and statistical analysis
- ✅ **Model Comparison** - Automatic selection between Logistic Regression and Random Forest
- ✅ **Premium UI** - Modern, responsive design with professional aesthetics

### Technical Features
- 🔬 **SHAP Integration** - Local and global feature importance
- 📊 **Comprehensive Metrics** - Accuracy, Precision, Recall, F1, ROC AUC
- 🔄 **Model Retraining** - Easy model updates through admin panel
- 📝 **Logging System** - Track predictions and system events
- ✔️ **Input Validation** - Robust error handling and user feedback
- 💾 **Session Management** - Remember predictions during session

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd loan_eligibility_app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate data and train the model**
   ```bash
   python src/train_model.py
   ```
   This will:
   - Generate synthetic loan data (~1000 records)
   - Train both Logistic Regression and Random Forest models
   - Evaluate and save the best model
   - Create necessary data and model files

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

---

## 📁 Project Structure

```
loan_eligibility_app/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration and constants
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/
│   └── loan_data.csv          # Synthetic loan dataset (generated)
│
├── models/
│   ├── model.pkl              # Trained ML model (generated)
│   └── preprocessor.pkl       # Preprocessing pipeline (generated)
│
├── logs/
│   └── app.log                # Application logs (generated)
│
└── src/
    ├── __init__.py            # Package initializer
    ├── data_loader.py         # Data loading and generation utilities
    ├── preprocessing.py       # Feature preprocessing pipelines
    ├── train_model.py         # Model training and evaluation
    ├── predict.py             # Prediction utilities
    └── explain.py             # SHAP explainability functions
```

---

## 📖 Usage Guide

### 1. Home Page
- View system overview and key metrics
- Check model status and dataset statistics
- Access quick start guide

### 2. Data Exploration
- Explore the loan dataset with interactive visualizations
- View statistical summaries and distributions
- Analyze correlations between features

### 3. Model Performance
- Review model evaluation metrics (Accuracy, Precision, Recall, F1, ROC AUC)
- Visualize confusion matrix, ROC curve, and precision-recall curve
- Examine feature importance

### 4. Loan Prediction
- Enter applicant details through an intuitive form
- Get instant predictions with confidence scores
- View probability gauge and interpretation

### 5. Explain My Result
- Understand which factors influenced the decision
- View SHAP waterfall plots
- See top contributing features

### 6. Batch Prediction
- Download CSV template
- Upload filled CSV with multiple applicants
- Get predictions for all applications
- Download results with predictions

### 7. Admin / Settings
- Train or retrain models
- View system information
- Check recent logs
- Clear cache

---

## 🔧 Configuration

Key configurations can be modified in `config.py`:

```python
# Dataset parameters
DATASET_SIZE = 1000                    # Number of synthetic records
RANDOM_STATE = 42                      # Reproducibility seed
TEST_SIZE = 0.2                        # Train/test split ratio

# Model parameters
MODELS_CONFIG = {
    'Logistic Regression': {...},
    'Random Forest': {...}
}

# Feature definitions
NUMERIC_FEATURES = [...]
CATEGORICAL_FEATURES = [...]
```

---

## 📊 Dataset Features

### Applicant Information
- **ApplicantIncome** - Monthly income of primary applicant
- **CoapplicantIncome** - Monthly income of coapplicant
- **Gender** - Male/Female
- **Married** - Yes/No
- **Dependents** - 0, 1, 2, 3+
- **Education** - Graduate/Not Graduate
- **Self_Employed** - Yes/No

### Loan Information
- **LoanAmount** - Requested loan amount (in thousands)
- **Loan_Amount_Term** - Loan duration in months
- **Credit_History** - Good (1) or Bad (0)
- **Property_Area** - Urban/Semiurban/Rural

### Target Variable
- **Loan_Status** - Y (Approved) / N (Rejected)

---

## 🤖 Machine Learning Pipeline

### 1. Data Preprocessing
- **Missing Value Imputation**
  - Numeric: Median imputation
  - Categorical: Most frequent imputation
- **Feature Encoding**
  - One-hot encoding for categorical variables
- **Feature Scaling**
  - StandardScaler for numeric features

### 2. Model Training
- **Algorithms**
  - Logistic Regression (baseline)
  - Random Forest Classifier (ensemble)
- **Evaluation**
  - 80/20 train-test split
  - 5-fold cross-validation
  - Multiple metrics (Accuracy, Precision, Recall, F1, ROC AUC)
- **Selection**
  - Best model chosen based on F1-score

### 3. Model Explainability
- **SHAP (SHapley Additive exPlanations)**
  - Global feature importance
  - Local prediction explanations
  - Waterfall plots for individual predictions

---

## 📈 Model Performance

Typical performance metrics on the synthetic dataset:

| Metric | Value |
|--------|-------|
| Accuracy | ~85% |
| Precision | ~82% |
| Recall | ~88% |
| F1-Score | ~85% |
| ROC AUC | ~90% |

*Note: Actual performance may vary based on the generated synthetic data*

---

## 🎨 UI/UX Features

### Premium Design Elements
- **Modern Color Scheme** - Professional bank-like aesthetics
- **Responsive Layout** - Works on desktop, tablet, and mobile
- **Interactive Charts** - Plotly-based visualizations
- **Custom CSS** - Polished cards, badges, and buttons
- **Smooth Animations** - Hover effects and transitions
- **Progress Indicators** - Spinners and loading states

### User Experience
- **Input Validation** - Real-time error checking
- **Helpful Tooltips** - Guidance for each field
- **Session State** - Remember last prediction
- **Download Options** - Export results as CSV
- **Clear Navigation** - Intuitive sidebar menu

---

## 🔍 Example Use Cases

### Single Prediction
```
Applicant: Graduate, Married, 1 Dependent
Income: $5,000 (applicant) + $2,000 (coapplicant)
Loan: $150,000 for 360 months
Credit History: Good
Property: Urban

Result: ✅ APPROVED (85% confidence)
```

### Batch Prediction
```
Upload CSV with 100 applicants
→ Get predictions for all
→ Download results with approval probabilities
→ 65 approved, 35 rejected
```

---

## 📝 Logging

All predictions and system events are logged to `logs/app.log`:

```
2025-11-21 17:30:15 - INFO - Model loaded successfully
2025-11-21 17:30:45 - INFO - Prediction made: Approved (confidence: 85%)
2025-11-21 17:31:20 - INFO - Batch prediction completed: 100 applications
```

---

## 🛠️ Troubleshooting

### Model not found error
**Solution**: Run `python src/train_model.py` to train the model

### Import errors
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

### Data file not found
**Solution**: The app will automatically generate synthetic data on first run

### SHAP errors
**Solution**: SHAP works best with tree-based models. If using Logistic Regression, explanations may be limited

---

## 🔮 Future Enhancements

- [ ] Add more ML algorithms (XGBoost, LightGBM)
- [ ] Implement user authentication
- [ ] Add database integration for storing predictions
- [ ] Create REST API for programmatic access
- [ ] Add A/B testing for model comparison
- [ ] Implement real-time model monitoring
- [ ] Add support for custom datasets
- [ ] Create mobile app version

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Mayank Sharma**
- AI/ML Developer
- Portfolio: [Your Portfolio Link]
- GitHub: [Your GitHub Link]

---

## 🙏 Acknowledgments

- **Streamlit** - For the amazing web framework
- **scikit-learn** - For ML algorithms and tools
- **SHAP** - For model explainability
- **Plotly** - For interactive visualizations

---

## 📞 Support

For issues, questions, or suggestions:
1. Check the troubleshooting section
2. Review the logs in `logs/app.log`
3. Open an issue on GitHub
4. Contact the developer

---

**Built with ❤️ using Python and Streamlit**
