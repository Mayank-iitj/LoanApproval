"""
Loan Eligibility Prediction System - Streamlit Web Application
A production-ready ML system for predicting loan eligibility with explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from pathlib import Path
import sys
import io

# Add src to path
sys.path.append(str(Path(__file__).parent))

import config
from src.data_loader import load_loan_data, generate_synthetic_loan_data, save_loan_data, get_data_summary
from src.preprocessing import create_preprocessing_pipeline, prepare_features
from src.predict import load_model_and_preprocessor, predict_single, predict_batch, interpret_prediction, validate_prediction_input
from src.train_model import train_and_compare_models, LoanEligibilityModel
from src.explain import LoanExplainer, get_simple_feature_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

# Configure logging
logging.basicConfig(
    filename=config.LOG_FILE_PATH,
    level=logging.INFO,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(**config.PAGE_CONFIG)

# Custom CSS
st.markdown(config.CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model():
    """Load model and preprocessor with caching"""
    try:
        model, preprocessor = load_model_and_preprocessor(
            config.MODEL_PATH,
            config.PREPROCESSOR_PATH
        )
        logger.info("Model loaded successfully")
        return model, preprocessor
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"⚠️ Error loading model: {e}")
        return None, None


@st.cache_data
def load_data():
    """Load loan data with caching"""
    try:
        if config.LOAN_DATA_PATH.exists():
            df = load_loan_data(config.LOAN_DATA_PATH)
            logger.info("Data loaded successfully")
            return df
        else:
            st.warning("⚠️ Data file not found. Please generate data from Admin panel.")
            return None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"⚠️ Error loading data: {e}")
        return None


def create_gauge_chart(probability, title="Approval Probability"):
    """Create a gauge chart for probability visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#ffcccc'},
                {'range': [30, 70], 'color': '#ffffcc'},
                {'range': [70, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig


def add_watermark(repo_url="https://github.com/mayankksp/loan-eligibility-app"):
    """Add a stylish watermark with link to repository"""
    watermark_html = f"""
    <div class="watermark">
        <a href="{repo_url}" target="_blank">
            <span class="watermark-icon">✨</span>
            <span>Created by Mayank Sharma</span>
        </a>
    </div>
    """
    st.markdown(watermark_html, unsafe_allow_html=True)


# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def home_page():
    """Home page with overview and statistics"""
    st.title("🏦 Loan Eligibility Prediction System")
    st.markdown("### Welcome to the AI-Powered Loan Decision System")
    
    st.markdown("""
    This application uses **Machine Learning** to predict loan eligibility based on applicant information.
    Get instant decisions with **explainable AI** and confidence scores.
    """)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    # Load data for stats
    df = load_data()
    model, preprocessor = load_model()
    
    if df is not None:
        summary = get_data_summary(df)
        
        with col1:
            st.metric("📊 Total Records", f"{summary['total_records']:,}")
        
        with col2:
            st.metric("✅ Approval Rate", f"{summary['approval_rate']:.1%}")
        
        with col3:
            st.metric("📈 Features", summary['num_features'])
        
        with col4:
            model_status = "✅ Ready" if model is not None else "❌ Not Found"
            st.metric("🤖 Model Status", model_status)
    
    st.markdown("---")
    
    # Features
    st.markdown("### 🎯 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Core Functionality
        - ✅ **Real-time Predictions** - Instant eligibility decisions
        - ✅ **Batch Processing** - Upload CSV for multiple predictions
        - ✅ **Model Explainability** - SHAP-based explanations
        - ✅ **Data Visualization** - Interactive charts and analysis
        """)
    
    with col2:
        st.markdown("""
        #### Technical Features
        - 🔬 **SHAP Integration** - Feature importance analysis
        - 📊 **Comprehensive Metrics** - Accuracy, Precision, Recall, F1, ROC AUC
        - 🔄 **Model Retraining** - Easy model updates
        - 📝 **Logging System** - Track all predictions
        """)
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    
    st.markdown("""
    1. **📊 Data Exploration** - Explore the loan dataset with visualizations
    2. **📈 Model Performance** - Review model metrics and evaluation
    3. **🎯 Loan Prediction** - Make predictions for individual applicants
    4. **💡 Explain My Result** - Understand prediction factors
    5. **📦 Batch Prediction** - Process multiple applications at once
    6. **⚙️ Admin / Settings** - Train models and manage system
    """)
    
    # Model info
    if model is not None:
        st.markdown("---")
        st.success("✅ **System Ready** - Model loaded and ready for predictions!")
    else:
        st.markdown("---")
        st.warning("⚠️ **Setup Required** - Please train a model from the Admin panel to get started.")


def data_exploration_page():
    """Data exploration with visualizations"""
    st.title("📊 Data Exploration")
    st.markdown("Explore the loan dataset with interactive visualizations")
    
    df = load_data()
    
    if df is None:
        st.warning("⚠️ No data available. Please generate data from the Admin panel.")
        return
    
    # Dataset overview
    st.markdown("### Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    
    summary = get_data_summary(df)
    
    with col1:
        st.metric("Total Applications", f"{summary['total_records']:,}")
    
    with col2:
        st.metric("Approved", f"{summary['approved_count']:,}")
    
    with col3:
        st.metric("Rejected", f"{summary['rejected_count']:,}")
    
    # Show sample data
    st.markdown("### Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistical summary
    st.markdown("### Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Visualizations
    st.markdown("### 📈 Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Correlations", "Income Analysis", "Loan Patterns"])
    
    with tab1:
        # Loan status distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                df,
                names='Loan_Status',
                title='Loan Approval Distribution',
                color='Loan_Status',
                color_discrete_map={'Y': '#2ca02c', 'N': '#d62728'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Property area distribution
            property_counts = df['Property_Area'].value_counts()
            fig = px.bar(
                x=property_counts.index,
                y=property_counts.values,
                title='Applications by Property Area',
                labels={'x': 'Property Area', 'y': 'Count'},
                color=property_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title='Feature Correlation Heatmap',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Income analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df,
                x='ApplicantIncome',
                color='Loan_Status',
                title='Applicant Income Distribution',
                nbins=30,
                color_discrete_map={'Y': '#2ca02c', 'N': '#d62728'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df,
                x='Loan_Status',
                y='ApplicantIncome',
                title='Income vs Loan Status',
                color='Loan_Status',
                color_discrete_map={'Y': '#2ca02c', 'N': '#d62728'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Loan amount patterns
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df,
                x='LoanAmount',
                color='Loan_Status',
                title='Loan Amount Distribution',
                nbins=30,
                color_discrete_map={'Y': '#2ca02c', 'N': '#d62728'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                df,
                x='ApplicantIncome',
                y='LoanAmount',
                color='Loan_Status',
                title='Income vs Loan Amount',
                color_discrete_map={'Y': '#2ca02c', 'N': '#d62728'}
            )
            st.plotly_chart(fig, use_container_width=True)


def model_performance_page():
    """Model performance metrics and visualizations"""
    st.title("📈 Model Performance")
    st.markdown("Evaluate model performance with comprehensive metrics")
    
    model, preprocessor = load_model()
    df = load_data()
    
    if model is None or df is None:
        st.warning("⚠️ Model or data not available. Please train a model from the Admin panel.")
        return
    
    # Prepare data
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    # Make predictions
    X_test_transformed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_transformed)
    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Display metrics
    st.markdown("### 📊 Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    
    with col2:
        st.metric("Precision", f"{precision:.2%}")
    
    with col3:
        st.metric("Recall", f"{recall:.2%}")
    
    with col4:
        st.metric("F1-Score", f"{f1:.2%}")
    
    with col5:
        st.metric("ROC AUC", f"{roc_auc:.2%}")
    
    st.markdown("---")
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix", "ROC Curve", "Precision-Recall", "Feature Importance"])
    
    with tab1:
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Rejected', 'Approved'],
            y=['Rejected', 'Approved'],
            title='Confusion Matrix',
            color_continuous_scale='Blues',
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics explanation
        st.info(f"""
        **Confusion Matrix Interpretation:**
        - True Negatives (Rejected correctly): {cm[0, 0]}
        - False Positives (Approved incorrectly): {cm[0, 1]}
        - False Negatives (Rejected incorrectly): {cm[1, 0]}
        - True Positives (Approved correctly): {cm[1, 1]}
        """)
    
    with tab2:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall_curve, y=precision_curve,
            mode='lines',
            name='Precision-Recall Curve',
            line=dict(color='#2ca02c', width=2)
        ))
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Feature importance
        try:
            # Get feature names after transformation
            from src.preprocessing import get_feature_names
            feature_names = get_feature_names(preprocessor, config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES)
            
            # Get importance
            importance_df = get_simple_feature_importance(model, feature_names)
            
            if not importance_df.empty:
                fig = px.bar(
                    importance_df.head(15),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 15 Feature Importance',
                    labels={'importance': 'Importance', 'feature': 'Feature'},
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")
        
        except Exception as e:
            st.error(f"Error displaying feature importance: {e}")


def loan_prediction_page():
    """Single loan prediction page"""
    st.title("🎯 Loan Prediction")
    st.markdown("Enter applicant details to get instant loan eligibility prediction")
    
    model, preprocessor = load_model()
    
    if model is None:
        st.warning("⚠️ Model not available. Please train a model from the Admin panel.")
        return
    
    # Input form
    st.markdown("### 📝 Applicant Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Personal Details")
        gender = st.selectbox("Gender", config.FEATURE_OPTIONS['Gender'])
        married = st.selectbox("Married", config.FEATURE_OPTIONS['Married'])
        dependents = st.selectbox("Number of Dependents", config.FEATURE_OPTIONS['Dependents'])
        education = st.selectbox("Education", config.FEATURE_OPTIONS['Education'])
        self_employed = st.selectbox("Self Employed", config.FEATURE_OPTIONS['Self_Employed'])
    
    with col2:
        st.markdown("#### Financial Details")
        applicant_income = st.number_input(
            "Applicant Monthly Income ($)",
            min_value=0,
            max_value=100000,
            value=5000,
            step=100
        )
        coapplicant_income = st.number_input(
            "Coapplicant Monthly Income ($)",
            min_value=0,
            max_value=50000,
            value=0,
            step=100
        )
        loan_amount = st.number_input(
            "Loan Amount (in thousands $)",
            min_value=0,
            max_value=1000,
            value=150,
            step=10
        )
        loan_term = st.selectbox(
            "Loan Term (months)",
            [120, 180, 240, 300, 360, 480],
            index=4
        )
        credit_history = st.selectbox("Credit History", config.FEATURE_OPTIONS['Credit_History'])
        property_area = st.selectbox("Property Area", config.FEATURE_OPTIONS['Property_Area'])
    
    # Predict button
    if st.button("🔮 Predict Loan Eligibility", type="primary", use_container_width=True):
        # Prepare input
        input_data = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Property_Area': property_area
        }
        
        # Validate input
        is_valid, message = validate_prediction_input(input_data, config.ALL_FEATURES)
        
        if not is_valid:
            st.error(f"❌ Validation Error: {message}")
            return
        
        # Make prediction
        try:
            with st.spinner("Making prediction..."):
                result = predict_single(input_data, model, preprocessor, config.ALL_FEATURES)
            
            # Store in session state
            st.session_state['last_prediction'] = result
            st.session_state['last_input'] = input_data
            
            # Display result
            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Result badge
                if result['prediction'] == 'Approved':
                    st.markdown(f"""
                    <div class="approved-badge">
                        ✅ LOAN APPROVED
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="rejected-badge">
                        ❌ LOAN REJECTED
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"**Confidence:** {result['confidence']:.1%}")
                st.markdown(f"**Approval Probability:** {result['probability_approved']:.1%}")
                st.markdown(f"**Rejection Probability:** {result['probability_rejected']:.1%}")
            
            with col2:
                # Gauge chart
                fig = create_gauge_chart(result['probability_approved'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.markdown("### 💡 Interpretation")
            interpretation = interpret_prediction(result, input_data)
            st.markdown(interpretation)
            
            # Log prediction
            logger.info(f"Prediction made: {result['prediction']} (Confidence: {result['confidence']:.2%})")
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")
            logger.error(f"Prediction error: {e}")


def explain_result_page():
    """Explain prediction using SHAP"""
    st.title("💡 Explain My Result")
    st.markdown("Understand which factors influenced the loan decision")
    
    model, preprocessor = load_model()
    
    if model is None:
        st.warning("⚠️ Model not available. Please train a model from the Admin panel.")
        return
    
    # Check if there's a previous prediction
    if 'last_prediction' not in st.session_state or 'last_input' not in st.session_state:
        st.info("ℹ️ No recent prediction found. Please make a prediction first from the 'Loan Prediction' page.")
        return
    
    result = st.session_state['last_prediction']
    input_data = st.session_state['last_input']
    
    # Display prediction summary
    st.markdown("### 📊 Prediction Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Decision", result['prediction'])
    
    with col2:
        st.metric("Confidence", f"{result['confidence']:.1%}")
    
    with col3:
        st.metric("Approval Probability", f"{result['probability_approved']:.1%}")
    
    st.markdown("---")
    
    # SHAP Explanation
    st.markdown("### 🔍 Feature Contributions (SHAP Analysis)")
    
    try:
        # Load data for explainer
        df = load_data()
        if df is not None:
            X, y = prepare_features(df)
            
            # Create explainer
            with st.spinner("Generating SHAP explanations..."):
                explainer = LoanExplainer(model, preprocessor, X.head(100))
                
                # Prepare input DataFrame
                input_df = pd.DataFrame([input_data], columns=config.ALL_FEATURES)
                
                # Get explanation
                explanation = explainer.explain_prediction(input_df, config.ALL_FEATURES)
                
                if explanation:
                    # Create waterfall plot
                    import matplotlib.pyplot as plt
                    fig = explainer.plot_waterfall(explanation, max_display=10, show=False)
                    
                    if fig:
                        st.pyplot(fig)
                        plt.close()
                    
                    # Feature contributions table
                    st.markdown("### 📋 Feature Contributions")
                    
                    shap_values = explanation['shap_values']
                    feature_names = explanation['feature_names']
                    
                    # Create DataFrame
                    contrib_df = pd.DataFrame({
                        'Feature': feature_names[:len(shap_values)],
                        'SHAP Value': shap_values,
                        'Impact': ['Positive' if v > 0 else 'Negative' for v in shap_values]
                    })
                    contrib_df['Abs SHAP'] = np.abs(contrib_df['SHAP Value'])
                    contrib_df = contrib_df.sort_values('Abs SHAP', ascending=False).head(10)
                    
                    st.dataframe(contrib_df[['Feature', 'SHAP Value', 'Impact']], use_container_width=True)
                    
                    st.info("""
                    **How to read SHAP values:**
                    - Positive values (green) push the prediction towards **Approval**
                    - Negative values (red) push the prediction towards **Rejection**
                    - Larger absolute values indicate stronger influence
                    """)
                else:
                    st.warning("Could not generate SHAP explanation. Showing basic interpretation instead.")
                    interpretation = interpret_prediction(result, input_data)
                    st.markdown(interpretation)
        else:
            st.warning("Data not available for SHAP analysis.")
    
    except Exception as e:
        st.error(f"Error generating explanation: {e}")
        logger.error(f"Explanation error: {e}")
        
        # Fallback to basic interpretation
        st.markdown("### 💡 Basic Interpretation")
        interpretation = interpret_prediction(result, input_data)
        st.markdown(interpretation)


def batch_prediction_page():
    """Batch prediction from CSV file"""
    st.title("📦 Batch Prediction")
    st.markdown("Upload a CSV file to get predictions for multiple applicants")
    
    model, preprocessor = load_model()
    
    if model is None:
        st.warning("⚠️ Model not available. Please train a model from the Admin panel.")
        return
    
    # Download template
    st.markdown("### 📥 Step 1: Download Template")
    
    # Create template
    template_data = {
        'Gender': ['Male'],
        'Married': ['Yes'],
        'Dependents': ['1'],
        'Education': ['Graduate'],
        'Self_Employed': ['No'],
        'ApplicantIncome': [5000],
        'CoapplicantIncome': [2000],
        'LoanAmount': [150],
        'Loan_Amount_Term': [360],
        'Credit_History': ['Yes'],
        'Property_Area': ['Urban']
    }
    template_df = pd.DataFrame(template_data)
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    template_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="📄 Download CSV Template",
        data=csv_data,
        file_name="loan_application_template.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.info("""
    **Template Instructions:**
    - Fill in the applicant details in the CSV file
    - Ensure all required columns are present
    - Use the exact column names and value formats
    - Credit_History should be 'Yes' or 'No'
    """)
    
    st.markdown("---")
    
    # Upload file
    st.markdown("### 📤 Step 2: Upload Filled CSV")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            batch_df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(batch_df)} applications.")
            
            # Show preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(batch_df.head(10), use_container_width=True)
            
            # Predict button
            if st.button("🔮 Make Batch Predictions", type="primary", use_container_width=True):
                with st.spinner(f"Making predictions for {len(batch_df)} applications..."):
                    try:
                        # Make predictions
                        result_df = predict_batch(batch_df, model, preprocessor)
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("### 🎯 Prediction Results")
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        
                        total = len(result_df)
                        approved = (result_df['Predicted_Loan_Status'] == 'Y').sum()
                        rejected = (result_df['Predicted_Loan_Status'] == 'N').sum()
                        
                        with col1:
                            st.metric("Total Applications", total)
                        
                        with col2:
                            st.metric("Approved", f"{approved} ({approved/total:.1%})")
                        
                        with col3:
                            st.metric("Rejected", f"{rejected} ({rejected/total:.1%})")
                        
                        # Show results
                        st.markdown("### 📊 Detailed Results")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Download results
                        csv_buffer = io.StringIO()
                        result_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download Results",
                            data=csv_data,
                            file_name="loan_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Log batch prediction
                        logger.info(f"Batch prediction completed: {total} applications, {approved} approved, {rejected} rejected")
                    
                    except Exception as e:
                        st.error(f"❌ Error making predictions: {e}")
                        logger.error(f"Batch prediction error: {e}")
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {e}")


def admin_page():
    """Admin panel for model training and system management"""
    st.title("⚙️ Admin / Settings")
    st.markdown("Manage models, data, and system settings")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Model Training", "Data Management", "System Info", "Logs"])
    
    with tab1:
        st.markdown("### 🤖 Model Training")
        
        st.info("""
        Train or retrain the loan eligibility prediction model.
        This will:
        - Load the existing dataset
        - Train both Logistic Regression and Random Forest models
        - Evaluate and select the best model
        - Save the model for predictions
        """)
        
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    # Load data
                    df = load_data()
                    
                    if df is None:
                        st.error("❌ No data available. Please generate data first.")
                        return
                    
                    # Prepare features
                    X, y = prepare_features(df)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
                    )
                    
                    st.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
                    
                    # Create and fit preprocessor
                    preprocessor = create_preprocessing_pipeline(
                        config.NUMERIC_FEATURES,
                        config.CATEGORICAL_FEATURES
                    )
                    preprocessor.fit(X_train)
                    
                    # Train models
                    models, best_model_name = train_and_compare_models(
                        X_train, X_test, y_train, y_test, preprocessor
                    )
                    
                    # Save best model
                    best_model = models[best_model_name]['model']
                    best_model.save(config.MODEL_PATH, config.PREPROCESSOR_PATH)
                    
                    # Display results
                    st.success(f"✅ Training completed! Best model: **{best_model_name}**")
                    
                    # Show metrics
                    metrics = models[best_model_name]['metrics']
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                    
                    with col2:
                        st.metric("Precision", f"{metrics['precision']:.2%}")
                    
                    with col3:
                        st.metric("Recall", f"{metrics['recall']:.2%}")
                    
                    with col4:
                        st.metric("F1-Score", f"{metrics['f1_score']:.2%}")
                    
                    with col5:
                        st.metric("ROC AUC", f"{metrics['roc_auc']:.2%}")
                    
                    # Clear cache to reload model
                    st.cache_resource.clear()
                    
                    logger.info(f"Model training completed: {best_model_name}")
                
                except Exception as e:
                    st.error(f"❌ Error training model: {e}")
                    logger.error(f"Training error: {e}")
    
    with tab2:
        st.markdown("### 📊 Data Management")
        
        # Current data info
        df = load_data()
        
        if df is not None:
            summary = get_data_summary(df)
            st.success(f"✅ Data loaded: {summary['total_records']} records")
        else:
            st.warning("⚠️ No data file found")
        
        st.markdown("---")
        
        # Generate new data
        st.markdown("#### Generate Synthetic Data")
        
        n_samples = st.number_input(
            "Number of samples to generate",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100
        )
        
        if st.button("🎲 Generate Data", use_container_width=True):
            with st.spinner(f"Generating {n_samples} synthetic records..."):
                try:
                    df = generate_synthetic_loan_data(n_samples=n_samples)
                    save_loan_data(df, config.LOAN_DATA_PATH)
                    
                    st.success(f"✅ Generated and saved {len(df)} records!")
                    
                    # Show summary
                    summary = get_data_summary(df)
                    st.info(f"Approval rate: {summary['approval_rate']:.1%}")
                    
                    # Clear cache
                    st.cache_data.clear()
                    
                    logger.info(f"Generated {n_samples} synthetic records")
                
                except Exception as e:
                    st.error(f"❌ Error generating data: {e}")
                    logger.error(f"Data generation error: {e}")
    
    with tab3:
        st.markdown("### ℹ️ System Information")
        
        # Model info
        model, preprocessor = load_model()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Status")
            if model is not None:
                st.success("✅ Model loaded")
                model_type = type(model).__name__
                st.info(f"Model type: {model_type}")
            else:
                st.error("❌ Model not found")
        
        with col2:
            st.markdown("#### Data Status")
            if config.LOAN_DATA_PATH.exists():
                st.success("✅ Data file exists")
                df = load_data()
                if df is not None:
                    st.info(f"Records: {len(df)}")
            else:
                st.error("❌ Data file not found")
        
        st.markdown("---")
        
        # File paths
        st.markdown("#### File Paths")
        st.code(f"""
Model Path: {config.MODEL_PATH}
Preprocessor Path: {config.PREPROCESSOR_PATH}
Data Path: {config.LOAN_DATA_PATH}
Log Path: {config.LOG_FILE_PATH}
        """)
        
        # Clear cache
        st.markdown("---")
        st.markdown("#### Cache Management")
        
        if st.button("🗑️ Clear All Caches", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ All caches cleared!")
    
    with tab4:
        st.markdown("### 📝 Recent Logs")
        
        try:
            if config.LOG_FILE_PATH.exists():
                with open(config.LOG_FILE_PATH, 'r') as f:
                    logs = f.readlines()
                
                # Show last 50 lines
                recent_logs = logs[-50:]
                
                st.text_area(
                    "Log Output",
                    value=''.join(recent_logs),
                    height=400
                )
            else:
                st.info("No log file found yet.")
        
        except Exception as e:
            st.error(f"Error reading logs: {e}")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application"""
    
    # Sidebar navigation
    st.sidebar.title("🏦 Navigation")
    
    page = st.sidebar.radio(
        "Go to",
        [
            "🏠 Home",
            "📊 Data Exploration",
            "📈 Model Performance",
            "🎯 Loan Prediction",
            "💡 Explain My Result",
            "📦 Batch Prediction",
            "⚙️ Admin / Settings"
        ]
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Quick Info")
    
    # Model status
    model, _ = load_model()
    if model is not None:
        st.sidebar.success("✅ Model Ready")
    else:
        st.sidebar.error("❌ Model Not Found")
    
    # Data status
    df = load_data()
    if df is not None:
        st.sidebar.info(f"📊 {len(df)} records loaded")
    else:
        st.sidebar.warning("⚠️ No data loaded")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👨‍💻 About")
    st.sidebar.markdown("""
    **Loan Eligibility Predictor**
    
    Built with:
    - Streamlit
    - scikit-learn
    - SHAP
    - Plotly
    
    © 2025 Mayank Sharma
    """)
    
    # Add watermark
    add_watermark()
    
    # Route to pages
    if page == "🏠 Home":
        home_page()
    elif page == "📊 Data Exploration":
        data_exploration_page()
    elif page == "📈 Model Performance":
        model_performance_page()
    elif page == "🎯 Loan Prediction":
        loan_prediction_page()
    elif page == "💡 Explain My Result":
        explain_result_page()
    elif page == "📦 Batch Prediction":
        batch_prediction_page()
    elif page == "⚙️ Admin / Settings":
        admin_page()


if __name__ == "__main__":
    main()
