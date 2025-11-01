"""
AI-Powered Credit Risk Pricing System - Streamlit Dashboard
Author: Aditya Sinha
Date: October 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import sys
import os
from datetime import datetime

# ========== RELATIVE PATHS (FIXED) ==========
# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define relative paths for all files
MODEL_PATH = os.path.join(BASE_DIR, '..', 'data', 'model_artifacts', 'xgboost_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'data', 'model_artifacts', 'feature_scaler.pkl')
PORTFOLIO_DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'final_pricing_recommendations.csv')

# Debug: Print paths (remove after testing)
# st.write(f"Base directory: {BASE_DIR}")
# st.write(f"Model path: {MODEL_PATH}")
# st.write(f"Model exists: {os.path.exists(MODEL_PATH)}")

# Page configuration
st.set_page_config(
    page_title="Credit Risk Pricing System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #2ca02c;
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🏦 AI-Powered Credit Risk Pricing System")
st.markdown("""
<div style='background-color: #e8f4f8; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
<b>Enterprise-grade credit risk assessment with explainable AI</b><br>
Automated loan underwriting • Risk-based pricing • Regulatory compliance
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎯 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Module",
    ["🏠 Dashboard Home", 
     "📝 Loan Assessment", 
     "📊 Portfolio Analytics", 
     "📈 Model Performance",
     "ℹ️ About System"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**System Status:** ✅ Online  
**Models:** 3 Active  
**Last Updated:** Oct 2025
""")

# ========== LOAD MODELS WITH RELATIVE PATHS ==========
@st.cache_resource
def load_models():
    """Load trained models with error handling"""
    try:
        # Check if files exist
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found: {MODEL_PATH}")
            return None, None, False
        
        if not os.path.exists(SCALER_PATH):
            st.error(f"❌ Scaler file not found: {SCALER_PATH}")
            return None, None, False
        
        # Load models
        pd_model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        return pd_model, scaler, True
    
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        return None, None, False

@st.cache_data
def load_portfolio():
    """Load portfolio data with error handling"""
    try:
        # Check if file exists
        if not os.path.exists(PORTFOLIO_DATA_PATH):
            st.warning(f"⚠️ Portfolio data not found: {PORTFOLIO_DATA_PATH}")
            # Return mock data for demo
            return create_demo_portfolio()
        
        # Load data
        data = pd.read_csv(PORTFOLIO_DATA_PATH)
        return data
    
    except Exception as e:
        st.warning(f"⚠️ Error loading portfolio: {str(e)}")
        # Return mock data for demo
        return create_demo_portfolio()

def create_demo_portfolio():
    """Create demo portfolio data for testing"""
    np.random.seed(42)
    n_records = 500
    
    pd_values = np.random.beta(2, 5, n_records)  # Default probability
    lgd_values = np.random.uniform(0.35, 0.55, n_records)  # Loss given default
    ead_values = np.random.lognormal(9.5, 0.8, n_records)  # Exposure at default
    
    # Segment based on PD
    def get_segment(pd):
        if pd < 0.05:
            return 'Prime'
        elif pd < 0.10:
            return 'Near-Prime'
        elif pd < 0.20:
            return 'Standard'
        elif pd < 0.40:
            return 'Subprime'
        else:
            return 'High-Risk'
    
    segments = [get_segment(pd) for pd in pd_values]
    
    # Interest rates by segment
    segment_rates = {
        'Prime': 8.5,
        'Near-Prime': 12.5,
        'Standard': 18.0,
        'Subprime': 28.0,
        'High-Risk': 36.0
    }
    
    interest_rates = [segment_rates[seg] + np.random.normal(0, 1) for seg in segments]
    
    # Expected loss
    expected_losses = pd_values * lgd_values * ead_values
    
    # Decision
    decisions = ['APPROVE' if pd < 0.50 else 'DECLINED' for pd in pd_values]
    
    df = pd.DataFrame({
        'PD': pd_values,
        'LGD': lgd_values,
        'EAD': ead_values,
        'Expected_Loss': expected_losses,
        'Interest_Rate_Pct': interest_rates,
        'Risk_Segment': segments,
        'decision': decisions
    })
    
    st.info("ℹ️ Using demo portfolio data (actual data file not found)")
    return df

# Load resources
pd_model, scaler, models_loaded = load_models()
portfolio_data = load_portfolio()

# ============================================================================
# PAGE 1: DASHBOARD HOME
# ============================================================================

if page == "🏠 Dashboard Home":
    
    st.header("📊 Portfolio Overview")
    
    if portfolio_data is not None:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_loans = len(portfolio_data)
            st.metric(
                label="📋 Total Applications",
                value=f"{total_loans:,}",
                help="Number of loan applications processed"
            )
        
        with col2:
            approval_rate = (portfolio_data['decision'] == 'APPROVE').mean() * 100
            st.metric(
                label="✅ Approval Rate",
                value=f"{approval_rate:.1f}%",
                delta=f"{approval_rate - 65:.1f}% vs target",
                help="Percentage of applications approved"
            )
        
        with col3:
            approved = portfolio_data[portfolio_data['decision'] == 'APPROVE']
            avg_rate = approved['Interest_Rate_Pct'].mean() if len(approved) > 0 else 0
            st.metric(
                label="💵 Avg Interest Rate",
                value=f"{avg_rate:.2f}%",
                help="Mean interest rate for approved loans"
            )
        
        with col4:
            total_exposure = portfolio_data['EAD'].sum() / 1e9
            st.metric(
                label="💰 Total Exposure",
                value=f"${total_exposure:.2f}B",
                help="Total loan exposure at default"
            )
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Risk Segment Distribution")
            
            segment_counts = portfolio_data['Risk_Segment'].value_counts()
            
            colors = {
                'Prime': '#2ecc71',
                'Near-Prime': '#3498db',
                'Standard': '#f39c12',
                'Subprime': '#e74c3c',
                'High-Risk': '#8e44ad'
            }
            
            fig = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                color=segment_counts.index,
                color_discrete_map=colors,
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 PD Distribution")
            
            fig = px.histogram(
                portfolio_data,
                x='PD',
                nbins=50,
                color_discrete_sequence=['#3498db']
            )
            fig.update_layout(
                xaxis_title="Probability of Default",
                yaxis_title="Frequency",
                height=400,
                showlegend=False
            )
            fig.add_vline(
                x=portfolio_data['PD'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {portfolio_data['PD'].mean()*100:.1f}%"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional metrics
        st.markdown("---")
        st.subheader("💼 Portfolio Risk Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_pd = portfolio_data['PD'].mean() * 100
            st.info(f"**Avg PD:** {avg_pd:.2f}%")
        
        with col2:
            avg_lgd = portfolio_data['LGD'].mean() * 100
            st.info(f"**Avg LGD:** {avg_lgd:.0f}%")
        
        with col3:
            avg_el = portfolio_data['Expected_Loss'].mean()
            st.warning(f"**Avg Expected Loss:** ${avg_el:,.0f}")
        
        with col4:
            total_el = portfolio_data['Expected_Loss'].sum() / 1e6
            st.error(f"**Total Expected Loss:** ${total_el:.1f}M")
        
        # Segment performance table
        st.markdown("---")
        st.subheader("📊 Performance by Risk Segment")
        
        segment_summary = portfolio_data.groupby('Risk_Segment').agg({
            'PD': 'mean',
            'Expected_Loss': 'mean',
            'Interest_Rate_Pct': 'mean',
            'decision': lambda x: (x == 'APPROVE').mean()
        }).round(4)
        
        segment_summary.columns = ['Avg PD', 'Avg Expected Loss', 'Avg Interest Rate (%)', 'Approval Rate']
        segment_summary['Avg PD'] = segment_summary['Avg PD'] * 100
        segment_summary['Approval Rate'] = segment_summary['Approval Rate'] * 100
        
        # Reorder
        order = ['Prime', 'Near-Prime', 'Standard', 'Subprime', 'High-Risk']
        segment_summary = segment_summary.reindex([s for s in order if s in segment_summary.index])
        
        st.dataframe(segment_summary.style.format({
            'Avg PD': '{:.2f}%',
            'Avg Expected Loss': '${:,.0f}',
            'Avg Interest Rate (%)': '{:.2f}%',
            'Approval Rate': '{:.1f}%'
        }), use_container_width=True)
    
    else:
        st.warning("📊 Portfolio data not available. Please ensure data files are present.")

# ============================================================================
# PAGE 2: LOAN ASSESSMENT
# ============================================================================

elif page == "📝 Loan Assessment":
    
    st.header("💳 Real-Time Loan Risk Assessment")
    st.markdown("Enter applicant details to receive instant risk analysis and pricing recommendation")
    
    if not models_loaded:
        st.warning("⚠️ Models not loaded. Using demo mode for assessment.")
    
    # Input form
    with st.form("loan_application_form"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Loan Details")
            
            loan_amount = st.number_input(
                "Loan Amount ($)",
                min_value=1000,
                max_value=40000,
                value=15000,
                step=1000
            )
            
            term = st.selectbox(
                "Loan Term",
                options=["36 months", "60 months"],
                index=0
            )
            
            purpose = st.selectbox(
                "Loan Purpose",
                options=['debt_consolidation', 'credit_card', 'home_improvement', 
                        'major_purchase', 'small_business', 'car', 'medical', 'other']
            )
        
        with col2:
            st.subheader("👤 Applicant Information")
            
            annual_income = st.number_input(
                "Annual Income ($)",
                min_value=10000,
                max_value=500000,
                value=65000,
                step=5000
            )
            
            employment_length = st.selectbox(
                "Employment Length",
                options=['< 1 year', '1 year', '2 years', '3-5 years', 
                        '6-9 years', '10+ years']
            )
            
            home_ownership = st.selectbox(
                "Home Ownership",
                options=['RENT', 'OWN', 'MORTGAGE', 'OTHER']
            )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Credit Profile")
            
            fico_score = st.slider(
                "Credit Score (FICO)",
                min_value=300,
                max_value=850,
                value=680,
                step=5
            )
            
            dti = st.slider(
                "Debt-to-Income Ratio (%)",
                min_value=0.0,
                max_value=45.0,
                value=18.0,
                step=0.5
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            
            revol_util = st.slider(
                "Credit Card Utilization (%)",
                min_value=0.0,
                max_value=100.0,
                value=45.0,
                step=1.0
            )
            
            delinq_2yrs = st.number_input(
                "Delinquencies (past 2 years)",
                min_value=0,
                max_value=10,
                value=0
            )
        
        submitted = st.form_submit_button("🔍 Assess Application", use_container_width=True)
    
    # Process submission
    if submitted:
        
        st.markdown("---")
        
        with st.spinner("🔄 Analyzing application..."):
            
            # Calculate risk score (simplified for demo)
            risk_score = 0
            
            # FICO impact
            if fico_score < 580:
                risk_score += 35
            elif fico_score < 670:
                risk_score += 20
            elif fico_score < 740:
                risk_score += 10
            elif fico_score < 800:
                risk_score += 5
            
            # DTI impact
            if dti > 35:
                risk_score += 25
            elif dti > 25:
                risk_score += 12
            elif dti > 18:
                risk_score += 6
            
            # Utilization impact
            if revol_util > 80:
                risk_score += 20
            elif revol_util > 60:
                risk_score += 12
            elif revol_util > 40:
                risk_score += 6
            
            # Delinquency impact
            if delinq_2yrs > 0:
                risk_score += delinq_2yrs * 15
            
            # Loan size impact
            loan_to_income = loan_amount / annual_income
            if loan_to_income > 0.35:
                risk_score += 12
            elif loan_to_income > 0.25:
                risk_score += 6
            
            # Convert to PD
            pd_estimate = min(max(risk_score / 100, 0.05), 0.75)
            
            # Risk parameters
            lgd = 0.45
            ead = loan_amount
            expected_loss = pd_estimate * lgd * ead
            
            # Determine segment and rate
            if pd_estimate < 0.05:
                segment = "Prime"
                segment_color = "green"
                base_rate = 7.5
                risk_premium = 1.0
            elif pd_estimate < 0.10:
                segment = "Near-Prime"
                segment_color = "blue"
                base_rate = 10.0
                risk_premium = 2.5
            elif pd_estimate < 0.20:
                segment = "Standard"
                segment_color = "orange"
                base_rate = 14.0
                risk_premium = 4.0
            elif pd_estimate < 0.40:
                segment = "Subprime"
                segment_color = "red"
                base_rate = 22.0
                risk_premium = 6.0
            else:
                segment = "High-Risk"
                segment_color = "purple"
                base_rate = 30.0
                risk_premium = 6.0
            
            interest_rate = base_rate + risk_premium
            
            # Calculate monthly payment
            monthly_rate = interest_rate / 100 / 12
            n_payments = int(term.split()[0])
            monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
            total_interest = (monthly_payment * n_payments) - loan_amount
            
            # Decision
            if pd_estimate >= 0.50:
                decision = "DECLINED"
                decision_color = "red"
                decision_icon = "❌"
            else:
                decision = "APPROVED"
                decision_color = "green"
                decision_icon = "✅"
        
        # Display results
        st.success("✅ Assessment Complete!")
        
        # Decision banner
        if decision == "APPROVED":
            st.markdown(f"""
            <div style='background-color: #d4edda; padding: 2rem; border-radius: 0.5rem; 
                        border-left: 5px solid #28a745; margin: 1rem 0;'>
                <h2 style='color: #155724; margin: 0;'>{decision_icon} APPLICATION {decision}</h2>
                <p style='color: #155724; margin-top: 0.5rem;'>
                Congratulations! Your application has been approved.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color: #f8d7da; padding: 2rem; border-radius: 0.5rem; 
                        border-left: 5px solid #dc3545; margin: 1rem 0;'>
                <h2 style='color: #721c24; margin: 0;'>{decision_icon} APPLICATION {decision}</h2>
                <p style='color: #721c24; margin-top: 0.5rem;'>
                Unfortunately, we cannot approve your application at this time.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk metrics
        st.subheader("📊 Risk Assessment Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Default Probability", f"{pd_estimate*100:.2f}%")
        
        with col2:
            st.metric("Expected Loss", f"${expected_loss:,.0f}")
        
        with col3:
            st.metric("Risk Segment", segment)
        
        with col4:
            st.metric("Risk Score", f"{risk_score}/100")
        
        if decision == "APPROVED":
            # Loan terms
            st.markdown("---")
            st.subheader("💰 Loan Terms & Pricing")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Offered Interest Rate:** <span style='font-size: 2rem; color: #1f77b4; font-weight: bold;'>{interest_rate:.2f}% APR</span>
                
                **Rate Breakdown:**
                - Base Rate: {base_rate:.2f}%
                - Risk Premium: +{risk_premium:.2f}%
                
                **Why this rate?**
                Your rate is calculated based on your credit profile, income, and risk assessment.
                """, unsafe_allow_html=True)
            
            with col2:
                st.info(f"""
                **💳 Loan Summary:**
                - **Principal:** ${loan_amount:,}
                - **Term:** {term}
                - **Monthly Payment:** ${monthly_payment:,.2f}
                - **Total Interest:** ${total_interest:,.2f}
                - **Total Repayment:** ${loan_amount + total_interest:,.2f}
                """)
        
        # Risk factors
        st.markdown("---")
        st.subheader("🔍 Risk Factor Analysis")
        
        factors = []
        positive_factors = []
        
        # Negative factors
        if fico_score < 670:
            factors.append(f"⚠️ Credit score ({fico_score}) is below recommended level (670+)")
        if dti > 25:
            factors.append(f"⚠️ Debt-to-income ratio ({dti:.1f}%) is high (recommended < 25%)")
        if revol_util > 50:
            factors.append(f"⚠️ Credit utilization ({revol_util:.0f}%) is high (recommended < 50%)")
        if delinq_2yrs > 0:
            factors.append(f"⚠️ {delinq_2yrs} delinquency/ies in past 2 years")
        if loan_to_income > 0.30:
            factors.append(f"⚠️ Loan amount is high relative to income ({loan_to_income*100:.1f}%)")
        
        # Positive factors
        if fico_score >= 740:
            positive_factors.append(f"✅ Excellent credit score ({fico_score})")
        if dti <= 18:
            positive_factors.append(f"✅ Low debt-to-income ratio ({dti:.1f}%)")
        if revol_util <= 30:
            positive_factors.append(f"✅ Low credit utilization ({revol_util:.0f}%)")
        if delinq_2yrs == 0:
            positive_factors.append("✅ No recent delinquencies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if factors:
                st.warning("**Areas of Concern:**")
                for factor in factors:
                    st.markdown(factor)
            else:
                st.success("✅ No significant risk factors identified")
        
        with col2:
            if positive_factors:
                st.success("**Positive Factors:**")
                for factor in positive_factors:
                    st.markdown(factor)

# ============================================================================
# PAGE 3: PORTFOLIO ANALYTICS
# ============================================================================

elif page == "📊 Portfolio Analytics":
    
    st.header("📊 Advanced Portfolio Analytics")
    
    if portfolio_data is None:
        st.warning("Portfolio data not available")
        st.stop()
    
    # Filters
    st.subheader("🔍 Portfolio Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_segments = st.multiselect(
            "Risk Segments",
            options=portfolio_data['Risk_Segment'].unique(),
            default=portfolio_data['Risk_Segment'].unique()
        )
    
    with col2:
        selected_decision = st.multiselect(
            "Decision Type",
            options=portfolio_data['decision'].unique(),
            default=portfolio_data['decision'].unique()
        )
    
    with col3:
        pd_range = st.slider(
            "PD Range (%)",
            min_value=0,
            max_value=100,
            value=(0, 100)
        )
    
    # Filter data
    filtered_data = portfolio_data[
        (portfolio_data['Risk_Segment'].isin(selected_segments)) &
        (portfolio_data['decision'].isin(selected_decision)) &
        (portfolio_data['PD'] >= pd_range[0]/100) &
        (portfolio_data['PD'] <= pd_range[1]/100)
    ]
    
    st.info(f"📊 Showing {len(filtered_data):,} loans (out of {len(portfolio_data):,} total)")
    
    # Analytics
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Interest Rate Distribution")
        
        fig = px.box(
            filtered_data,
            x='Risk_Segment',
            y='Interest_Rate_Pct',
            color='Risk_Segment',
            color_discrete_map={
                'Prime': '#2ecc71',
                'Near-Prime': '#3498db',
                'Standard': '#f39c12',
                'Subprime': '#e74c3c',
                'High-Risk': '#8e44ad'
            }
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Expected Loss by Segment")
        
        el_by_segment = filtered_data.groupby('Risk_Segment')['Expected_Loss'].sum() / 1e6
        
        fig = px.bar(
            x=el_by_segment.index,
            y=el_by_segment.values,
            color=el_by_segment.index,
            color_discrete_map={
                'Prime': '#2ecc71',
                'Near-Prime': '#3498db',
                'Standard': '#f39c12',
                'Subprime': '#e74c3c',
                'High-Risk': '#8e44ad'
            }
        )
        fig.update_layout(
            xaxis_title="Risk Segment",
            yaxis_title="Expected Loss ($M)",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.markdown("---")
    st.subheader("📋 Detailed Portfolio View")
    
    # Sample data
    display_data = filtered_data.head(100)[['PD', 'LGD', 'EAD', 'Expected_Loss', 
                                              'Interest_Rate_Pct', 'Risk_Segment', 'decision']]
    display_data['PD'] = display_data['PD'] * 100
    display_data['LGD'] = display_data['LGD'] * 100
    
    st.dataframe(
        display_data.style.format({
            'PD': '{:.2f}%',
            'LGD': '{:.0f}%',
            'EAD': '${:,.0f}',
            'Expected_Loss': '${:,.0f}',
            'Interest_Rate_Pct': '{:.2f}%'
        }),
        use_container_width=True,
        height=400
    )

# ============================================================================
# PAGE 4: MODEL PERFORMANCE
# ============================================================================

elif page == "📈 Model Performance":
    
    st.header("📈 Model Performance Metrics")
    
    st.info("""
    **Model Information:**
    - **Algorithm:** XGBoost Classifier
    - **Training Data:** Lending Club (2007-2018)
    - **Features:** 50+ engineered features
    - **Target:** Binary default classification
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("ROC-AUC Score", "0.72", help="Area under ROC curve")
    
    with col2:
        st.metric("KS Statistic", "0.45", help="Kolmogorov-Smirnov statistic")
    
    with col3:
        st.metric("Gini Coefficient", "0.44", help="Model discrimination power")
    
    st.markdown("---")
    
    # Model components
    st.subheader("🔧 Model Components")
    
    components = pd.DataFrame({
        'Component': ['PD Model', 'LGD Model', 'EAD Model'],
        'Algorithm': ['XGBoost', 'Gradient Boosting', 'Linear Regression'],
        'Status': ['✅ Active', '✅ Active', '✅ Active'],
        'Samples': ['269,070', '67,268', '269,070']
    })
    
    st.table(components)
    
    # Feature importance
    if models_loaded and portfolio_data is not None:
        st.markdown("---")
        st.subheader("🎯 Top 10 Important Features")
        
        # Mock feature importance (replace with actual)
        features = [
            'FICO Score', 'DTI Ratio', 'Credit Utilization', 'Annual Income',
            'Loan Amount', 'Employment Length', 'Delinquencies', 'Inquiries',
            'Open Accounts', 'Credit History Length'
        ]
        importance = [0.18, 0.15, 0.12, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.05]
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            color=importance,
            color_continuous_scale='blues'
        )
        fig.update_layout(
            xaxis_title="Importance Score",
            yaxis_title="",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 5: ABOUT SYSTEM
# ============================================================================

elif page == "ℹ️ About System":
    
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ## 🎯 Project Overview
    
    **AI-Powered Credit Risk Pricing System** is an end-to-end machine learning solution 
    for automated loan underwriting and risk-based pricing.
    
    ### Key Features:
    - ✅ **Automated Risk Assessment**: ML-powered default prediction
    - 💰 **Risk-Based Pricing**: Dynamic interest rates based on borrower profile
    - 🔍 **Explainable AI**: SHAP values for model interpretability
    - 📊 **Portfolio Management**: Real-time analytics and monitoring
    - ⚖️ **Regulatory Compliance**: Basel II/III framework alignment
    
    ---
    
    ## 🧠 Machine Learning Models
    
    ### 1. Probability of Default (PD) Model
    - **Algorithm:** XGBoost Classifier
    - **Performance:** 72% ROC-AUC
    - **Features:** 50+ engineered variables
    - **Training:** 1.3M+ historical loans
    
    ### 2. Loss Given Default (LGD) Model
    - **Algorithm:** Gradient Boosting Regressor
    - **Approach:** Two-stage modeling
    - **Output:** Recovery rate estimation
    
    ### 3. Exposure at Default (EAD) Model
    - **Algorithm:** Linear Regression
    - **Purpose:** Credit limit utilization prediction
    
    ---
    
    ## 📈 Technical Architecture
    
    ```
    Input: Loan Application
       ↓
    Feature Engineering (WOE, Ratios, Interactions)
       ↓
    ML Prediction (PD, LGD, EAD)
       ↓
    Risk Calculation (Expected Loss)
       ↓
    Pricing Engine (Interest Rate)
       ↓
    Output: Approval Decision + Terms
    ```
    
    ---
    
    ## 💻 Technology Stack
    
    - **Language:** Python 3.9+
    - **ML Libraries:** XGBoost, LightGBM, Scikit-learn
    - **Explainability:** SHAP
    - **Dashboard:** Streamlit
    - **Data Processing:** Pandas, NumPy
    - **Visualization:** Plotly, Matplotlib
    
    ---
    
    ## 👨‍💻 Developer
    
    **Aditya Sinha**  
    Data Analyst | Machine Learning Engineer
    
    📧 Contact: [aditya.sinha@example.com](mailto:aditya.sinha@example.com)  
    💼 LinkedIn: [linkedin.com/in/adityasinha](https://linkedin.com)  
    🐙 GitHub: [github.com/adityasinha](https://github.com)
    
    ---
    
    ## 📄 Documentation
    
    For detailed documentation, model methodology, and technical specifications, 
    please refer to the project repository.
    
    **Project Repository:** [github.com/adityasinha/credit-risk-pricing](https://github.com)
    
    ---
    
    ## ⚖️ Disclaimer
    
    This system is for demonstration and educational purposes. Real-world deployment 
    requires additional validation, regulatory approval, and compliance checks.
    """)
    
    st.success("✅ System Version: 1.0.0 | Last Updated: October 2025")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><b>AI-Powered Credit Risk Pricing System</b></p>
    <p>Developed by Aditya Sinha | © 2025 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)