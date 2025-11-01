<!-- "# credit-risk-pricing"  -->
# 🏦 Credit Risk Assessment & Pricing Engine

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.25+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> End-to-end machine learning system for credit default prediction and risk-based pricing. Built using industry-standard PD/LGD/EAD framework with SHAP explainability for regulatory compliance.

---

## 🎯 Project Overview

This project implements a production-ready credit risk assessment platform that predicts loan default probability and calculates risk-adjusted interest rates. Built with 269,000+ Lending Club loan records, the system achieves 72% ROC-AUC and provides transparent, explainable decisions through SHAP analysis.

**Live Demo:** [Try the Streamlit Dashboard](https://your-streamlit-app-link.app/)

---

## 🔑 Key Features

- ✅ Default Prediction: XGBoost classifier with 72% ROC-AUC, KS=0.45
- ✅ Feature Engineering: 50+ predictive features using WOE/IV analysis
- ✅ Risk-Based Pricing: 5-tier segmentation (8-36% APR) with RAROC logic
- ✅ Explainability: SHAP force plots and waterfall charts for every decision
- ✅ Model Monitoring: Drift detection, backtesting, stress testing
- ✅ Production Dashboard: Real-time scoring, portfolio analytics, scenario analysis

---

## 📊 Results

| Metric | Value | Benchmark |
|--------|-------|-----------|
| ROC-AUC | 0.72 | 0.65-0.75 (Industry) |
| KS Statistic | 0.45 | >0.40 (Good) |
| Gini Coefficient | 0.44 | >0.40 (Strong) |
| Loans Analyzed | 269,070 | 12 years data |
| Processing Speed | 2-3 min/app | 80% faster than manual |

---

## 🛠️ Tech Stack

**Machine Learning:**
- Python 3.9+
- Scikit-learn, XGBoost, LightGBM
- SHAP (explainability)
- Imbalanced-learn (SMOTE for class imbalance)

**Data Processing:**
- Pandas, NumPy
- Databricks (ETL pipelines)

**Visualization & Deployment:**
- Streamlit (interactive dashboard)
- Plotly, Matplotlib, Seaborn (charting)
- Streamlit Cloud (hosting)

**Risk Framework:**
- PD/LGD/EAD modeling (Basel III compliant)
- RAROC decision logic

---

## 📁 Project Structure

```
credit-risk-assessment/
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb       # Data exploration & cleaning
│   ├── 02_feature_engineering.ipynb     # WOE/IV analysis
│   ├── 03_pd_model_training.ipynb       # XGBoost model development
│   ├── 04_lgd_ead_models.ipynb          # Loss & exposure modeling
│   ├── 05_shap_explainability.ipynb     # SHAP analysis
│   └── 06_pricing_engine.ipynb          # Risk-based pricing logic
│
├── src/
│   ├── data_preprocessing.py            # Data cleaning utilities
│   ├── feature_engineering.py           # WOE/IV feature transformer
│   ├── model_training.py                # Model training pipeline
│   ├── pricing_engine.py                # APR calculation & RAROC
│   └── utils.py                         # Helper functions
│
├── streamlit_app/
│   ├── app.py                           # Main Streamlit dashboard
│   ├── pages/
│   │   ├── 01_Portfolio_Overview.py
│   │   ├── 02_Live_Scoring.py
│   │   ├── 03_Explainability.py
│   │   └── 04_Model_Performance.py
│   └── assets/                          # Images, logos
│
├── data/
│   └── sample_loans.csv                 # Sample dataset (5K records)
│
├── models/
│   ├── xgboost_pd_model.pkl
│   ├── lgd_model.pkl
│   └── scaler.pkl
│
├── requirements.txt                     # Python dependencies
├── README.md                            # Documentation
└── LICENSE                              # MIT License
```

---

## 🚀 Quick Start

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/credit-risk-assessment.git
cd credit-risk-assessment
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run Streamlit Dashboard Locally

```bash
streamlit run streamlit_app/app.py
```

### Step 4: Explore Notebooks

```bash
jupyter notebook notebooks/
```

---

## 📖 Methodology

### Feature Engineering

- Analyzed 150+ raw features from Lending Club dataset
- Calculated Weight of Evidence (WOE) and Information Value (IV)
- Selected 50+ features with IV > 0.02
- Removed multicollinear pairs (correlation > 0.8)

Example features created:
- Loan-to-income ratio
- Credit utilization rate
- Payment-to-income ratio
- Delinquency history flags

### Model Training

- Data split: 60% train, 20% validation, 20% test (stratified)
- Handled class imbalance with SMOTE (balanced to 1:1 ratio)
- Trained XGBoost classifier with hyperparameters:
  - n_estimators=200
  - max_depth=5
  - learning_rate=0.05
  - subsample=0.8
  - colsample_bytree=0.8
  - early_stopping_rounds=20

Result: 72% ROC-AUC on test data

### Risk-Based Pricing

Formula:
```
APR = Risk_Free_Rate + EL_Rate + Operating_Cost + Profit_Margin + Risk_Premium
```

Components:
- Risk_Free_Rate: 3% (treasury benchmark)
- EL_Rate: (PD × LGD) / Loan_Term
- Operating_Cost: 2%
- Profit_Margin: 1%
- Risk_Premium: 1-8% (tiered by PD)

5-Tier Segmentation:
- Prime (PD < 5%): 8.5% APR
- Near-Prime (5-10%): 12.5% APR
- Standard (10-20%): 18% APR
- Subprime (20-40%): 28% APR
- High-Risk (>40%): 36% APR

### Explainability

- Used SHAP (SHapley Additive exPlanations) for model interpretability
- Generated force plots for individual predictions
- Created waterfall charts showing feature contributions
- Automated adverse action notices for declined applicants

---

## 📊 Dashboard Features

### Portfolio Overview
- Total loans, default rate, risk exposure
- Risk distribution by segment
- Monthly default trends

### Live Scoring
- Upload CSV or enter applicant data manually
- Instant default probability prediction
- Recommended APR and approval decision
- SHAP explanation plot

### Model Performance
- ROC curve and Precision-Recall curve
- Confusion matrix and KS plot
- Feature importance rankings
- Model drift monitoring

### Scenario Analysis
- Stress test portfolio under recession
- Adjust unemployment, interest rates, default rates
- Real-time impact visualization

---

## 💼 Business Impact

- 80% efficiency gain over manual underwriting
- Fair lending with transparent decisions
- Regulatory compliance with audit trail
- Risk-adjusted pricing ensures profitability (RAROC > 10%)

---

## 🔮 Future Enhancements

- Real-time API deployment (FastAPI/Flask)
- Multi-model comparison (CatBoost, LightGBM)
- Time-series behavioral analytics (LSTM)
- Integration with alternative data sources
- Automated model retraining (MLflow, Airflow)
- Advanced stress testing (Monte Carlo, copula models)

---

## 📚 References

- Basel III Framework: https://www.bis.org/bcbs/basel3.htm
- SHAP Paper: https://arxiv.org/abs/1705.07874
- Lending Club Data: https://www.kaggle.com/datasets/wordsforthewise/lending-club
- XGBoost Docs: https://xgboost.readthedocs.io/

---

## 👤 Author

**Aditya Sinha**

- Analyst @ Axtria
- IIT BHU (Dual Degree)
- FRM Part I Certified
- LinkedIn: https://linkedin.com/in/aditya-sinha
- GitHub: https://github.com/yourusername
- Email: sinhaditya2019@gmail.com

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- Lending Club for open loan data
- Streamlit for dashboard framework
- Open-source ML community

---

⭐ If you find this useful, please star the repository!
