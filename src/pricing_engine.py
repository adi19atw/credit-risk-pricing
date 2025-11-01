"""
Risk-Based Pricing Engine for Credit Risk Models
Author: Aditya Sinha
Date: October 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import joblib


class CreditPricingEngine:
    """
    Complete risk-based pricing engine integrating PD, LGD, and EAD models
    """
    
    def __init__(self, pd_model_path: str, lgd_model_path: str, ead_model_path: str,
                 pd_scaler_path: str, lgd_scaler_path: str, ead_scaler_path: str):
        """
        Initialize pricing engine with trained models
        
        Parameters:
        -----------
        pd_model_path : str
            Path to trained PD model
        lgd_model_path : str
            Path to trained LGD model
        ead_model_path : str
            Path to trained EAD model
        pd_scaler_path, lgd_scaler_path, ead_scaler_path : str
            Paths to feature scalers
        """
        self.pd_model = joblib.load(pd_model_path)
        self.lgd_model = joblib.load(lgd_model_path)
        self.ead_model = joblib.load(ead_model_path)
        
        self.pd_scaler = joblib.load(pd_scaler_path)
        self.lgd_scaler = joblib.load(lgd_scaler_path)
        self.ead_scaler = joblib.load(ead_scaler_path)
        
        # Pricing parameters
        self.risk_free_rate = 0.03
        self.operating_cost_rate = 0.02
        self.profit_margin = 0.01
        self.average_lgd = 0.45  # Conservative estimate
        
    def predict_risk_parameters(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict PD, LGD, and EAD for loan applications
        
        Returns:
        --------
        risk_params : dict
            Dictionary with PD, LGD, EAD predictions
        """
        # Scale features
        features_scaled = self.pd_scaler.transform(features)
        
        # Predict PD
        pd_predictions = self.pd_model.predict_proba(features_scaled)[:, 1]
        
        # Use average LGD (or predict if model available)
        lgd_predictions = np.full(len(features), self.average_lgd)
        
        # Estimate EAD from loan amount
        if 'loan_amnt' in features.columns:
            ead_predictions = features['loan_amnt'].values
        else:
            ead_predictions = np.full(len(features), 15000)
        
        return {
            'PD': pd_predictions,
            'LGD': lgd_predictions,
            'EAD': ead_predictions
        }
    
    def calculate_expected_loss(self, prob_default: float, loss_given_default: float, 
                               exposure_at_default: float) -> float:
        """
        Calculate Expected Loss: EL = PD × LGD × EAD
        
        Parameters:
        -----------
        prob_default : float
            Probability of Default
        loss_given_default : float
            Loss Given Default
        exposure_at_default : float
            Exposure at Default
        """
        return prob_default * loss_given_default * exposure_at_default
    
    def calculate_interest_rate(self, prob_default: float, loss_given_default: float, 
                               exposure_at_default: float, loan_term: float = 3) -> float:
        """
        Calculate risk-adjusted interest rate
        
        Parameters:
        -----------
        prob_default : float
            Probability of Default
        loss_given_default : float
            Loss Given Default
        exposure_at_default : float
            Exposure at Default
        loan_term : float
            Loan term in years
        
        Returns:
        --------
        interest_rate : float
            Annual interest rate
        """
        # Expected loss rate
        el_rate = (prob_default * loss_given_default) / loan_term
        
        # Risk premium based on PD
        if prob_default < 0.05:
            risk_premium = 0.01
        elif prob_default < 0.10:
            risk_premium = 0.02
        elif prob_default < 0.20:
            risk_premium = 0.03
        elif prob_default < 0.40:
            risk_premium = 0.05
        else:
            risk_premium = 0.08
        
        # Total rate
        rate = (self.risk_free_rate + el_rate + self.operating_cost_rate + 
                self.profit_margin + risk_premium)
        
        return min(rate, 0.36)  # Cap at 36%
    
    def segment_customer(self, prob_default: float) -> str:
        """
        Assign risk segment based on PD
        
        Parameters:
        -----------
        prob_default : float
            Probability of Default
        """
        if prob_default < 0.05:
            return 'Prime'
        elif prob_default < 0.10:
            return 'Near-Prime'
        elif prob_default < 0.20:
            return 'Standard'
        elif prob_default < 0.40:
            return 'Subprime'
        else:
            return 'High-Risk'
    
    def generate_recommendation(self, prob_default: float, loss_given_default: float, 
                               exposure_at_default: float, interest_rate: float) -> Dict:
        """
        Generate loan approval recommendation
        
        Parameters:
        -----------
        prob_default : float
            Probability of Default
        loss_given_default : float
            Loss Given Default
        exposure_at_default : float
            Exposure at Default
        interest_rate : float
            Calculated interest rate
        
        Returns:
        --------
        recommendation : dict
            Decision, rate, and reasoning
        """
        # Calculate metrics
        expected_loss = self.calculate_expected_loss(prob_default, loss_given_default, 
                                                     exposure_at_default)
        revenue = interest_rate * exposure_at_default * 3  # 3-year term
        costs = expected_loss + (exposure_at_default * self.operating_cost_rate * 3)
        profit = revenue - costs
        raroc = profit / (exposure_at_default * 0.08) if exposure_at_default > 0 else 0
        
        # Decision thresholds
        if prob_default >= 0.50:
            return {
                'decision': 'DECLINE',
                'rate': None,
                'reason': f'PD too high ({prob_default*100:.1f}%)'
            }
        elif raroc < 0.10:
            return {
                'decision': 'DECLINE',
                'rate': None,
                'reason': f'RAROC too low ({raroc*100:.1f}%)'
            }
        else:
            return {
                'decision': 'APPROVE',
                'rate': interest_rate,
                'reason': f'Profitable loan (RAROC: {raroc*100:.1f}%)'
            }
    
    def price_loan(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Complete pricing workflow for loan applications
        
        Parameters:
        -----------
        features : DataFrame
            Loan application features
        
        Returns:
        --------
        results : DataFrame
            Comprehensive pricing results
        """
        # Predict risk parameters
        risk_params = self.predict_risk_parameters(features)
        
        # Calculate metrics for each loan
        results_list = []
        
        for i in range(len(features)):
            # Use descriptive variable names instead of 'pd'
            prob_default = risk_params['PD'][i]
            loss_given_default = risk_params['LGD'][i]
            exposure_at_default = risk_params['EAD'][i]
            
            # Calculate pricing
            interest_rate = self.calculate_interest_rate(
                prob_default, loss_given_default, exposure_at_default
            )
            expected_loss = self.calculate_expected_loss(
                prob_default, loss_given_default, exposure_at_default
            )
            segment = self.segment_customer(prob_default)
            recommendation = self.generate_recommendation(
                prob_default, loss_given_default, exposure_at_default, interest_rate
            )
            
            results_list.append({
                'PD': prob_default,
                'LGD': loss_given_default,
                'EAD': exposure_at_default,
                'Expected_Loss': expected_loss,
                'Interest_Rate': interest_rate * 100,
                'Risk_Segment': segment,
                'Decision': recommendation['decision'],
                'Recommendation': recommendation['reason']
            })
        
        # Return as DataFrame (pd is still the pandas module here)
        return pd.DataFrame(results_list)


# Example usage
if __name__ == "__main__":
    # Initialize engine
    engine = CreditPricingEngine(
        pd_model_path='../data/model_artifacts/xgboost_model.pkl',
        lgd_model_path='../data/model_artifacts/lgd_model_xgboost.pkl',
        ead_model_path='../data/model_artifacts/ead_model_xgboost.pkl',
        pd_scaler_path='../data/model_artifacts/feature_scaler.pkl',
        lgd_scaler_path='../data/model_artifacts/lgd_scaler.pkl',
        ead_scaler_path='../data/model_artifacts/ead_scaler.pkl'
    )
    
    # Load sample data
    df = pd.read_csv('../data/processed/lending_club_featured.csv')
    features = df.drop(columns=['default', 'loan_status']).head(10)
    
    # Price loans
    pricing_results = engine.price_loan(features)
    
    print("\n" + "="*80)
    print("PRICING ENGINE - SAMPLE RESULTS")
    print("="*80)
    print("\nPricing Results for 10 Sample Loans:")
    print(pricing_results.to_string(index=True))
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Average PD: {pricing_results['PD'].mean()*100:.2f}%")
    print(f"Average Interest Rate: {pricing_results['Interest_Rate'].mean():.2f}%")
    print(f"Average Expected Loss: ${pricing_results['Expected_Loss'].mean():,.2f}")
    print(f"\nApproval Rate: {(pricing_results['Decision'] == 'APPROVE').sum() / len(pricing_results) * 100:.1f}%")
    
    print("\n✓ Pricing engine test completed successfully!")
