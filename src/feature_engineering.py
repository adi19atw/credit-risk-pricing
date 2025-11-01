"""
Feature Engineering Module for Credit Risk Modeling
Author: Aditya Sinha
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class CreditFeatureEngineer:
    """
    Feature engineering pipeline for credit risk modeling
    """
    
    def __init__(self):
        self.woe_mappings = {}
        self.selected_features = []
        
    def calculate_woe_iv(self, data: pd.DataFrame, feature: str, 
                         target: str, bins: int = 10) -> Tuple[float, pd.DataFrame]:
        """Calculate WOE and IV for a feature"""
        df_woe = data[[feature, target]].copy()
        
        # Bin continuous variables
        if df_woe[feature].dtype in ['int64', 'float64'] and df_woe[feature].nunique() > 10:
            try:
                df_woe['bin'] = pd.qcut(df_woe[feature], q=bins, duplicates='drop')
            except:
                df_woe['bin'] = pd.cut(df_woe[feature], bins=bins, duplicates='drop')
        else:
            df_woe['bin'] = df_woe[feature]
        
        # Calculate WOE and IV
        grouped = df_woe.groupby('bin', observed=True)[target].agg(['count', 'sum'])
        grouped.columns = ['Total', 'Events']
        grouped['Non-Events'] = grouped['Total'] - grouped['Events']
        
        total_events = grouped['Events'].sum()
        total_non_events = grouped['Non-Events'].sum()
        
        grouped['% Events'] = np.maximum(grouped['Events'], 0.5) / total_events
        grouped['% Non-Events'] = np.maximum(grouped['Non-Events'], 0.5) / total_non_events
        grouped['WOE'] = np.log(grouped['% Non-Events'] / grouped['% Events'])
        grouped['IV'] = (grouped['% Non-Events'] - grouped['% Events']) * grouped['WOE']
        
        iv_value = grouped['IV'].sum()
        grouped['Feature'] = feature
        grouped = grouped.reset_index()
        
        return iv_value, grouped
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived financial features"""
        df = df.copy()
        
        # Credit utilization
        if 'revol_util' in df.columns:
            df['revol_util_squared'] = df['revol_util'] ** 2
            df['high_utilization'] = (df['revol_util'] > 80).astype(int)
        
        # Debt burden ratios
        if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
            df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
            df['high_loan_burden'] = (df['loan_to_income'] > 0.3).astype(int)
        
        if 'installment' in df.columns and 'annual_inc' in df.columns:
            df['payment_to_income'] = (df['installment'] * 12) / (df['annual_inc'] + 1)
        
        # FICO score
        if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
            df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
        
        # Delinquency indicators
        if 'delinq_2yrs' in df.columns:
            df['has_delinquency'] = (df['delinq_2yrs'] > 0).astype(int)
        
        # Account ratios
        if 'open_acc' in df.columns and 'total_acc' in df.columns:
            df['open_acc_ratio'] = df['open_acc'] / (df['total_acc'] + 1)
        
        return df
    
    def fit(self, data: pd.DataFrame, target: str, iv_threshold: float = 0.02):
        """Fit feature engineering pipeline"""
        # Calculate IV for all features
        iv_results = []
        features = [col for col in data.columns if col != target]
        
        for feature in features:
            try:
                iv_value, woe_details = self.calculate_woe_iv(data, feature, target)
                iv_results.append({'Feature': feature, 'IV': iv_value})
                
                # Store WOE mappings
                self.woe_mappings[feature] = dict(zip(woe_details['bin'], woe_details['WOE']))
            except:
                continue
        
        # Select features based on IV threshold
        iv_df = pd.DataFrame(iv_results).sort_values('IV', ascending=False)
        self.selected_features = iv_df[iv_df['IV'] >= iv_threshold]['Feature'].tolist()
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations"""
        df = self.create_derived_features(data)
        return df[self.selected_features]


# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('../data/processed/lending_club_cleaned.csv')
    
    # Initialize feature engineer
    fe = CreditFeatureEngineer()
    
    # Fit and transform
    fe.fit(df, target='default', iv_threshold=0.02)
    df_transformed = fe.transform(df)
    
    print(f"Selected features: {len(fe.selected_features)}")
    print(f"Transformed data shape: {df_transformed.shape}")
