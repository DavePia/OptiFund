import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

class DataPreprocessor:
    def __init__(self, stock_data_path=None, esg_data_path=None):
        """
        Initialize the preprocessor with paths to data files.
        
        Args:
            stock_data_path (str): Path to main OHLCV data CSV
            esg_data_path (str): Path to ESG Risk Ratings CSV
        """
        # Load from environment
        self.stock_data_path = stock_data_path or os.getenv('STOCK_DATA_PATH', 'stock_details_5_years.csv')
        self.esg_data_path = esg_data_path or os.getenv('ESG_DATA_PATH', 'SP 500 ESG Risk Ratings.csv')
        self.data = None
        self.returns = None
        self.cov_matrix = None
        self.esg_scores = None
        
    def load_and_clean_stock_data(self):
        """Load and clean the main stock dataset"""
        self.data = pd.read_csv(self.stock_data_path, parse_dates=['Date'])
        self.data = self.data.sort_values(['Company', 'Date'])
        
        # Handle missing values - forward fill then backfill for each company
        self.data = self.data.groupby('Company', group_keys=False).apply(lambda x: x.ffill().bfill())
        
        return self.data
    
    def load_and_clean_esg_data(self):
        """Load and clean the ESG dataset"""
        esg_data = pd.read_csv(self.esg_data_path)
        
        # Clean column names (remove spaces and make lowercase)
        esg_data.columns = [col.strip().replace(' ', '_').lower() for col in esg_data.columns]
        
        # We'll use 'symbol' as the key to merge with stock data
        esg_data = esg_data.rename(columns={'symbol': 'Company'})
        
        # Select relevant columns - we'll focus on total_esg_risk_score
        esg_cols = ['Company', 'total_esg_risk_score', 'esg_risk_level']
        esg_data = esg_data[esg_cols]
        
        # Convert risk score to numeric, coerce errors to NaN
        esg_data['total_esg_risk_score'] = pd.to_numeric(
            esg_data['total_esg_risk_score'], errors='coerce'
        )
        
        # For missing ESG scores, we'll fill with the median
        median_esg = esg_data['total_esg_risk_score'].median()
        esg_data['total_esg_risk_score'] = esg_data['total_esg_risk_score'].fillna(median_esg)
        
        # Normalize ESG scores (lower risk is better, so we invert)
        max_score = esg_data['total_esg_risk_score'].max()
        min_score = esg_data['total_esg_risk_score'].min()
        esg_data['esg_normalized'] = 1 - (
            (esg_data['total_esg_risk_score'] - min_score) / 
            (max_score - min_score + 1e-10)  # Avoid division by zero
        )
        
        return esg_data
    
    def merge_datasets(self):
        """Merge stock data with ESG data"""
        stock_data = self.load_and_clean_stock_data()
        esg_data = self.load_and_clean_esg_data()
        
        # Merge on company symbol
        merged_data = pd.merge(
            stock_data, 
            esg_data, 
            on='Company', 
            how='left'
        )
        
        # Fill any remaining missing ESG scores (companies not in ESG dataset)
        merged_data['esg_normalized'] = merged_data['esg_normalized'].fillna(0.5)  # Neutral score
        merged_data['total_esg_risk_score'] = merged_data['total_esg_risk_score'].fillna(
            esg_data['total_esg_risk_score'].median()
        )
        
        return merged_data
    
    def calculate_returns(self):
        """Calculate daily returns for each stock"""
        if self.data is None:
            self.data = self.merge_datasets()
            
        # Calculate returns
        returns = self.data.copy()
        returns['return'] = returns.groupby('Company')['Close'].pct_change()
        self.returns = returns.pivot(index='Date', columns='Company', values='return').dropna()
        
        return self.returns
    
    def calculate_covariance_matrix(self):
        """Calculate covariance matrix of returns"""
        if self.returns is None:
            self.calculate_returns()
            
        self.cov_matrix = self.returns.cov()
        return self.cov_matrix
    
    def get_esg_scores(self):
        """Get ESG scores for each company"""
        if self.data is None:
            self.data = self.merge_datasets()
            
        # Get the most recent ESG score for each company
        self.esg_scores = self.data.groupby('Company')['esg_normalized'].last()
        return self.esg_scores
    
    def preprocess_all(self):
        """Run all preprocessing steps"""
        self.data = self.merge_datasets()
        self.calculate_returns()
        self.calculate_covariance_matrix()
        self.get_esg_scores()
        
        return {
            'data': self.data,
            'returns': self.returns,
            'cov_matrix': self.cov_matrix,
            'esg_scores': self.esg_scores
        }