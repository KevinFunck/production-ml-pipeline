"""
Data processing and feature engineering.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.logger import setup_logger
from config.config import PROCESSED_DATA_DIR

logger = setup_logger(__name__)


class DataProcessor:
    """Process and transform raw data."""
    
    def __init__(self, output_dir: Path = PROCESSED_DATA_DIR):
        """
        Initialize DataProcessor.
        
        Args:
            output_dir: Directory to save processed data
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.encoders = {}
        logger.info(f"DataProcessor initialized with output_dir: {output_dir}")
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """
        Handle missing values.
        
        Args:
            df: DataFrame with missing values
            strategy: Strategy to handle missing values ('mean', 'median', 'drop', 'forward_fill')
            
        Returns:
            DataFrame with missing values handled
        """
        initial_nulls = df.isnull().sum().sum()
        
        if initial_nulls == 0:
            logger.info("No missing values to handle")
            return df
        
        df_processed = df.copy()
        
        if strategy == "mean":
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(
                df_processed[numeric_cols].mean()
            )
        elif strategy == "median":
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(
                df_processed[numeric_cols].median()
            )
        elif strategy == "drop":
            df_processed = df_processed.dropna()
        elif strategy == "forward_fill":
            df_processed = df_processed.fillna(method='ffill')
        
        logger.info(f"Handled {initial_nulls} missing values using strategy: {strategy}")
        return df_processed
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: DataFrame with potential duplicates
            
        Returns:
            DataFrame without duplicates
        """
        initial_rows = len(df)
        df_processed = df.drop_duplicates().reset_index(drop=True)
        removed_rows = initial_rows - len(df_processed)
        
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} duplicate rows")
        
        return df_processed
    
    def encode_categorical(self, df: pd.DataFrame, categorical_cols: list, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: DataFrame with categorical columns
            categorical_cols: List of categorical column names
            fit: Whether to fit new encoders or use existing
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_processed = df.copy()
        
        for col in categorical_cols:
            if col in df_processed.columns:
                if fit:
                    self.encoders[col] = LabelEncoder()
                    df_processed[col] = self.encoders[col].fit_transform(df_processed[col].astype(str))
                    logger.info(f"Fitted LabelEncoder for column: {col}")
                else:
                    df_processed[col] = self.encoders[col].transform(df_processed[col].astype(str))
                    logger.info(f"Applied existing LabelEncoder for column: {col}")
        
        return df_processed
    
    def scale_features(self, df: pd.DataFrame, numeric_cols: list, fit: bool = True) -> pd.DataFrame:
        """
        Scale numeric features using StandardScaler.
        
        Args:
            df: DataFrame with numeric columns
            numeric_cols: List of numeric column names
            fit: Whether to fit new scaler or use existing
            
        Returns:
            DataFrame with scaled features
        """
        df_processed = df.copy()
        
        if fit:
            scaled_data = self.scaler.fit_transform(df_processed[numeric_cols])
            logger.info("Fitted StandardScaler")
        else:
            scaled_data = self.scaler.transform(df_processed[numeric_cols])
            logger.info("Applied existing StandardScaler")
        
        df_processed[numeric_cols] = scaled_data
        return df_processed
    
    def process_pipeline(self, df: pd.DataFrame, 
                        numeric_cols: list = None,
                        categorical_cols: list = None,
                        target_col: str = None) -> pd.DataFrame:
        """
        Execute complete processing pipeline.
        
        Args:
            df: Raw DataFrame
            numeric_cols: List of numeric column names
            categorical_cols: List of categorical column names
            target_col: Target column name (will not be scaled)
            
        Returns:
            Processed DataFrame
        """
        logger.info("Starting data processing pipeline...")
        
        # Handle missing values
        df_processed = self.handle_missing_values(df, strategy="mean")
        
        # Remove duplicates
        df_processed = self.remove_duplicates(df_processed)
        
        # Encode categorical variables
        if categorical_cols:
            df_processed = self.encode_categorical(df_processed, categorical_cols, fit=True)
        
        # Scale numeric features (excluding target)
        if numeric_cols:
            scale_cols = [col for col in numeric_cols if col != target_col]
            if scale_cols:
                df_processed = self.scale_features(df_processed, scale_cols, fit=True)
        
        logger.info("Data processing pipeline completed")
        return df_processed
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Save processed data to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to: {filepath}")
        return filepath
