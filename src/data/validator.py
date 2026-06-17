"""
Data validation and quality checks.
"""
import pandas as pd
import numpy as np
from src.logger import setup_logger

logger = setup_logger(__name__)


class DataValidator:
    """Validate data quality and consistency."""
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame) -> dict:
        """
        Check for missing values.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with missing value counts and percentages
        """
        missing_counts = df.isnull().sum()
        missing_percent = (missing_counts / len(df)) * 100
        
        result = {
            "counts": missing_counts[missing_counts > 0],
            "percentages": missing_percent[missing_percent > 0]
        }
        
        if len(result["counts"]) > 0:
            logger.warning(f"Missing values detected:\n{result['counts']}")
        else:
            logger.info("No missing values detected")
        
        return result
    
    @staticmethod
    def check_duplicates(df: pd.DataFrame) -> int:
        """
        Check for duplicate rows.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Number of duplicate rows
        """
        duplicates = df.duplicated().sum()
        
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
        else:
            logger.info("No duplicate rows found")
        
        return duplicates
    
    @staticmethod
    def check_data_types(df: pd.DataFrame) -> dict:
        """
        Validate data types.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with data types
        """
        data_types = df.dtypes.to_dict()
        logger.info(f"Data types: {data_types}")
        return data_types
    
    @staticmethod
    def check_outliers(df: pd.DataFrame, numeric_cols: list) -> dict:
        """
        Detect outliers using IQR method.
        
        Args:
            df: DataFrame to check
            numeric_cols: List of numeric column names
            
        Returns:
            Dictionary with outlier information
        """
        outliers = {}
        
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    outliers[col] = {
                        "count": outlier_count,
                        "percentage": (outlier_count / len(df)) * 100,
                        "bounds": {"lower": lower_bound, "upper": upper_bound}
                    }
                    logger.warning(f"Column '{col}': {outlier_count} outliers detected")
        
        if not outliers:
            logger.info("No outliers detected")
        
        return outliers
    
    @staticmethod
    def validate_all(df: pd.DataFrame, numeric_cols: list = None) -> dict:
        """
        Run all validation checks.
        
        Args:
            df: DataFrame to validate
            numeric_cols: List of numeric column names
            
        Returns:
            Dictionary with all validation results
        """
        logger.info("Starting comprehensive data validation...")
        
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        validation_report = {
            "shape": df.shape,
            "missing_values": DataValidator.check_missing_values(df),
            "duplicates": DataValidator.check_duplicates(df),
            "data_types": DataValidator.check_data_types(df),
            "outliers": DataValidator.check_outliers(df, numeric_cols)
        }
        
        logger.info("Data validation completed")
        return validation_report
