"""
Unit tests for data modules.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.fetcher import DataFetcher
from src.data.validator import DataValidator
from src.data.processor import DataProcessor


class TestDataFetcher:
    """Test DataFetcher class."""
    
    def test_fetcher_initialization(self, tmp_path):
        """Test DataFetcher initialization."""
        fetcher = DataFetcher(tmp_path)
        assert fetcher.output_dir == tmp_path
    
    def test_load_local_csv(self, tmp_path):
        """Test loading local CSV file."""
        # Create test data
        test_file = tmp_path / "test.csv"
        df_original = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        df_original.to_csv(test_file, index=False)
        
        # Load data
        fetcher = DataFetcher(tmp_path)
        df_loaded = fetcher.load_local_csv(str(test_file))
        
        assert df_loaded.shape == (3, 2)
        assert list(df_loaded.columns) == ['A', 'B']


class TestDataValidator:
    """Test DataValidator class."""
    
    def test_check_missing_values(self):
        """Test missing values detection."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan],
            'B': [4, 5, 6]
        })
        
        result = DataValidator.check_missing_values(df)
        assert 'A' in result['counts'].index
        assert result['counts']['A'] == 1
    
    def test_check_duplicates(self):
        """Test duplicate detection."""
        df = pd.DataFrame({
            'A': [1, 1, 2],
            'B': [4, 4, 6]
        })
        
        duplicates = DataValidator.check_duplicates(df)
        assert duplicates == 1
    
    def test_check_data_types(self):
        """Test data type checking."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        
        data_types = DataValidator.check_data_types(df)
        assert 'A' in data_types
        assert 'B' in data_types


class TestDataProcessor:
    """Test DataProcessor class."""
    
    def test_processor_initialization(self, tmp_path):
        """Test DataProcessor initialization."""
        processor = DataProcessor(tmp_path)
        assert processor.output_dir == tmp_path
    
    def test_handle_missing_values(self):
        """Test handling missing values."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan],
            'B': [4, np.nan, 6]
        })
        
        processor = DataProcessor()
        df_processed = processor.handle_missing_values(df, strategy='mean')
        
        assert df_processed.isnull().sum().sum() == 0
    
    def test_remove_duplicates(self):
        """Test removing duplicates."""
        df = pd.DataFrame({
            'A': [1, 1, 2],
            'B': [4, 4, 6]
        })
        
        processor = DataProcessor()
        df_processed = processor.remove_duplicates(df)
        
        assert len(df_processed) == 2
    
    def test_scale_features(self):
        """Test feature scaling."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        
        processor = DataProcessor()
        df_scaled = processor.scale_features(df, ['A', 'B'])
        
        # Check if features are scaled
        assert np.isclose(df_scaled['A'].mean(), 0, atol=1e-10)
        assert np.isclose(df_scaled['B'].mean(), 0, atol=1e-10)
