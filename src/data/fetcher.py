"""
Data fetching and ingestion from various sources.
"""
import requests
import pandas as pd
from pathlib import Path
from typing import Optional
from src.logger import setup_logger
from config.config import RAW_DATA_DIR, DATA_SOURCES

logger = setup_logger(__name__)


class DataFetcher:
    """Handle data fetching from various sources."""
    
    def __init__(self, output_dir: Path = RAW_DATA_DIR):
        """
        Initialize DataFetcher.
        
        Args:
            output_dir: Directory to save fetched data
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataFetcher initialized with output_dir: {output_dir}")
    
    def fetch_from_url(self, url: str, filename: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data from URL (CSV format).
        
        Args:
            url: URL to fetch data from
            filename: Optional filename to save as
            
        Returns:
            Loaded DataFrame
            
        Raises:
            Exception: If fetching fails
        """
        try:
            logger.info(f"Fetching data from URL: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Read CSV from response
            df = pd.read_csv(pd.io.common.StringIO(response.text))
            logger.info(f"Successfully fetched data with shape: {df.shape}")
            
            # Save to raw data directory if filename provided
            if filename:
                filepath = self.output_dir / filename
                df.to_csv(filepath, index=False)
                logger.info(f"Data saved to: {filepath}")
            
            return df
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch data from {url}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error processing fetched data: {str(e)}")
            raise
    
    def fetch_iris_dataset(self) -> pd.DataFrame:
        """
        Fetch Iris dataset.
        
        Returns:
            Iris dataset DataFrame
        """
        logger.info("Fetching Iris dataset...")
        return self.fetch_from_url(
            DATA_SOURCES["iris"],
            filename="iris_raw.csv"
        )
    
    def load_local_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load CSV file from local path.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Loaded DataFrame
        """
        try:
            logger.info(f"Loading local CSV: {filepath}")
            df = pd.read_csv(filepath)
            logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise
