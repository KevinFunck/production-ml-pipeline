"""
Comprehensive data pipeline with exploration and analysis.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.logger import setup_logger
from src.data.fetcher import DataFetcher
from src.data.validator import DataValidator
import pandas as pd

logger = setup_logger(__name__)


def explore_data():
    """
    Explore and analyze the dataset.
    """
    
    logger.info("=" * 60)
    logger.info("DATA EXPLORATION & ANALYSIS")
    logger.info("=" * 60)
    
    try:
        # Fetch data
        logger.info("\n[STEP 1] Fetching Data")
        logger.info("-" * 60)
        
        fetcher = DataFetcher()
        df = fetcher.fetch_iris_dataset()
        
        # Basic info
        logger.info(f"\nDataset Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Display first few rows
        logger.info("\nFirst 5 rows:")
        logger.info(f"\n{df.head()}")
        
        # ============================================
        # DATA SUMMARY STATISTICS
        # ============================================
        logger.info("\n[STEP 2] Summary Statistics")
        logger.info("-" * 60)
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        logger.info(f"\n{df[numeric_cols].describe()}")
        
        # ============================================
        # DATA VALIDATION
        # ============================================
        logger.info("\n[STEP 3] Data Validation")
        logger.info("-" * 60)
        
        validation_report = DataValidator.validate_all(df, numeric_cols=numeric_cols)
        
        logger.info(f"\nValidation Report:")
        logger.info(f"  Shape: {validation_report['shape']}")
        logger.info(f"  Duplicates: {validation_report['duplicates']}")
        logger.info(f"  Outliers: {len(validation_report['outliers'])} columns")
        
        # ============================================
        # VALUE DISTRIBUTION
        # ============================================
        logger.info("\n[STEP 4] Value Distribution")
        logger.info("-" * 60)
        
        # Target distribution
        logger.info("\nTarget Distribution (species):")
        species_counts = df['species'].value_counts()
        for species, count in species_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"  {species}: {count} ({percentage:.1f}%)")
        
        # ============================================
        # CORRELATION ANALYSIS
        # ============================================
        logger.info("\n[STEP 5] Feature Correlation")
        logger.info("-" * 60)
        
        correlation_matrix = df[numeric_cols].corr()
        logger.info(f"\nCorrelation Matrix:")
        logger.info(f"\n{correlation_matrix}")
        
        # ============================================
        # SAVE ANALYSIS
        # ============================================
        analysis_path = Path(__file__).parent / "data_analysis.csv"
        df[numeric_cols].describe().to_csv(analysis_path)
        logger.info(f"\nAnalysis saved to: {analysis_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("DATA EXPLORATION COMPLETED!")
        logger.info("=" * 60)
        
        return {"success": True}
        
    except Exception as e:
        logger.error(f"Exploration failed: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    result = explore_data()
    sys.exit(0 if result.get("success") else 1)
