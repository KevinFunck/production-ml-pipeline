"""
Main ML Pipeline orchestrator.
Execute complete pipeline: fetch -> process -> train -> evaluate -> save
"""
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.logger import setup_logger
from src.data.fetcher import DataFetcher
from src.data.validator import DataValidator
from src.data.processor import DataProcessor
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator

logger = setup_logger(__name__)


def main():
    """Execute complete ML pipeline."""
    
    logger.info("=" * 60)
    logger.info("STARTING PRODUCTION ML PIPELINE")
    logger.info("=" * 60)
    
    try:
        # ============================================
        # STAGE 1: DATA INGESTION
        # ============================================
        logger.info("\n[STAGE 1] DATA INGESTION")
        logger.info("-" * 60)
        
        fetcher = DataFetcher()
        df_raw = fetcher.fetch_iris_dataset()
        logger.info(f"Raw data shape: {df_raw.shape}")
        logger.info(f"Columns: {df_raw.columns.tolist()}")
        
        # ============================================
        # STAGE 2: DATA VALIDATION
        # ============================================
        logger.info("\n[STAGE 2] DATA VALIDATION")
        logger.info("-" * 60)
        
        numeric_cols = df_raw.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'species']  # Remove target
        
        validation_report = DataValidator.validate_all(df_raw, numeric_cols=numeric_cols)
        
        # ============================================
        # STAGE 3: DATA PROCESSING
        # ============================================
        logger.info("\n[STAGE 3] DATA PROCESSING")
        logger.info("-" * 60)
        
        processor = DataProcessor()
        
        # Remove duplicates first
        df_raw = processor.remove_duplicates(df_raw)
        
        # Prepare data for processing
        X = df_raw.drop('species', axis=1)
        y = df_raw['species']
        
        # Get feature names
        feature_names = X.columns.tolist()
        
        # Encode target variable
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Process features
        X_processed = processor.process_pipeline(
            X,
            numeric_cols=feature_names,
            categorical_cols=[],
            target_col=None
        )
        
        # Create processed dataset
        df_processed = X_processed.copy()
        df_processed['species'] = y.values
        
        processor.save_processed_data(df_processed, "iris_processed.csv")
        logger.info(f"Processed data shape: {df_processed.shape}")
        
        # ============================================
        # STAGE 4: MODEL TRAINING
        # ============================================
        logger.info("\n[STAGE 4] MODEL TRAINING")
        logger.info("-" * 60)
        
        trainer = ModelTrainer()
        
        # Split data
        X_train, X_test, y_train, y_test = trainer.train_test_split_data(
            X_processed, y_encoded
        )
        
        # Train multiple models
        logger.info("\nTraining multiple models...")
        trainer.train_logistic_regression(X_train, y_train)
        trainer.train_random_forest(X_train, y_train)
        trainer.train_xgboost(X_train, y_train)
        
        # ============================================
        # STAGE 5: MODEL SELECTION
        # ============================================
        logger.info("\n[STAGE 5] MODEL SELECTION")
        logger.info("-" * 60)
        
        best_model_name, best_result = trainer.select_best_model()
        best_model = best_result["model"]
        
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"CV Score: {best_result['cv_mean']:.4f} (+/- {best_result['cv_std']:.4f})")
        
        # ============================================
        # STAGE 6: MODEL EVALUATION
        # ============================================
        logger.info("\n[STAGE 6] MODEL EVALUATION")
        logger.info("-" * 60)
        
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
        
        metrics = ModelEvaluator.evaluate_classification(y_test, y_pred, y_pred_proba)
        ModelEvaluator.log_metrics(metrics)
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = ModelEvaluator.feature_importance_report(
                best_model, 
                feature_names
            )
        
        # Model comparison
        comparison = ModelEvaluator.compare_models(trainer.models)
        
        # ============================================
        # STAGE 7: MODEL PERSISTENCE
        # ============================================
        logger.info("\n[STAGE 7] MODEL PERSISTENCE")
        logger.info("-" * 60)
        
        model_path = trainer.save_model(best_model, "iris_classifier", version="1.0.0")
        logger.info(f"Model saved to: {model_path}")
        
        # ============================================
        # PIPELINE COMPLETE
        # ============================================
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"\nBest Model: {best_model_name}")
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test F1-Score: {metrics['f1']:.4f}")
        logger.info(f"Model saved to: {model_path}")
        logger.info("=" * 60)
        
        return {
            "success": True,
            "best_model": best_model,
            "best_model_name": best_model_name,
            "metrics": metrics,
            "model_path": str(model_path)
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result.get("success") else 1)
