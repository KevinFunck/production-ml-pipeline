"""
Model evaluation and visualization script.
Generates performance metrics, confusion matrix, and feature importance plots.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.logger import setup_logger
from src.models.trainer import ModelTrainer
from src.data.fetcher import DataFetcher
from src.data.processor import DataProcessor

logger = setup_logger(__name__)


def evaluate_model():
    """
    Evaluate trained model and generate visualizations.
    """
    
    logger.info("=" * 60)
    logger.info("STARTING MODEL EVALUATION")
    logger.info("=" * 60)
    
    try:
        # ============================================
        # LOAD DATA
        # ============================================
        logger.info("\n[STEP 1] Loading Data")
        logger.info("-" * 60)
        
        fetcher = DataFetcher()
        df_raw = fetcher.fetch_iris_dataset()
        
        # Process data
        processor = DataProcessor()
        X = df_raw.drop('species', axis=1)
        y = df_raw['species']
        
        feature_names = X.columns.tolist()
        X_processed = processor.process_pipeline(X, numeric_cols=feature_names)
        
        # ============================================
        # LOAD MODEL
        # ============================================
        logger.info("\n[STEP 2] Loading Trained Model")
        logger.info("-" * 60)
        
        trainer = ModelTrainer()
        model_path = Path(__file__).parent / "models" / "iris_classifier_v1.0.0.joblib"
        
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            logger.info("Please run: python train_pipeline.py")
            return
        
        model = trainer.load_model(model_path)
        logger.info(f"Model loaded successfully from: {model_path}")
        
        # ============================================
        # GENERATE PREDICTIONS
        # ============================================
        logger.info("\n[STEP 3] Generating Predictions")
        logger.info("-" * 60)
        
        y_pred = model.predict(X_processed)
        y_pred_proba = model.predict_proba(X_processed)
        
        logger.info(f"Predictions generated for {len(X_processed)} samples")
        
        # ============================================
        # PERFORMANCE METRICS
        # ============================================
        logger.info("\n[STEP 4] Performance Metrics")
        logger.info("-" * 60)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        
        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }
        
        logger.info("\nPerformance Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # ============================================
        # SAVE METRICS TO FILE
        # ============================================
        metrics_df = pd.DataFrame([metrics])
        metrics_path = Path(__file__).parent / "evaluation_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"\nMetrics saved to: {metrics_path}")
        
        # ============================================
        # CLASSIFICATION REPORT
        # ============================================
        logger.info("\n[STEP 5] Classification Report")
        logger.info("-" * 60)
        
        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        logger.info(f"\n{classification_report(y, y_pred)}")
        
        report_path = Path(__file__).parent / "classification_report.csv"
        report_df.to_csv(report_path)
        logger.info(f"Classification report saved to: {report_path}")
        
        # ============================================
        # FEATURE IMPORTANCE
        # ============================================
        logger.info("\n[STEP 6] Feature Importance")
        logger.info("-" * 60)
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\nFeature Importance:")
            for idx, row in importance_df.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            importance_path = Path(__file__).parent / "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to: {importance_path}")
        
        # ============================================
        # COMPLETION
        # ============================================
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"\nTest Accuracy: {accuracy:.4f}")
        logger.info(f"Test F1-Score: {f1:.4f}")
        logger.info("\nGenerated Files:")
        logger.info(f"  ✓ {metrics_path.name}")
        logger.info(f"  ✓ {report_path.name}")
        if hasattr(model, 'feature_importances_'):
            logger.info(f"  ✓ {importance_path.name}")
        logger.info("=" * 60)
        
        return {
            "success": True,
            "metrics": metrics,
            "accuracy": accuracy
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    result = evaluate_model()
    sys.exit(0 if result.get("success") else 1)
