"""
Model evaluation and performance metrics.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from src.logger import setup_logger

logger = setup_logger(__name__)


class ModelEvaluator:
    """Evaluate model performance."""
    
    @staticmethod
    def evaluate_classification(y_true: pd.Series, y_pred: np.ndarray, 
                               y_pred_proba: np.ndarray = None) -> Dict:
        """
        Evaluate classification model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for ROC-AUC)
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "recall": recall_score(y_true, y_pred, average='weighted'),
            "f1": f1_score(y_true, y_pred, average='weighted'),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred)
        }
        
        # ROC-AUC for binary/multiclass classification
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {str(e)}")
        
        logger.info(f"Evaluation complete - Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    @staticmethod
    def log_metrics(metrics: Dict) -> None:
        """
        Log evaluation metrics.
        
        Args:
            metrics: Dictionary with evaluation metrics
        """
        logger.info("=" * 50)
        logger.info("MODEL EVALUATION METRICS")
        logger.info("=" * 50)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1']:.4f}")
        
        if "roc_auc" in metrics:
            logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        logger.info("\nConfusion Matrix:")
        logger.info(f"\n{metrics['confusion_matrix']}")
        logger.info("\nClassification Report:")
        logger.info(f"\n{metrics['classification_report']}")
        logger.info("=" * 50)
    
    @staticmethod
    def feature_importance_report(model: Any, feature_names: list) -> pd.DataFrame:
        """
        Generate feature importance report.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance sorted
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Feature Importance Report:")
        logger.info(f"\n{importance_df}")
        
        return importance_df
    
    @staticmethod
    def compare_models(models_results: Dict) -> pd.DataFrame:
        """
        Compare multiple model performances.
        
        Args:
            models_results: Dictionary of model results
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name, result in models_results.items():
            comparison_data.append({
                'Model': model_name,
                'CV Mean': result['cv_mean'],
                'CV Std': result['cv_std'],
                'Training Score': result['training_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('CV Mean', ascending=False)
        
        logger.info("Model Comparison:")
        logger.info(f"\n{comparison_df}")
        
        return comparison_df
