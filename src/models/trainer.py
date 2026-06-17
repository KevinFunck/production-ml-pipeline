"""
Model training and selection.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from src.logger import setup_logger
from config.config import MODELS_DIR, RANDOM_STATE, TRAIN_TEST_SPLIT

logger = setup_logger(__name__)


class ModelTrainer:
    """Train and manage ML models."""
    
    def __init__(self, models_dir: Path = MODELS_DIR):
        """
        Initialize ModelTrainer.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        logger.info(f"ModelTrainer initialized with models_dir: {models_dir}")
    
    def train_test_split_data(self, X: pd.DataFrame, y: pd.Series, 
                             test_size: float = TRAIN_TEST_SPLIT) -> Tuple:
        """
        Split data into training and testing sets.
        
        Args:
            X: Features
            y: Target variable
            test_size: Fraction of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data with test_size={test_size}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=RANDOM_STATE,
            stratify=y
        )
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training Logistic Regression...")
        
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        result = {
            "model": model,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "training_score": model.score(X_train, y_train)
        }
        
        logger.info(f"Logistic Regression - CV Score: {cv_scores.mean():.4f}")
        self.models["logistic_regression"] = result
        return result
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        result = {
            "model": model,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "training_score": model.score(X_train, y_train),
            "feature_importance": model.feature_importances_
        }
        
        logger.info(f"Random Forest - CV Score: {cv_scores.mean():.4f}")
        self.models["random_forest"] = result
        return result
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training XGBoost...")
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            verbosity=0,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        result = {
            "model": model,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "training_score": model.score(X_train, y_train),
            "feature_importance": model.feature_importances_
        }
        
        logger.info(f"XGBoost - CV Score: {cv_scores.mean():.4f}")
        self.models["xgboost"] = result
        return result
    
    def select_best_model(self) -> Tuple[str, Any]:
        """
        Select the best model based on cross-validation score.
        
        Returns:
            Tuple of (best_model_name, best_model_result)
        """
        if not self.models:
            logger.error("No trained models available")
            raise ValueError("No trained models available")
        
        best_name = max(self.models.keys(), 
                       key=lambda k: self.models[k]["cv_mean"])
        best_result = self.models[best_name]
        
        self.best_model_name = best_name
        self.best_model = best_result["model"]
        
        logger.info(f"Best model selected: {best_name} with CV score: {best_result['cv_mean']:.4f}")
        return best_name, best_result
    
    def save_model(self, model: Any, model_name: str, version: str = "1.0.0") -> Path:
        """
        Save model to disk.
        
        Args:
            model: Trained model to save
            model_name: Name of the model
            version: Model version
            
        Returns:
            Path to saved model
        """
        filepath = self.models_dir / f"{model_name}_v{version}.joblib"
        joblib.dump(model, filepath)
        logger.info(f"Model saved to: {filepath}")
        return filepath
    
    def load_model(self, filepath: Path) -> Any:
        """
        Load model from disk.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded model
        """
        model = joblib.load(filepath)
        logger.info(f"Model loaded from: {filepath}")
        return model
