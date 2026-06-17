"""
Unit tests for model modules.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator


@pytest.fixture
def sample_data():
    """Create sample classification data."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_classes=3,
        n_informative=3,
        random_state=42
    )
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series


class TestModelTrainer:
    """Test ModelTrainer class."""
    
    def test_trainer_initialization(self, tmp_path):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(tmp_path)
        assert trainer.models_dir == tmp_path
        assert len(trainer.models) == 0
    
    def test_train_test_split(self, sample_data):
        """Test train-test split."""
        X, y = sample_data
        trainer = ModelTrainer()
        
        X_train, X_test, y_train, y_test = trainer.train_test_split_data(X, y)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
    
    def test_train_logistic_regression(self, sample_data):
        """Test Logistic Regression training."""
        X, y = sample_data
        trainer = ModelTrainer()
        
        X_train, X_test, y_train, y_test = trainer.train_test_split_data(X, y)
        result = trainer.train_logistic_regression(X_train, y_train)
        
        assert 'model' in result
        assert 'cv_mean' in result
        assert result['cv_mean'] > 0
    
    def test_train_random_forest(self, sample_data):
        """Test Random Forest training."""
        X, y = sample_data
        trainer = ModelTrainer()
        
        X_train, X_test, y_train, y_test = trainer.train_test_split_data(X, y)
        result = trainer.train_random_forest(X_train, y_train)
        
        assert 'model' in result
        assert 'feature_importance' in result
    
    def test_select_best_model(self, sample_data):
        """Test best model selection."""
        X, y = sample_data
        trainer = ModelTrainer()
        
        X_train, X_test, y_train, y_test = trainer.train_test_split_data(X, y)
        trainer.train_logistic_regression(X_train, y_train)
        trainer.train_random_forest(X_train, y_train)
        
        best_name, best_result = trainer.select_best_model()
        
        assert best_name in trainer.models.keys()
        assert trainer.best_model is not None


class TestModelEvaluator:
    """Test ModelEvaluator class."""
    
    def test_evaluate_classification(self):
        """Test classification evaluation."""
        y_true = pd.Series([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        metrics = ModelEvaluator.evaluate_classification(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'confusion_matrix' in metrics
    
    def test_feature_importance_report(self, sample_data):
        """Test feature importance reporting."""
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = sample_data
        
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        model.fit(X, y)
        
        feature_names = X.columns.tolist()
        importance_df = ModelEvaluator.feature_importance_report(model, feature_names)
        
        assert importance_df is not None
        assert len(importance_df) == len(feature_names)
