"""
Quick inference script for testing predictions on new data.
"""
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.logger import setup_logger
from src.models.trainer import ModelTrainer
from src.data.processor import DataProcessor
import numpy as np

logger = setup_logger(__name__)


def predict_iris_species(sepal_length: float, sepal_width: float, 
                         petal_length: float, petal_width: float):
    """
    Predict iris species for given measurements.
    
    Args:
        sepal_length: Sepal length in cm
        sepal_width: Sepal width in cm
        petal_length: Petal length in cm
        petal_width: Petal width in cm
    """
    
    try:
        logger.info("=" * 60)
        logger.info("IRIS SPECIES PREDICTION")
        logger.info("=" * 60)
        
        # Load model
        trainer = ModelTrainer()
        model_path = Path(__file__).parent / "models" / "iris_classifier_v1.0.0.joblib"
        
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            logger.info("Please run: python train_pipeline.py")
            return None
        
        model = trainer.load_model(model_path)
        
        # Create feature array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Map to species names
        species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        predicted_species = species_map[prediction]
        
        # Get confidence
        confidence = max(probabilities) * 100
        
        # Log results
        logger.info(f"\nInput Measurements:")
        logger.info(f"  Sepal Length: {sepal_length} cm")
        logger.info(f"  Sepal Width: {sepal_width} cm")
        logger.info(f"  Petal Length: {petal_length} cm")
        logger.info(f"  Petal Width: {petal_width} cm")
        
        logger.info(f"\nPrediction: {predicted_species}")
        logger.info(f"Confidence: {confidence:.2f}%")
        
        logger.info(f"\nProbabilities:")
        for i, species in species_map.items():
            prob = probabilities[i] * 100
            logger.info(f"  {species}: {prob:.2f}%")
        
        logger.info("=" * 60)
        
        return {
            "species": predicted_species,
            "confidence": confidence,
            "probabilities": {
                species_map[i]: float(probabilities[i])
                for i in range(len(probabilities))
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return None


if __name__ == "__main__":
    # Example predictions
    print("\n" + "=" * 60)
    print("Example 1: Small Setosa-like flower")
    print("=" * 60)
    predict_iris_species(5.1, 3.5, 1.4, 0.2)
    
    print("\n" + "=" * 60)
    print("Example 2: Medium Versicolor-like flower")
    print("=" * 60)
    predict_iris_species(5.9, 3.0, 4.2, 1.5)
    
    print("\n" + "=" * 60)
    print("Example 3: Large Virginica-like flower")
    print("=" * 60)
    predict_iris_species(6.5, 3.0, 5.5, 1.8)
