"""
FastAPI application for ML model serving.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import API_TITLE, API_VERSION
from src.logger import setup_logger
from src.models.trainer import ModelTrainer

logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Production ML Pipeline API for Iris Classification"
)

# Load model on startup
model_trainer = ModelTrainer()
loaded_model = None


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    global loaded_model
    try:
        model_path = Path(__file__).parent / "models" / "iris_classifier_v1.0.0.joblib"
        if model_path.exists():
            loaded_model = model_trainer.load_model(model_path)
            logger.info("Model loaded successfully on startup")
        else:
            logger.warning(f"Model file not found at {model_path}. Model needs to be trained first.")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")


# ============================================
# Request/Response Models
# ============================================

class IrisFeatures(BaseModel):
    """Iris dataset features."""
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }


class PredictionResponse(BaseModel):
    """Model prediction response."""
    prediction: str
    probability: dict
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


# ============================================
# API Endpoints
# ============================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": loaded_model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    """
    Make prediction on iris features.
    
    Args:
        features: Iris flower features
        
    Returns:
        Prediction and probabilities
        
    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if loaded_model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Prepare features
        X = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        # Make prediction
        prediction = loaded_model.predict(X)[0]
        probabilities = loaded_model.predict_proba(X)[0]
        
        # Map predictions to species names
        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        predicted_species = species_map.get(prediction, "unknown")
        
        # Create probability dict
        probability_dict = {
            species_map[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }
        
        logger.info(f"Prediction made: {predicted_species} (confidence: {max(probabilities):.4f})")
        
        return {
            "prediction": predicted_species,
            "probability": probability_dict,
            "message": f"Predicted species: {predicted_species}"
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-batch")
async def predict_batch(samples: list):
    """
    Make batch predictions.
    
    Args:
        samples: List of feature dictionaries
        
    Returns:
        List of predictions
    """
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        
        for sample in samples:
            X = np.array([[
                sample["sepal_length"],
                sample["sepal_width"],
                sample["petal_length"],
                sample["petal_width"]
            ]])
            
            pred = loaded_model.predict(X)[0]
            proba = loaded_model.predict_proba(X)[0]
            
            species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
            
            predictions.append({
                "prediction": species_map[pred],
                "probability": {
                    species_map[i]: float(proba[i])
                    for i in range(len(proba))
                }
            })
        
        logger.info(f"Batch prediction completed for {len(samples)} samples")
        return predictions
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
