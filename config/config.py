"""
Configuration settings for the ML Pipeline project.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# API Configuration
API_TITLE = "Production ML Pipeline API"
API_VERSION = "1.0.0"
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Model Configuration
MODEL_NAME = "production_model"
MODEL_VERSION = "1.0.0"
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Data Sources
DATA_SOURCES = {
    "iris": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
}
