# Development Guide

## Setup Development Environment

### Prerequisites
- Python 3.9+
- pip or conda
- Git
- Docker (optional, for containerization)

### Installation

```bash
# Clone the repository
git clone https://github.com/KevinFunck/production-ml-pipeline.git
cd production-ml-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
# or: pip install -r requirements.txt
```

## Running the Pipeline

### 1. Training Pipeline

Execute the complete ML pipeline:

```bash
make train
# or: python train_pipeline.py
```

This will:
1. Fetch the Iris dataset
2. Validate data quality
3. Process and scale features
4. Train multiple models (Logistic Regression, Random Forest, XGBoost)
5. Evaluate and select the best model
6. Save the trained model

### 2. API Server

Run the FastAPI server for serving predictions:

```bash
make run-api
# or: uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

**API Documentation:**
- Interactive docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

**Example Prediction:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{
    \"sepal_length\": 5.1,
    \"sepal_width\": 3.5,
    \"petal_length\": 1.4,
    \"petal_width\": 0.2
  }"
```

## Testing

### Run Unit Tests

```bash
make test
# or: pytest tests/ -v --cov=src
```

Coverage report will be generated in `htmlcov/index.html`

### Run Specific Tests

```bash
# Test data module
pytest tests/test_data.py -v

# Test models module
pytest tests/test_models.py -v
```

## Code Quality

### Linting

```bash
make lint
```

## Docker

### Build Docker Image

```bash
make docker-build
```

### Run with Docker

```bash
make docker-run
```

### Using Docker Compose

```bash
# Start services
make docker-compose-up

# Stop services
make docker-compose-down
```

## Project Structure Deep Dive

### `/config`
Central configuration management
- `config.py`: All configurable parameters

### `/src/data`
Data processing pipeline
- `fetcher.py`: Data ingestion from various sources
- `validator.py`: Data quality checks and validation
- `processor.py`: Feature engineering and preprocessing

### `/src/models`
Model training and evaluation
- `trainer.py`: Model training with multiple algorithms
- `evaluator.py`: Performance metrics and evaluation

### `/src/api`
FastAPI application for model serving
- `app.py`: REST API endpoints

### `/tests`
Unit tests with pytest
- `test_data.py`: Data module tests
- `test_models.py`: Model module tests

## Logging

The pipeline uses structured JSON logging for production readiness:
- Console output: Human-readable format
- File output: JSON format for log aggregation

Logs are saved to `/logs` directory:
```
logs/
├── __main__.log        # Main pipeline logs
├── src.data.fetcher.log
├── src.data.validator.log
├── src.data.processor.log
├── src.models.trainer.log
└── src.models.evaluator.log
```

## Git Workflow

### Commit Convention

Use semantic commit messages:
- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code refactoring
- `docs:` - Documentation
- `test:` - Test additions
- `ci:` - CI/CD configuration

### Example

```bash
git add .
git commit -m "feat: add data validation module"
git push origin main
```

## Best Practices

### 1. Data Handling
- Always validate data before processing
- Check for missing values and outliers
- Log all data quality issues

### 2. Model Training
- Use cross-validation for robust evaluation
- Save trained models with versioning
- Compare multiple algorithms

### 3. API Development
- Use type hints with Pydantic models
- Implement comprehensive error handling
- Include logging for debugging

### 4. Code Quality
- Follow PEP 8 style guide
- Add docstrings to all functions
- Write unit tests for all modules

### 5. Deployment
- Use Docker for consistency
- Implement health checks
- Monitor predictions and metrics

## Troubleshooting

### Model Not Found Error

If you see "Model not found" when running the API:
```bash
# First, run the training pipeline
python train_pipeline.py

# Then start the API
make run-api
```

### Missing Dependencies

```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Tests Failing

```bash
# Check if all dependencies are installed
pip install -r requirements.txt

# Run tests with verbose output
pytest tests/ -v -s
```

## Performance Optimization

### For Training
- Increase cross-validation folds for more robust evaluation
- Use parallel processing with `n_jobs=-1`
- Implement hyperparameter tuning

### For Serving
- Cache model in memory (already done in API)
- Use batch prediction for multiple requests
- Implement request/response compression

## Contributing

1. Create a feature branch: `git checkout -b feat/new-feature`
2. Make your changes
3. Run tests: `make test`
4. Run linting: `make lint`
5. Commit with conventional messages
6. Push to repository
7. Create Pull Request

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Pytest Documentation](https://docs.pytest.org/)

## Support

For issues or questions, please open an issue on GitHub.
