# Architecture & Design

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Applications                      │
│                     (Web, Mobile, etc.)                      │
└────────────────┬────────────────────────────────────────────┘
                 │ HTTP/REST
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│                   (API Layer - Port 8000)                    │
├─────────────────────────────────────────────────────────────┤
│ Routes:                                                       │
│ • GET  /              - API Info                             │
│ • GET  /health        - Health Check                         │
│ • POST /predict       - Single Prediction                    │
│ • POST /predict-batch - Batch Predictions                    │
│ • GET  /docs          - Interactive API Docs                 │
└────────────────┬────────────────────────────────────────────┘
                 │
    ┌────────────┴─────────────┐
    │                          │
┌───▼──────────────┐  ┌────────▼──────────┐
│  Model Loading   │  │  Request Handler  │
│  (at startup)    │  │  (prediction)     │
└─────────────────┘  └──────────┬─────────┘
                                 │
                         ┌───────▼────────┐
                         │  ML Pipeline   │
                         │   Module       │
                         └───────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
      ┌───────▼────────┐ ┌───────▼────────┐ ┌─────▼──────────┐
      │ Data Processor │ │ Model Trainer  │ │ Model          │
      │ • Validation   │ │ • Training     │ │ Evaluator      │
      │ • Cleaning     │ │ • Selection    │ │ • Metrics      │
      │ • Scaling      │ │ • Serialization│ │ • Reports      │
      └────────────────┘ └────────────────┘ └────────────────┘
              │
      ┌───────▼──────────┐
      │  Raw Data Files  │
      │  • data/raw/     │
      │  • CSV format    │
      └──────────────────┘
```

## Data Flow

### Training Pipeline

```
1. Data Ingestion
   ↓
   └─→ Fetch from URL/API or load local CSV
   
2. Data Validation
   ↓
   └─→ Check missing values, duplicates, outliers
   
3. Data Processing
   ↓
   ├─→ Handle missing values (mean/median/forward-fill)
   ├─→ Remove duplicates
   ├─→ Encode categorical variables
   └─→ Scale numeric features
   
4. Train-Test Split
   ↓
   └─→ 80% training, 20% testing (stratified)
   
5. Model Training
   ↓
   ├─→ Logistic Regression
   ├─→ Random Forest
   └─→ XGBoost
   
6. Model Selection
   ↓
   └─→ Select best based on CV score
   
7. Evaluation
   ↓
   ├─→ Accuracy, Precision, Recall, F1-Score
   ├─→ Confusion Matrix
   ├─→ ROC-AUC
   └─→ Feature Importance
   
8. Model Persistence
   ↓
   └─→ Save to models/ directory (joblib format)
```

### Prediction Pipeline

```
Client Request
    ↓
API Endpoint (/predict)
    ↓
Validate Input (Pydantic)
    ↓
Load Model from Memory
    ↓
Preprocess Features
    ↓
Make Prediction
    ↓
Format Response
    ↓
Return JSON Response
    ↓
Log Prediction (JSON format)
```

## Module Dependencies

```
src/
├── api/
│   └── app.py
│       ├── imports: FastAPI, Pydantic
│       ├── imports: src.logger
│       └── imports: src.models.trainer
│
├── data/
│   ├── fetcher.py
│   │   ├── imports: requests, pandas
│   │   └── imports: src.logger
│   │
│   ├── validator.py
│   │   ├── imports: pandas, numpy
│   │   └── imports: src.logger
│   │
│   └── processor.py
│       ├── imports: sklearn.preprocessing
│       ├── imports: pandas, numpy
│       └── imports: src.logger
│
├── models/
│   ├── trainer.py
│   │   ├── imports: sklearn, xgboost
│   │   ├── imports: joblib
│   │   └── imports: src.logger
│   │
│   └── evaluator.py
│       ├── imports: sklearn.metrics
│       ├── imports: pandas
│       └── imports: src.logger
│
└── logger.py
    └── imports: logging, python-json-logger
```

## Design Patterns Used

### 1. Factory Pattern
Model trainer creates different model types:
```python
trainer.train_logistic_regression()
trainer.train_random_forest()
trainer.train_xgboost()
```

### 2. Strategy Pattern
Data processor applies different strategies:
```python
processor.handle_missing_values(strategy='mean')
processor.handle_missing_values(strategy='median')
```

### 3. Singleton Pattern
Logger instance:
```python
logger = setup_logger(__name__)  # Created once, reused
```

### 4. Repository Pattern
Model storage:
```python
trainer.save_model(model, "iris_classifier", "1.0.0")
trainer.load_model(filepath)
```

## Configuration Management

```
config/
└── config.py
    ├── Paths (data, models, logs)
    ├── Model hyperparameters
    ├── API settings
    ├── Logging configuration
    └── Data sources
```

## Error Handling Strategy

### Data Layer
- File not found → log warning, raise exception
- Invalid data type → validation error
- Missing values → handle per strategy

### Model Layer
- Training failure → log error, raise exception
- Model not found → return None or raise exception
- Prediction error → log and return error response

### API Layer
- Invalid input → 422 validation error
- Model not loaded → 503 service unavailable
- Prediction error → 500 internal server error

## Testing Strategy

```
Unit Tests:
├── Test data fetching
├── Test data validation
├── Test data processing
├── Test model training
├── Test model evaluation
└── Test API endpoints

Coverage Goal: 80%+

Test Execution:
pytest tests/ -v --cov=src --cov-report=html
```

## Logging Strategy

### Log Levels
- DEBUG: Detailed execution information
- INFO: Major steps in pipeline
- WARNING: Data quality issues, model warnings
- ERROR: Exceptions and failures

### Log Destinations
- Console: Human-readable format (development)
- Files: JSON format (production, aggregation)

### Log Files
```
logs/
├── __main__.log              # Main pipeline
├── src.data.fetcher.log     # Data fetching
├── src.data.validator.log   # Data validation
├── src.data.processor.log   # Data processing
├── src.models.trainer.log   # Model training
├── src.models.evaluator.log # Model evaluation
└── src.api.app.log          # API requests
```

## Performance Considerations

### Model Training
- Cross-validation: 5-fold (balance accuracy vs speed)
- Parallel processing: `n_jobs=-1` for tree-based models
- Feature scaling: StandardScaler for efficiency

### API Serving
- Model loaded in memory (on startup)
- Batch prediction endpoint for efficiency
- Connection pooling for requests
- Caching for repeated predictions

### Data Processing
- Streaming for large files
- Vectorized operations with NumPy/Pandas
- In-memory caching where possible

## Scalability

### Horizontal Scaling
- Multiple API instances behind load balancer
- Shared model storage (S3, NFS)
- Stateless API design

### Vertical Scaling
- Increase server resources
- GPU acceleration for inference
- Model quantization for smaller models

### Data Scaling
- Batch processing for large datasets
- Incremental model updates
- Feature sampling for high-dimensional data

## Security Considerations

### Input Validation
- Pydantic models validate all inputs
- Type checking on parameters
- Range validation for features

### Data Privacy
- No sensitive data logging
- Environment variables for secrets
- Secure model storage

### API Security
- Error messages don't expose internals
- Logging of all predictions
- Rate limiting capability (not implemented)

## Future Enhancements

1. **Model Versioning**
   - Model registry
   - A/B testing support
   - Automatic rollback

2. **Feature Store**
   - Centralized feature management
   - Feature versioning
   - Online/offline consistency

3. **Model Monitoring**
   - Prediction monitoring
   - Data drift detection
   - Performance degradation alerts

4. **AutoML Integration**
   - Hyperparameter tuning
   - Architecture search
   - Ensemble methods

5. **Advanced Deployment**
   - Model serving optimization
   - Distributed inference
   - Edge deployment
