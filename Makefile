.PHONY: help install test train run-api docker-build docker-run clean lint

help:
	@echo "Production ML Pipeline - Available Commands"
	@echo "============================================"
	@echo "make install       - Install dependencies"
	@echo "make lint          - Run code linting"
	@echo "make test          - Run tests"
	@echo "make train         - Run training pipeline"
	@echo "make run-api       - Run FastAPI server"
	@echo "make docker-build  - Build Docker image"
	@echo "make docker-run    - Run Docker container"
	@echo "make clean         - Clean up generated files"

install:
	pip install -r requirements.txt
	@echo "Dependencies installed!"

lint:
	flake8 src/ tests/ --max-line-length=100
	pylint src/ tests/ --disable=all --enable=E,F
	@echo "Linting complete!"

test:
	pytest tests/ -v --cov=src --cov-report=html
	@echo "Tests complete! Coverage report in htmlcov/index.html"

train:
	python train_pipeline.py

run-api:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t production-ml-pipeline:latest .
	@echo "Docker image built!"

docker-run:
	docker run -p 8000:8000 -v $(PWD)/data:/app/data -v $(PWD)/models:/app/models production-ml-pipeline:latest

docker-compose-up:
	docker-compose up -d
	@echo "Docker Compose started!"

docker-compose-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	@echo "Cleanup complete!"
