# Production Deployment Guide

## Overview

This guide covers deploying the Production ML Pipeline to a production environment.

## Pre-Deployment Checklist

- [ ] All tests pass locally
- [ ] Code review completed
- [ ] Documentation updated
- [ ] Dependencies up to date
- [ ] Environment variables configured
- [ ] Model trained and validated
- [ ] Monitoring configured

## Deployment Options

### Option 1: Docker Compose (Recommended for Small Deployments)

#### Setup

1. **Prepare the server:**
   ```bash
   # Update system
   sudo apt-get update && apt-get upgrade -y
   
   # Install Docker and Docker Compose
   sudo apt-get install docker.io docker-compose -y
   ```

2. **Clone repository:**
   ```bash
   cd /opt
   sudo git clone https://github.com/KevinFunck/production-ml-pipeline.git
   cd production-ml-pipeline
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with production values
   sudo nano .env
   ```

4. **Build and deploy:**
   ```bash
   sudo docker-compose build
   sudo docker-compose up -d
   ```

5. **Verify:**
   ```bash
   sudo docker-compose logs -f ml-pipeline
   curl http://localhost:8000/health
   ```

### Option 2: Kubernetes Deployment

#### Prerequisites
- Kubernetes cluster (AWS EKS, Google GKE, Azure AKS)
- kubectl configured
- Container registry access

#### Deployment

1. **Build and push Docker image:**
   ```bash
   docker build -t your-registry/ml-pipeline:1.0.0 .
   docker push your-registry/ml-pipeline:1.0.0
   ```

2. **Create Kubernetes resources:**
   
   **deployment.yaml:**
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: ml-pipeline-api
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: ml-pipeline
     template:
       metadata:
         labels:
           app: ml-pipeline
       spec:
         containers:
         - name: api
           image: your-registry/ml-pipeline:1.0.0
           ports:
           - containerPort: 8000
           env:
           - name: LOG_LEVEL
             value: "INFO"
           livenessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 10
             periodSeconds: 5
           resources:
             requests:
               memory: "512Mi"
               cpu: "250m"
             limits:
               memory: "1Gi"
               cpu: "500m"
   
   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: ml-pipeline-service
   spec:
     type: LoadBalancer
     ports:
     - port: 80
       targetPort: 8000
     selector:
       app: ml-pipeline
   ```

3. **Deploy:**
   ```bash
   kubectl apply -f deployment.yaml
   kubectl get pods
   kubectl logs <pod-name>
   ```

### Option 3: AWS ECS

1. **Push to ECR:**
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
   docker build -t ml-pipeline:1.0.0 .
   docker tag ml-pipeline:1.0.0 <account>.dkr.ecr.us-east-1.amazonaws.com/ml-pipeline:1.0.0
   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/ml-pipeline:1.0.0
   ```

2. **Create ECS task definition** and deploy

## Monitoring & Logging

### Health Checks

```bash
# Check API health
curl http://your-domain:8000/health

# Expected response:
# {"status":"healthy","model_loaded":true}
```

### Logging

Logs are available in `/logs` directory:

```bash
# View recent logs
tail -f logs/ml_pipeline.log

# Filter for errors
grep "ERROR" logs/ml_pipeline.log

# JSON logging for log aggregation
tail -f logs/__main__.log | jq '.'
```

### Monitoring with Prometheus

1. **Add Prometheus client:**
   ```bash
   pip install prometheus-client
   ```

2. **Expose metrics at `/metrics` endpoint** (can be added to API)

3. **Configure Prometheus scraping:**
   ```yaml
   global:
     scrape_interval: 15s
   
   scrape_configs:
     - job_name: 'ml-pipeline'
       static_configs:
         - targets: ['localhost:8000']
   ```

## Database & Model Storage

### Model Versioning

Store models with timestamps:
```
models/
├── iris_classifier_v1.0.0.joblib
├── iris_classifier_v1.0.1.joblib
└── iris_classifier_v1.1.0.joblib
```

### Backup Strategy

```bash
# Backup models
tar -czf ml_pipeline_backup_$(date +%Y%m%d).tar.gz models/ data/

# Upload to S3
aws s3 cp ml_pipeline_backup_*.tar.gz s3://your-bucket/backups/
```

## Scaling Considerations

### Horizontal Scaling
- Use load balancer (nginx, HAProxy)
- Run multiple API containers
- Use shared storage for models

### Performance Optimization
- Cache predictions when possible
- Use batch prediction endpoint
- Implement request queuing
- Monitor response times

## Security

### Best Practices

1. **Environment Variables:**
   ```bash
   # Don't commit .env file
   echo ".env" >> .gitignore
   ```

2. **API Authentication:**
   - Implement API key authentication
   - Use HTTPS/TLS
   - Rate limiting

3. **Model Security:**
   - Validate model integrity
   - Version control models
   - Monitor for model drift

4. **Network Security:**
   - Firewall rules
   - VPC/Security groups
   - WAF (Web Application Firewall)

## Rollback Procedure

If issues occur in production:

```bash
# With Docker Compose
docker-compose down
docker-compose up -d  # This will use previous image

# With Kubernetes
kubectl rollout undo deployment/ml-pipeline-api
kubectl rollout status deployment/ml-pipeline-api
```

## Disaster Recovery

1. **Backup Schedule:**
   - Daily automated backups
   - Store in multiple locations
   - Test recovery regularly

2. **Recovery Procedure:**
   ```bash
   # Restore from backup
   tar -xzf ml_pipeline_backup_YYYYMMDD.tar.gz
   
   # Retrain if needed
   python train_pipeline.py
   
   # Restart service
   docker-compose restart
   ```

## Cost Optimization

- Use spot instances for non-critical workloads
- Implement auto-scaling
- Monitor resource usage
- Clean up old models/logs regularly

## Troubleshooting

### API Won't Start

```bash
# Check logs
docker-compose logs ml-pipeline

# Check port availability
lsof -i :8000

# Check model file
ls -lh models/
```

### High Memory Usage

```bash
# Monitor memory
docker stats

# Reduce batch size
# Implement request queue
# Use model caching
```

### Slow Predictions

```bash
# Check logs for slow queries
grep "took" logs/*.log

# Profile code
python -m cProfile -s cumtime train_pipeline.py

# Optimize model
# Use GPU acceleration
# Implement caching
```

## Support & Maintenance

- Monitor system logs daily
- Review performance metrics weekly
- Update dependencies monthly
- Conduct security audits quarterly

## Additional Resources

- [Docker Production Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [ML Model Deployment](https://mlflow.org/docs/latest/models/)
