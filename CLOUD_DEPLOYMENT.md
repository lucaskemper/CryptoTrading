# Cloud Deployment Guide

This guide covers deploying the Crypto Trading & Arbitrage Bot to various cloud platforms with full monitoring, logging, and scalability features.

## 🏗️ Architecture Overview

The cloud-ready trading bot includes:

- **Multi-stage Docker builds** for optimized container images
- **Kubernetes manifests** for orchestration and scaling
- **PostgreSQL database** for persistent data storage
- **Redis cache** for session management and caching
- **Prometheus + Grafana** for monitoring and alerting
- **Nginx reverse proxy** for load balancing and SSL termination
- **Health checks** and metrics endpoints
- **Automated backups** and disaster recovery

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Kubernetes cluster (optional)
- API keys for exchanges and services

### 1. Docker Compose Deployment

```bash
# Build and deploy
./scripts/deploy.sh deploy-docker

# Check status
./scripts/deploy.sh status docker

# View logs
./scripts/deploy.sh logs docker

# Health check
./scripts/deploy.sh health
```

### 2. Kubernetes Deployment

```bash
# Deploy to Kubernetes
./scripts/deploy.sh deploy-k8s

# Check status
./scripts/deploy.sh status k8s

# View logs
./scripts/deploy.sh logs k8s
```

## 📊 Monitoring & Observability

### Health Endpoints

- **Health Check**: `GET /health`
- **Metrics**: `GET /metrics` (Prometheus format)
- **Status**: `GET /status` (Detailed JSON)
- **Trading Info**: `GET /trading/info`
- **System Info**: `GET /system/info`

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (default: admin/admin)

Pre-configured dashboards include:
- Trading performance metrics
- System resource usage
- Error rates and latency
- Portfolio P&L tracking

### Prometheus Metrics

Key metrics collected:
- Trading signals generated
- Orders executed
- Portfolio P&L
- System resource usage
- API response times
- Error rates

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Deployment environment | `production` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `TRADING_ENABLED` | Enable live trading | `false` |
| `SIMULATION_MODE` | Run in simulation mode | `true` |
| `REDIS_URL` | Redis connection string | `redis://redis:6379` |
| `DATABASE_URL` | PostgreSQL connection string | Auto-configured |

### API Keys (Secrets)

Update `k8s/secrets.yaml` with your API keys:

```yaml
data:
  binance_api_key: "your-base64-encoded-key"
  binance_secret_key: "your-base64-encoded-secret"
  kraken_api_key: "your-base64-encoded-key"
  kraken_secret_key: "your-base64-encoded-secret"
  openai_api_key: "your-base64-encoded-key"
  newsapi_key: "your-base64-encoded-key"
  reddit_client_id: "your-base64-encoded-id"
  reddit_client_secret: "your-base64-encoded-secret"
```

## 🗄️ Database Schema

The PostgreSQL database includes tables for:

- **Market Data**: Price and volume data from exchanges
- **Sentiment Data**: Social media and news sentiment
- **Trading Signals**: Generated trading signals
- **Positions**: Open and closed trading positions
- **Orders**: Order execution history
- **Portfolio Metrics**: Performance and risk metrics
- **System Logs**: Application logs and errors

## 🔒 Security Features

- **Non-root containers** for security
- **Secrets management** for API keys
- **Network isolation** with Kubernetes namespaces
- **Rate limiting** via Nginx
- **Health checks** for automatic recovery
- **Resource limits** to prevent resource exhaustion

## 📈 Scaling

### Horizontal Scaling

```bash
# Scale trading bot replicas
kubectl scale deployment trading-bot --replicas=3 -n crypto-trading
```

### Vertical Scaling

Update resource limits in `k8s/deployment.yaml`:

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

## 🔄 Backup & Recovery

### Automated Backups

Database backups run automatically:

```bash
# Manual backup
docker exec trading-backup /backup.sh

# Restore from backup
docker exec -i trading-postgres psql -U tradingbot -d tradingbot < backup.sql
```

### Disaster Recovery

1. **Database Recovery**:
   ```bash
   kubectl exec -it postgres-pod -n crypto-trading -- pg_restore -U tradingbot -d tradingbot backup.sql
   ```

2. **Configuration Recovery**:
   ```bash
   kubectl get configmap trading-bot-config -n crypto-trading -o yaml > config-backup.yaml
   ```

## 🐛 Troubleshooting

### Common Issues

1. **Container won't start**:
   ```bash
   # Check logs
   kubectl logs deployment/trading-bot -n crypto-trading
   
   # Check events
   kubectl get events -n crypto-trading
   ```

2. **Health check failing**:
   ```bash
   # Check endpoint
   curl http://localhost:8080/health
   
   # Check system resources
   kubectl top pods -n crypto-trading
   ```

3. **Database connection issues**:
   ```bash
   # Test database connection
   kubectl exec -it postgres-pod -n crypto-trading -- psql -U tradingbot -d tradingbot -c "SELECT 1"
   ```

### Log Analysis

```bash
# View application logs
kubectl logs -f deployment/trading-bot -n crypto-trading

# View system logs
kubectl logs -f deployment/postgres -n crypto-trading

# Search for errors
kubectl logs deployment/trading-bot -n crypto-trading | grep ERROR
```

## 🌐 Cloud Platform Specifics

### AWS EKS

```bash
# Create EKS cluster
eksctl create cluster --name trading-cluster --region us-west-2

# Deploy to EKS
kubectl apply -f k8s/
```

### Google GKE

```bash
# Create GKE cluster
gcloud container clusters create trading-cluster --zone us-central1-a

# Deploy to GKE
kubectl apply -f k8s/
```

### Azure AKS

```bash
# Create AKS cluster
az aks create --resource-group trading-rg --name trading-cluster

# Deploy to AKS
kubectl apply -f k8s/
```

## 📋 Maintenance

### Regular Tasks

1. **Monitor resource usage**:
   ```bash
   kubectl top pods -n crypto-trading
   ```

2. **Update configurations**:
   ```bash
   kubectl apply -f k8s/configmap.yaml
   ```

3. **Rotate secrets**:
   ```bash
   kubectl apply -f k8s/secrets.yaml
   ```

4. **Clean up old data**:
   ```bash
   kubectl exec -it postgres-pod -n crypto-trading -- psql -U tradingbot -d tradingbot -c "DELETE FROM system_logs WHERE created_at < NOW() - INTERVAL '30 days';"
   ```

## 🚨 Alerts & Notifications

Configure alerts in Grafana for:

- High CPU/Memory usage
- Trading bot errors
- Database connection failures
- Low disk space
- Unusual trading activity

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

## 🤝 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review application logs
3. Check system resource usage
4. Verify configuration settings
5. Test individual components

---

**Note**: Always test in a staging environment before deploying to production. Monitor the system closely during initial deployment and adjust resource limits as needed. 