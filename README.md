# Crypto Trading & Arbitrage Bot

🚀 **Production-Ready** automated trading and arbitrage bot for Solana, Ethereum, and 20+ cryptocurrencies, combining statistical arbitrage strategies with LLM-powered sentiment analysis.

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://docs.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-blue?logo=kubernetes)](https://kubernetes.io/)
[![Prometheus](https://img.shields.io/badge/Monitoring-Prometheus-orange?logo=prometheus)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Dashboard-Grafana-orange?logo=grafana)](https://grafana.com/)

## 🌟 Features

- **Statistical Arbitrage**: Cointegration-based pair trading with Z-score signals
- **Sentiment Analysis**: LLM-powered market sentiment from news and social media
- **Risk Management**: Dynamic position sizing, stop-loss, and take-profit
- **Real-time Execution**: High-frequency and daily trading capabilities
- **Backtesting**: Historical strategy validation and optimization
- **Multi-Asset Support**: Trade 20+ cryptocurrencies out of the box
- **Cloud Ready**: Full Docker & Kubernetes deployment with monitoring
- **Production Monitoring**: Prometheus metrics, Grafana dashboards, health checks
- **Database Integration**: PostgreSQL with comprehensive schema
- **Security**: Non-root containers, secrets management, rate limiting

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Server    │    │   Monitoring    │    │   Database      │
│   (Port 8080)   │    │   (Prometheus)  │    │   (PostgreSQL)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────────────────────┼─────────────────────────────────┐
│                    Trading Bot Core                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │Data Collector│ │  Strategy   │ │Risk Manager │ │Order Manager│ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd CryptoTrading

# Deploy with Docker Compose
./scripts/deploy.sh deploy-docker

# Check status
./scripts/deploy.sh status docker

# View logs
./scripts/deploy.sh logs docker

# Health check
./scripts/deploy.sh health
```

### Option 2: Kubernetes Deployment

```bash
# Deploy to Kubernetes
./scripts/deploy.sh deploy-k8s

# Check status
./scripts/deploy.sh status k8s

# View logs
./scripts/deploy.sh logs k8s
```

### Option 3: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp config/secrets.env.example config/secrets.env
# Edit config/secrets.env with your API keys

# Run the bot
python src/main.py
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

## 🗄️ Database Schema

The PostgreSQL database includes comprehensive tables for:

- **Market Data**: Price and volume data from exchanges
- **Sentiment Data**: Social media and news sentiment
- **Trading Signals**: Generated trading signals
- **Positions**: Open and closed trading positions
- **Orders**: Order execution history
- **Portfolio Metrics**: Performance and risk metrics
- **System Logs**: Application logs and errors

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

## Supported Assets

The bot supports 20+ major cryptocurrencies by default, including:

```
BTC, ETH, SOL, ADA, DOT, LINK, MATIC, AVAX, UNI, ATOM,
LTC, XRP, BCH, ETC, FIL, NEAR, ALGO, VET, ICP, FTM
```

You can add more assets by editing `config/config.yaml` under the `trading.symbols` list.

## Multi-Asset Data Collection

A new script, `collect_multi_assets.py`, is provided to collect historical and sample data for all supported assets:

### Usage

```bash
# Generate sample data for all assets (no API keys required)
python3 collect_multi_assets.py

# (Optional) Collect real historical data (requires API keys and uncommenting in the script)
# python3 collect_multi_assets.py --real
```

Sample and real data will be saved in `data/market_data_<SYMBOL>/` for each asset.

## Statistical Arbitrage Strategy

The statistical arbitrage module (`src/strategy/stat_arb.py`) implements a comprehensive pair trading strategy:

### Key Features

1. **Pair Selection & Cointegration Testing**
   - Identifies cointegrated asset pairs using statistical tests
   - Calculates correlation coefficients and cointegration p-values
   - Filters pairs based on correlation and spread volatility thresholds

2. **Spread Calculation & Signal Generation**
   - Continuously computes price spreads between selected pairs
   - Calculates Z-scores to measure deviation from historical mean
   - Generates entry signals when Z-score exceeds thresholds
   - Creates exit signals on mean reversion or risk management triggers

3. **Position Management & Risk Control**
   - Dynamic position sizing based on volatility
   - Market-neutral positions (long one asset, short the other)
   - Stop-loss and take-profit mechanisms
   - Real-time PnL tracking and performance metrics

4. **Performance Tracking**
   - Win rate, Sharpe ratio, and drawdown calculations
   - Trade history and signal logging
   - Real-time performance monitoring

### Usage Example

```python
from src.strategy.stat_arb import StatisticalArbitrage

# Create strategy instance
config = {
    'z_score_threshold': 2.0,
    'lookback_period': 100,
    'correlation_threshold': 0.7
}
strategy = StatisticalArbitrage(config)

# Update with market data
strategy.update_price_data('ETH', 2000.0, datetime.now())
strategy.update_price_data('SOL', 100.0, datetime.now())

# Find cointegrated pairs
assets = ['ETH', 'SOL', 'BTC', 'ADA']
cointegrated_pairs = strategy.find_cointegrated_pairs(assets)

# Generate trading signals
signals = strategy.generate_signals()

# Update positions
strategy.update_positions(signals)

# Get performance summary
performance = strategy.get_performance_summary()
```

### Strategy Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `z_score_threshold` | Z-score threshold for signal generation | 2.0 |
| `lookback_period` | Historical data window for calculations | 100 |
| `correlation_threshold` | Minimum correlation for pair selection | 0.7 |
| `cointegration_pvalue_threshold` | Maximum p-value for cointegration test | 0.05 |
| `position_size_limit` | Maximum position size as portfolio fraction | 0.1 |
| `stop_loss_pct` | Stop-loss percentage | 0.05 |
| `take_profit_pct` | Take-profit percentage | 0.1 |

## Core Modules

### Order Manager

The order manager (`src/execution/order_manager.py`) is the core execution component responsible for:

#### Key Features

1. **Order Creation & Submission**
   - Converts trade signals into executable orders
   - Supports market, limit, and stop orders
   - Handles multiple exchanges via CCXT library
   - Automatic retry logic with exponential backoff

2. **Order Tracking & Management**
   - Real-time order status monitoring
   - Partial fill handling
   - Order cancellation and replacement
   - Exchange-specific order parameter formatting

3. **Risk Integration**
   - Pre-trade risk checks
   - Position size validation
   - Integration with risk manager
   - Slippage protection for market orders

4. **Error Handling & Robustness**
   - Comprehensive error handling
   - Network failure recovery
   - Order rejection handling
   - Detailed logging and audit trail

#### Usage Example

```python
from src.execution.order_manager import OrderManager, TradeSignal, OrderSide, OrderType

# Initialize order manager
order_manager = OrderManager("binance")

# Create trade signal
signal = TradeSignal(
    symbol="BTC/USDT",
    side=OrderSide.BUY,
    quantity=0.001,
    order_type=OrderType.LIMIT,
    price=50000.0,
    strategy_name="stat_arb"
)

# Submit order
order = await order_manager.submit_order(signal)

# Track order status
await order_manager.track_order(order.id)

# Cancel if needed
await order_manager.cancel_order(order.id)
```

### Risk Manager

The enhanced risk manager (`src/execution/risk_manager.py`) provides comprehensive risk management with portfolio integration, sector analysis, and database logging:

#### Key Features

1. **Portfolio-Level Risk Management**
   - Total exposure monitoring across all positions
   - Sector-based risk analysis and limits
   - Correlation matrix calculations
   - Value at Risk (VaR) calculations

2. **Position-Level Controls**
   - Individual position size limits
   - Stop-loss and take-profit monitoring
   - Real-time PnL tracking
   - Position correlation limits

3. **Circuit Breakers & Emergency Controls**
   - Maximum drawdown limits
   - Daily loss limits
   - Emergency position closure
   - Trading suspension capabilities

4. **Database Integration**
   - Risk metrics logging to PostgreSQL
   - Historical risk analysis
   - Performance tracking
   - Audit trail maintenance

#### Usage Example

```python
from src.execution.risk_manager import RiskManager

# Initialize risk manager
risk_manager = RiskManager(
    max_position_size=10000,
    max_total_exposure=0.8,
    max_daily_drawdown=0.05
)

# Check if order is allowed
is_allowed, reason = risk_manager.check_order_risk({
    'symbol': 'BTC/USDT',
    'side': 'buy',
    'quantity': 0.1,
    'price': 50000
})

if is_allowed:
    # Execute order
    pass
else:
    print(f"Order rejected: {reason}")
```

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

## 🌐 Cloud Platform Support

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

- [Cloud Deployment Guide](CLOUD_DEPLOYMENT.md) - Detailed cloud deployment instructions
- [Architecture Documentation](docs/architecture.md) - System architecture details
- [API Documentation](docs/api.md) - REST API endpoints
- [Strategy Documentation](docs/strategy.md) - Trading strategy details

## 🤝 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review application logs
3. Check system resource usage
4. Verify configuration settings
5. Test individual components

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⚠️ Disclaimer**: This software is for educational and research purposes. Trading cryptocurrencies involves substantial risk of loss. Always test thoroughly in simulation mode before using real funds.