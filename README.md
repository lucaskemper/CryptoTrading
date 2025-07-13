# 🚀 Crypto Trading & Arbitrage Bot

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-ready-blue.svg)](https://kubernetes.io/)

> **Advanced automated trading and arbitrage bot for Solana and Ethereum with statistical arbitrage strategies and LLM-powered sentiment analysis.**

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This is a sophisticated crypto trading bot that combines **statistical arbitrage** with **LLM-powered sentiment analysis** to identify and execute profitable trading opportunities. The bot supports multiple exchanges, real-time data collection, advanced risk management, and cloud-ready deployment.

### Key Capabilities

- **Multi-Exchange Support**: Binance, Kraken, and more
- **Statistical Arbitrage**: Cointegration analysis, Z-score signals, mean reversion
- **Sentiment Analysis**: News, Reddit, Twitter sentiment using GPT-3.5-turbo
- **Risk Management**: Advanced position sizing, stop-loss, take-profit
- **Real-time Monitoring**: Web dashboard, Prometheus metrics, Grafana
- **Cloud Ready**: Docker, Kubernetes, AWS/GCP deployment
- **Backtesting**: Historical strategy validation and optimization

## ✨ Features

### 🤖 Trading Strategies
- **Statistical Arbitrage**: Cointegration-based pair trading
- **Sentiment Analysis**: LLM-powered market sentiment scoring
- **Signal Combination**: Multi-strategy signal fusion
- **Portfolio Rebalancing**: Dynamic position management
- **Mean Reversion**: RSI, Bollinger Bands, MACD indicators

### 📊 Data Collection
- **Real-time Market Data**: Price, volume, order book
- **Sentiment Sources**: Reddit, news APIs, social media
- **Historical Data**: Backtesting and strategy validation
- **Multi-Asset Support**: ETH, SOL, BTC, and 20+ cryptocurrencies

### 🛡️ Risk Management
- **Position Sizing**: Dynamic allocation based on volatility
- **Stop-Loss/Take-Profit**: Automated risk controls
- **Portfolio Limits**: Maximum exposure and drawdown controls
- **Correlation Analysis**: Diversification and risk mitigation
- **Volatility Monitoring**: Real-time risk assessment

### 🔧 Technical Features
- **Async Architecture**: High-performance concurrent processing
- **Modular Design**: Pluggable strategies and components
- **Comprehensive Testing**: 90%+ test coverage
- **Production Ready**: Monitoring, logging, error handling
- **Scalable**: Horizontal and vertical scaling support

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Strategy      │    │   Execution     │
│                 │    │   Engine        │    │   Engine        │
│ • Exchanges     │───▶│ • Stat Arbitrage│───▶│ • Order Manager │
│ • News APIs     │    │ • Sentiment     │    │ • Risk Manager  │
│ • Social Media  │    │ • Signal Gen    │    │ • Position Mgr  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Store    │    │   Monitoring    │    │   Dashboard     │
│                 │    │                 │    │                 │
│ • PostgreSQL    │    │ • Prometheus    │    │ • Web UI        │
│ • Redis Cache   │    │ • Grafana       │    │ • Real-time     │
│ • CSV Files     │    │ • Health Checks │    │ • Performance   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Modules

- **`src/data_collector.py`**: Multi-source data ingestion
- **`src/strategy/stat_arb.py`**: Statistical arbitrage logic
- **`src/strategy/sentiment.py`**: LLM sentiment analysis
- **`src/strategy/signal_generator.py`**: Signal combination
- **`src/execution/order_manager.py`**: Order execution
- **`src/execution/risk_manager.py`**: Risk controls
- **`src/utils/monitoring.py`**: Metrics and health checks

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional)
- API keys for exchanges and services

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/crypto-trading-bot.git
cd crypto-trading-bot

# Setup development environment
./setup_dev.sh

# Or manually:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy and edit configuration
cp config/config.yaml config/local.yaml

# Add your API keys to secrets file
nano config/secrets.env
```

**Required API Keys:**
```bash
# Exchange APIs
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_SECRET_KEY=your_kraken_secret_key

# Sentiment Analysis
OPENAI_API_KEY=your_openai_api_key
NEWSAPI_KEY=your_newsapi_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
```

### 3. Run the Bot

```bash
# Simulation mode (default)
python run_bot.py --simulation

# Live trading mode
python run_bot.py --live

# Test mode
python run_bot.py --test
```

### 4. Start Dashboard

```bash
# Web dashboard
python dashboard.py

# Access at: http://localhost:5001
```

## ⚙️ Configuration

### Trading Parameters

```yaml
# config/config.yaml
strategy:
  statistical_arbitrage:
    enabled: true
    z_score_threshold: 1.0
    cointegration_lookback: 50
    correlation_threshold: 0.3
    
  sentiment_analysis:
    enabled: true
    model: "gpt-3.5-turbo"
    confidence_threshold: 0.5

risk:
  max_position_size: 10000
  risk_per_trade: 0.10
  stop_loss_percentage: 0.08
  take_profit_percentage: 0.15
  max_total_exposure: 0.7
```

### Risk Management

- **Position Sizing**: Dynamic allocation based on volatility
- **Stop-Loss**: 8% default, configurable per strategy
- **Take-Profit**: 15% default, trailing stops available
- **Portfolio Limits**: 70% max exposure, 30% per asset
- **Drawdown Controls**: 5% daily, 15% total maximum

## 📖 Usage

### Basic Usage

```python
from src.main import TradingBot

# Initialize bot
bot = TradingBot()

# Start in simulation mode
await bot.start()
```

### Strategy Examples

#### Statistical Arbitrage
```python
from src.strategy.stat_arb import StatisticalArbitrage

# Initialize strategy
stat_arb = StatisticalArbitrage(
    z_score_threshold=1.0,
    correlation_threshold=0.3
)

# Generate signals
signals = await stat_arb.generate_signals(market_data)
```

#### Sentiment Analysis
```python
from src.strategy.sentiment import SentimentAnalyzer

# Initialize analyzer
sentiment = SentimentAnalyzer(model="gpt-3.5-turbo")

# Analyze sentiment
score = await sentiment.analyze_text("Bitcoin reaches new highs")
```

### Demo Applications

```bash
# Statistical arbitrage demo
python examples/stat_arb_demo.py

# Sentiment analysis demo
python examples/sentiment_demo.py

# Order management demo
python examples/order_manager_demo.py

# Backtesting demo
python examples/backtest_demo.py
```

## 🔌 API Documentation

### REST Endpoints

- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /status` - Bot status and performance
- `GET /positions` - Current positions
- `GET /trades` - Recent trades
- `GET /signals` - Generated signals

### WebSocket Events

- `market_data` - Real-time price updates
- `signal_generated` - New trading signals
- `position_update` - Position changes
- `trade_executed` - Trade confirmations

## 🧪 Testing

### Run All Tests
```bash
# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test suites
python -m pytest tests/test_stat_arb.py -v
python -m pytest tests/test_sentiment.py -v
python -m pytest tests/test_risk_manager.py -v
```

### Test Coverage
- **Unit Tests**: 90%+ coverage
- **Integration Tests**: End-to-end workflows
- **Performance Tests**: Load and stress testing
- **Security Tests**: API key validation, input sanitization

## 🚀 Deployment

### Docker Deployment

```bash
# Build image
docker build -t crypto-trading-bot .

# Run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps
```

### Kubernetes Deployment

```bash
# Deploy to cluster
kubectl apply -f k8s/

# Check deployment
kubectl get pods -n crypto-trading

# View logs
kubectl logs -f deployment/trading-bot -n crypto-trading
```

### Cloud Deployment

#### AWS ECS
```bash
# Deploy to ECS
aws ecs create-service --cluster crypto-trading --service-name trading-bot
```

#### Google Cloud Run
```bash
# Deploy to Cloud Run
gcloud run deploy crypto-trading-bot --source .
```

## 📊 Monitoring

### Metrics Dashboard

Access Grafana at `http://localhost:3000` (admin/admin)

**Key Metrics:**
- Trading performance (PnL, win rate, Sharpe ratio)
- System resources (CPU, memory, network)
- API response times and error rates
- Portfolio exposure and risk metrics

### Alerts

- **High Drawdown**: >15% portfolio loss
- **API Errors**: >5% error rate
- **System Resources**: >80% CPU/memory usage
- **Trading Alerts**: Large position changes

### Logging

- **Structured Logging**: JSON format with correlation IDs
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Log Rotation**: Daily rotation with compression
- **Centralized Logging**: ELK stack integration

## 🔧 Development

### Project Structure
```
crypto-trading-bot/
├── src/                    # Main source code
│   ├── strategy/          # Trading strategies
│   ├── execution/         # Order execution
│   ├── utils/             # Utilities and helpers
│   └── main.py           # Main application
├── tests/                 # Test suite
├── examples/              # Demo applications
├── config/               # Configuration files
├── docker/               # Docker configurations
├── k8s/                  # Kubernetes manifests
├── docs/                 # Documentation
└── scripts/              # Utility scripts
```

### Development Setup

```bash
# Setup development environment
./setup_dev.sh

# Install pre-commit hooks
pre-commit install

# Run linting
flake8 src/ tests/
black src/ tests/

# Run type checking
mypy src/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Performance

### Benchmarks

- **Data Processing**: 1000+ market data points/second
- **Signal Generation**: <100ms latency
- **Order Execution**: <50ms average
- **Memory Usage**: <2GB typical
- **CPU Usage**: <30% average

### Scalability

- **Horizontal Scaling**: Multiple bot instances
- **Vertical Scaling**: Resource limits and requests
- **Database Scaling**: Read replicas, sharding
- **Cache Scaling**: Redis cluster, CDN

## 🛡️ Security

### Security Features

- **API Key Encryption**: Secure storage and rotation
- **Network Security**: TLS/SSL encryption
- **Input Validation**: Sanitized user inputs
- **Rate Limiting**: API abuse prevention
- **Audit Logging**: Complete activity tracking

### Best Practices

- Never commit API keys to version control
- Use environment variables for secrets
- Regularly rotate API keys
- Monitor for suspicious activity
- Keep dependencies updated

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

### Getting Help

- **Documentation**: Check the `docs/` directory
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join community discussions
- **Email**: support@cryptotradingbot.com

### Community

- **Discord**: Join our trading community
- **Telegram**: Real-time updates and alerts
- **Twitter**: Follow for news and updates
- **Blog**: Technical articles and tutorials

---

**⚠️ Disclaimer**: This software is for educational and research purposes. Trading cryptocurrencies involves substantial risk of loss. Use at your own risk and never invest more than you can afford to lose.

**📊 Performance**: Past performance does not guarantee future results. Always backtest strategies thoroughly before live trading.

---

Made with ❤️ by the Crypto Trading Bot Team
