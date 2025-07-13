# Crypto Trading & Arbitrage Bot

Automated trading and arbitrage bot for Solana and Ethereum, combining statistical arbitrage strategies with LLM-powered sentiment analysis.

## Features

- **Statistical Arbitrage**: Cointegration-based pair trading with Z-score signals
- **Sentiment Analysis**: LLM-powered market sentiment from news and social media
- **Risk Management**: Dynamic position sizing, stop-loss, and take-profit
- **Real-time Execution**: High-frequency and daily trading capabilities
- **Backtesting**: Historical strategy validation and optimization
- **Cloud Ready**: Docker containerization and cloud deployment support

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd CryptoTrading

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp config/secrets.env.example config/secrets.env
# Edit config/secrets.env with your API keys
```

### Configuration

Edit `config/config.yaml` to customize strategy parameters:

```yaml
strategy:
  statistical_arbitrage:
    enabled: true
    z_score_threshold: 2.0
    cointegration_lookback: 100
    correlation_threshold: 0.7
    position_size_limit: 0.1
    stop_loss_pct: 0.05
    take_profit_pct: 0.1
```

### Running the Demo

```bash
# Run the statistical arbitrage demo
python examples/stat_arb_demo.py

# Run tests
python -m pytest tests/
```

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

## Architecture

### Core Modules

- **`src/strategy/stat_arb.py`**: Statistical arbitrage implementation
- **`src/strategy/sentiment.py`**: LLM sentiment analysis
- **`src/strategy/signal_generator.py`**: Signal combination logic
- **`src/execution/order_manager.py`**: Order execution and management
- **`src/execution/risk_manager.py`**: Risk controls and position management
- **`src/data_collector.py`**: Market and sentiment data collection
- **`src/utils/`**: Logging, configuration, and utility functions

### Data Flow

1. **Data Collection**: Real-time price and sentiment data ingestion
2. **Pair Analysis**: Cointegration testing and spread calculation
3. **Signal Generation**: Z-score based entry/exit signals
4. **Risk Management**: Position sizing and risk controls
5. **Execution**: Order placement and position management
6. **Monitoring**: Performance tracking and logging

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_stat_arb.py

# Run with coverage
python -m pytest --cov=src tests/
```

## Development

### Adding New Features

1. Follow the modular structure in `src/`
2. Add corresponding tests in `tests/`
3. Update configuration in `config/config.yaml`
4. Document changes in `docs/`

### Code Standards

- Python 3.9+ syntax
- PEP8 style guide
- Type hints where possible
- Comprehensive docstrings
- Unit tests for all functions

## Deployment

### Docker

```bash
# Build and run with Docker
docker-compose up --build

# Or build manually
docker build -t crypto-trading-bot .
docker run -d crypto-trading-bot
```

### Cloud Deployment

The bot is designed for cloud deployment on AWS, GCP, or Azure with:
- Container orchestration (Kubernetes, ECS)
- Auto-scaling based on market conditions
- Monitoring and alerting
- Secure secret management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license information here]

## Support

For questions and support, please open an issue on GitHub.