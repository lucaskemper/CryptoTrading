# Backtesting System Documentation

## Overview

The backtesting system provides a comprehensive framework for testing crypto trading strategies using historical data simulation. It includes realistic market data generation, strategy execution, risk management, and detailed performance analysis.

## Architecture

### Core Components

1. **BacktestEngine** - Main orchestrator that runs complete backtests
2. **DataSimulator** - Generates realistic historical market data
3. **StrategyRunner** - Executes trading strategies during simulation
4. **PerformanceAnalyzer** - Calculates performance metrics and generates reports

### Data Flow

```
DataSimulator → BacktestEngine → StrategyRunner → PerformanceAnalyzer
     ↓              ↓                ↓                ↓
Market Data → Strategy Signals → Trade Execution → Performance Report
```

## Quick Start

### Basic Backtest

```python
from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig
from datetime import datetime, timedelta

# Create configuration
config = BacktestConfig(
    start_date=datetime.now() - timedelta(days=180),
    end_date=datetime.now(),
    symbols=['ETH/USDT', 'SOL/USDT', 'BTC/USDT'],
    initial_capital=100000.0,
    strategy_config={
        'statistical_arbitrage': {
            'z_score_threshold': 2.0,
            'lookback_period': 100
        }
    }
)

# Run backtest
engine = BacktestEngine(config)
results = engine.run_backtest()

# Print results
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
```

### Command Line Usage

```bash
# Basic backtest
python run_backtest.py --days 180 --capital 100000

# With custom parameters
python run_backtest.py \
    --days 90 \
    --capital 50000 \
    --z-threshold 2.5 \
    --lookback 150 \
    --position-size 0.15 \
    --sentiment \
    --plots \
    --save
```

## Configuration

### BacktestConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_date` | datetime | - | Start date for backtest |
| `end_date` | datetime | - | End date for backtest |
| `symbols` | List[str] | ['ETH/USDT', 'SOL/USDT', 'BTC/USDT'] | Trading symbols |
| `initial_capital` | float | 100000.0 | Initial portfolio value |
| `max_position_size` | float | 0.1 | Maximum position size (fraction) |
| `stop_loss_pct` | float | 0.05 | Stop loss percentage |
| `take_profit_pct` | float | 0.1 | Take profit percentage |
| `slippage` | float | 0.001 | Slippage per trade |
| `commission` | float | 0.001 | Commission per trade |
| `data_frequency` | str | '1h' | Data frequency (1m, 5m, 15m, 1h, 4h, 1d) |
| `sentiment_enabled` | bool | True | Enable sentiment analysis |
| `generate_plots` | bool | True | Generate performance plots |
| `save_results` | bool | True | Save results to files |

### Strategy Configuration

```python
strategy_config = {
    'statistical_arbitrage': {
        'z_score_threshold': 2.0,        # Z-score threshold for signals
        'lookback_period': 100,          # Historical data window
        'correlation_threshold': 0.7,    # Minimum correlation for pairs
        'cointegration_pvalue_threshold': 0.05,  # Cointegration test p-value
        'min_spread_std': 0.001,        # Minimum spread volatility
        'position_size_limit': 0.1,      # Maximum position size
        'stop_loss_pct': 0.05,          # Stop loss percentage
        'take_profit_pct': 0.1,         # Take profit percentage
        'max_positions': 5,              # Maximum concurrent positions
        'rebalance_frequency': 300,      # Rebalancing frequency (seconds)
        'dynamic_threshold_window': 100, # Window for dynamic thresholds
        'spread_model': 'ols',           # Spread calculation model
        'slippage': 0.001               # Slippage simulation
    },
    'sentiment_analysis': {
        'model': 'gpt-3.5-turbo',       # Sentiment model
        'confidence_threshold': 0.7,     # Minimum confidence
        'sentiment_weight': 0.3         # Weight in signal combination
    },
    'signal_generator': {
        'combination_method': 'weighted', # Signal combination method
        'stat_weight': 0.7,             # Statistical arbitrage weight
        'sentiment_weight': 0.3,        # Sentiment weight
        'min_confidence': 0.3,          # Minimum signal confidence
        'enable_risk_checks': True,     # Enable risk management
        'require_risk_approval': False  # Require risk approval
    },
    'risk_management': {
        'max_position_size': 0.1,       # Maximum position size
        'max_total_exposure': 0.8,      # Maximum total exposure
        'max_single_asset_exposure': 0.3, # Maximum single asset exposure
        'max_open_positions': 10,       # Maximum open positions
        'max_daily_drawdown': 0.05,     # Maximum daily drawdown
        'max_total_drawdown': 0.15,     # Maximum total drawdown
        'volatility_threshold': 0.1     # Volatility threshold
    }
}
```

## Data Simulation

### Market Data Generation

The `DataSimulator` generates realistic market data with the following features:

- **Price Movements**: Random walk with volatility clustering (GARCH-like effects)
- **Volume Patterns**: Daily and weekly patterns with realistic noise
- **OHLCV Data**: Complete OHLCV data with realistic spreads
- **Correlation**: Configurable correlation between assets
- **Trends**: Slight upward trends with mean reversion

### Sentiment Data Generation

When enabled, the system generates sentiment data with:

- **Sentiment Scores**: Normal distribution around zero
- **Persistence**: Autocorrelation in sentiment changes
- **Market Correlation**: Sentiment correlated with price movements
- **Confidence Levels**: Realistic confidence scores

## Performance Analysis

### Basic Metrics

- **Total Return**: Overall portfolio return
- **Annualized Return**: Annualized return rate
- **Volatility**: Annualized volatility
- **Maximum Drawdown**: Largest peak-to-trough decline

### Risk Metrics

- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return to maximum drawdown ratio
- **Information Ratio**: Excess return to tracking error
- **Value at Risk (VaR)**: 95% confidence level loss
- **Conditional VaR**: Expected loss beyond VaR

### Trade Analysis

- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit to gross loss ratio
- **Average Win/Loss**: Average profit and loss per trade
- **Largest Win/Loss**: Largest individual trade P&L
- **Trade Duration**: Average holding period

## Advanced Features

### Parameter Optimization

```python
# Test multiple parameter combinations
z_thresholds = [1.5, 2.0, 2.5, 3.0]
lookback_periods = [50, 100, 150, 200]

results = []
for z_threshold in z_thresholds:
    for lookback in lookback_periods:
        config.strategy_config['statistical_arbitrage']['z_score_threshold'] = z_threshold
        config.strategy_config['statistical_arbitrage']['lookback_period'] = lookback
        
        engine = BacktestEngine(config)
        result = engine.run_backtest()
        
        results.append({
            'z_threshold': z_threshold,
            'lookback': lookback,
            'sharpe_ratio': result.sharpe_ratio,
            'total_return': result.total_return
        })

# Find best parameters
best_result = max(results, key=lambda x: x['sharpe_ratio'])
```

### Strategy Comparison

```python
strategies = {
    'Statistical Arbitrage Only': {
        'statistical_arbitrage': {...},
        'sentiment_analysis': None
    },
    'Sentiment Only': {
        'statistical_arbitrage': None,
        'sentiment_analysis': {...}
    },
    'Combined Strategy': {
        'statistical_arbitrage': {...},
        'sentiment_analysis': {...}
    }
}

comparison_results = []
for name, config in strategies.items():
    engine = BacktestEngine(config)
    results = engine.run_backtest()
    comparison_results.append({
        'strategy': name,
        'sharpe_ratio': results.sharpe_ratio,
        'total_return': results.total_return
    })
```

### Custom Data Sources

```python
# Use real historical data instead of simulation
class CustomDataSimulator(DataSimulator):
    def _generate_market_data(self, symbol):
        # Load real historical data
        data = pd.read_csv(f'data/{symbol}_historical.csv')
        self.market_data[symbol] = data
```

## Output and Reports

### Generated Files

When `save_results=True`, the system generates:

- **`backtest_summary.json`**: Performance metrics summary
- **`trade_history.csv`**: Complete trade history
- **`equity_curve.csv`**: Portfolio value over time
- **`performance_report.html`**: HTML performance report
- **`performance_plots.png`**: Performance visualization plots

### Performance Report

The HTML report includes:

- Performance metrics summary
- Risk analysis
- Trade statistics
- Position analysis
- Signal analysis
- Performance attribution

### Plots Generated

1. **Equity Curve**: Portfolio value over time
2. **Drawdown**: Portfolio drawdown analysis
3. **Trade Distribution**: P&L distribution histogram
4. **Monthly Returns**: Monthly return bar chart

## Best Practices

### Configuration

1. **Start Small**: Begin with shorter time periods and fewer assets
2. **Parameter Testing**: Test multiple parameter combinations
3. **Risk Management**: Always include stop-loss and position sizing
4. **Realistic Assumptions**: Use realistic slippage and commission rates

### Analysis

1. **Multiple Metrics**: Don't rely on a single performance metric
2. **Out-of-Sample Testing**: Test on different time periods
3. **Walk-Forward Analysis**: Use rolling windows for parameter optimization
4. **Stress Testing**: Test under different market conditions

### Common Pitfalls

1. **Overfitting**: Avoid optimizing too many parameters
2. **Look-Ahead Bias**: Ensure no future data leakage
3. **Survivorship Bias**: Include delisted assets in historical data
4. **Transaction Costs**: Include realistic trading costs

## Troubleshooting

### Common Issues

1. **Memory Usage**: For long backtests, consider using lower frequency data
2. **Performance**: Use `data_frequency='1h'` for faster testing
3. **Data Quality**: Check generated data for realistic patterns
4. **Strategy Errors**: Ensure all strategy components are properly initialized

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug output
engine = BacktestEngine(config)
results = engine.run_backtest()
```

## Integration with Live Trading

The backtesting system is designed to be compatible with live trading:

1. **Same Strategies**: Use identical strategy components
2. **Same Risk Management**: Apply same risk rules
3. **Same Signal Generation**: Use same signal logic
4. **Performance Comparison**: Compare live vs backtest performance

## Future Enhancements

- **Real Data Integration**: Connect to real historical data sources
- **Advanced Risk Models**: Implement more sophisticated risk models
- **Machine Learning**: Add ML-based signal generation
- **Portfolio Optimization**: Add portfolio-level optimization
- **Real-Time Backtesting**: Support for real-time strategy testing 