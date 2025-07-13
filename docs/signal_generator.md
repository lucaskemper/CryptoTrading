# Signal Generator Module

## Overview

The Signal Generator is the central logic layer that combines outputs from statistical arbitrage strategy and sentiment analysis to produce actionable, risk-aware trading signals. It acts as the intelligent bridge between strategy modules and execution systems.

## Key Features

- **Multi-Signal Combination**: Combines statistical arbitrage and sentiment signals using various methods
- **Risk Management Integration**: Pre-trade risk validation through the risk manager
- **Configurable Logic**: Multiple combination methods and adjustable parameters
- **Comprehensive Logging**: Detailed signal tracking and performance metrics
- **Standardized Output**: Unified `TradeSignal` format for execution systems

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Statistical     │    │ Sentiment       │    │ Risk Manager    │
│ Arbitrage       │    │ Analyzer        │    │                 │
│ Strategy        │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Signal Generator │
                    │                 │
                    │ • Combine Logic  │
                    │ • Risk Checks    │
                    │ • Validation     │
                    │ • Formatting     │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Order Manager   │
                    │ (Execution)     │
                    └─────────────────┘
```

## Core Components

### TradeSignal Dataclass

Standardized signal structure with validation:

```python
@dataclass
class TradeSignal:
    symbol: str              # Trading pair (e.g., 'BTC-ETH')
    side: str               # 'buy' or 'sell'
    quantity: float         # Position size
    order_type: str         # 'market' or 'limit'
    price: Optional[float]  # Limit price (None for market)
    confidence: float       # Signal confidence (0-1)
    sources: List[str]      # Signal sources
    metadata: Dict[str, Any] # Additional data
    timestamp: datetime     # Generation timestamp
    signal_type: str        # Signal type
    risk_checked: bool      # Risk validation status
```

### SignalGenerator Class

Main class that orchestrates signal generation:

```python
class SignalGenerator:
    def __init__(self, stat_arb, sentiment_analyzer, risk_manager, config)
    def generate_signals(self, market_data, sentiment_data) -> List[TradeSignal]
    def update_config(self, new_config)
    def get_performance_metrics(self) -> Dict[str, Any]
```

## Signal Combination Methods

### 1. Consensus Method
Both statistical arbitrage and sentiment signals must agree on direction.

**Logic:**
- Statistical signal: `entry_long` + Sentiment: `positive` → Generate buy signal
- Statistical signal: `entry_short` + Sentiment: `negative` → Generate sell signal
- Any disagreement → No signal generated

**Use Case:** Conservative approach, reduces false signals

### 2. Weighted Method
Combines signals using configurable weights.

**Logic:**
```python
weighted_confidence = (stat_weight * stat_confidence + 
                      sentiment_weight * sentiment_confidence)
```

**Use Case:** Balanced approach, allows fine-tuning of signal importance

### 3. Filter Method
Uses sentiment as a filter for statistical signals.

**Logic:**
- Strong negative sentiment blocks long positions
- Strong positive sentiment blocks short positions
- Aligned sentiment increases confidence
- Contrary sentiment reduces confidence

**Use Case:** Sentiment-driven filtering, reduces risk in adverse market conditions

### 4. Hybrid Method
Combines consensus and weighted approaches.

**Logic:**
1. Try consensus first
2. If no consensus, fall back to weighted method
3. Apply minimum confidence threshold

**Use Case:** Adaptive approach, balances conservatism with signal generation

## Configuration

### Basic Configuration

```yaml
strategy:
  signal_generator:
    enabled: true
    combination_method: "consensus"  # consensus, weighted, filter, hybrid
    stat_weight: 0.6                # statistical arbitrage weight
    sentiment_weight: 0.4           # sentiment weight
    min_confidence: 0.3             # minimum confidence threshold
    enable_risk_checks: true        # enable risk manager integration
    require_risk_approval: false    # require risk manager approval
```

### Sentiment Thresholds

```yaml
sentiment_thresholds:
  positive: 0.2    # Above this = positive sentiment
  negative: -0.2   # Below this = negative sentiment
  neutral: 0.0     # Neutral sentiment level
```

### Risk Management

```yaml
enable_risk_checks: true           # Enable pre-trade risk validation
require_risk_approval: false       # Require risk manager approval
max_signals_per_batch: 10         # Maximum signals per generation cycle
signal_timeout_seconds: 300        # Signal validity timeout
```

## Usage Examples

### Basic Usage

```python
from src.strategy.signal_generator import SignalGenerator, create_signal_generator
from src.strategy.stat_arb import create_stat_arb_strategy
from src.strategy.sentiment import SentimentAnalyzer
from src.execution.risk_manager import RiskManager

# Create components
stat_arb = create_stat_arb_strategy()
sentiment_analyzer = SentimentAnalyzer()
risk_manager = RiskManager()

# Create signal generator
signal_gen = SignalGenerator(
    stat_arb=stat_arb,
    sentiment_analyzer=sentiment_analyzer,
    risk_manager=risk_manager
)

# Generate signals
signals = signal_gen.generate_signals(sentiment_data=sentiment_data)
```

### Factory Function

```python
# Using factory function
signal_gen = create_signal_generator(
    stat_arb=stat_arb,
    sentiment_analyzer=sentiment_analyzer,
    risk_manager=risk_manager,
    config={'combination_method': 'weighted'}
)
```

### Custom Configuration

```python
config = {
    'combination_method': 'filter',
    'stat_weight': 0.7,
    'sentiment_weight': 0.3,
    'min_confidence': 0.5,
    'sentiment_thresholds': {
        'positive': 0.3,
        'negative': -0.3,
        'neutral': 0.0
    }
}

signal_gen = SignalGenerator(
    stat_arb=stat_arb,
    sentiment_analyzer=sentiment_analyzer,
    risk_manager=risk_manager,
    config=config
)
```

## Signal Processing Pipeline

### 1. Signal Ingestion
- Fetch statistical arbitrage signals from `StatisticalArbitrage.generate_signals()`
- Aggregate sentiment data from `SentimentAnalyzer.aggregate_sentiment()`
- Align signals by timestamp and asset pairs

### 2. Signal Combination
- Apply configured combination method (consensus, weighted, filter, hybrid)
- Calculate confidence scores based on signal strength and alignment
- Filter signals below minimum confidence threshold

### 3. Signal Validation
- Validate signal parameters (symbol, side, quantity, etc.)
- Check for required fields and valid ranges
- Log validation results

### 4. Risk Management
- Submit signals to risk manager for pre-trade validation
- Apply position limits, exposure checks, and drawdown controls
- Mark signals as risk-checked or rejected

### 5. Signal Output
- Format signals into standardized `TradeSignal` objects
- Add metadata (z-scores, sentiment scores, combination method)
- Log signal generation summary
- Return actionable signals for execution

## Performance Metrics

The signal generator tracks various performance metrics:

```python
metrics = signal_gen.get_performance_metrics()
# Returns:
{
    'total_signals': 150,
    'approved_signals': 120,
    'rejected_signals': 30,
    'approval_rate': 0.8,
    'combination_method': 'consensus',
    'min_confidence': 0.3
}
```

## Signal History

Retrieve signal history for analysis:

```python
# Get signals from last 24 hours
history = signal_gen.get_signal_history(hours=24)

# History format:
[
    {
        'symbol': 'BTC-ETH',
        'side': 'buy',
        'quantity': 0.1,
        'confidence': 0.75,
        'timestamp': '2024-01-15T10:30:00',
        'sources': ['stat_arb', 'sentiment'],
        'risk_checked': True
    }
]
```

## Integration with Execution

### Order Manager Integration

```python
from src.execution.order_manager import OrderManager

order_manager = OrderManager()

# Generate signals
signals = signal_gen.generate_signals(sentiment_data=sentiment_data)

# Submit signals to order manager
for signal in signals:
    if signal.risk_checked:  # Only execute risk-approved signals
        order_manager.submit_order(signal)
```

### Risk Manager Integration

```python
# Risk manager automatically validates signals during generation
# Signals are marked with risk_checked status

for signal in signals:
    if signal.risk_checked:
        print(f"Signal {signal.symbol} approved by risk manager")
    else:
        print(f"Signal {signal.symbol} failed risk checks")
```

## Best Practices

### 1. Configuration Management
- Use configuration files for different market conditions
- Adjust weights based on market volatility
- Set appropriate confidence thresholds

### 2. Risk Management
- Always enable risk checks in production
- Monitor approval rates and adjust thresholds
- Use circuit breakers for extreme market conditions

### 3. Performance Monitoring
- Track signal generation metrics
- Monitor combination method effectiveness
- Analyze signal-to-execution ratios

### 4. Testing
- Test all combination methods
- Validate signal parameters
- Simulate risk manager integration

## Troubleshooting

### Common Issues

1. **No Signals Generated**
   - Check minimum confidence threshold
   - Verify statistical arbitrage signals
   - Review sentiment data quality

2. **Low Approval Rate**
   - Adjust risk manager parameters
   - Review position size limits
   - Check exposure constraints

3. **High False Positive Rate**
   - Increase minimum confidence threshold
   - Use consensus combination method
   - Adjust sentiment thresholds

### Debugging

Enable debug logging:

```python
import logging
logging.getLogger('src.strategy.signal_generator').setLevel(logging.DEBUG)
```

Check signal generation logs:

```python
# Signal generation includes detailed logging
signals = signal_gen.generate_signals(sentiment_data=sentiment_data)
# Check logs for signal combination details
```

## Future Enhancements

### Planned Features

1. **Machine Learning Integration**
   - ML-based signal weighting
   - Adaptive confidence thresholds
   - Pattern recognition for signal quality

2. **Advanced Combination Methods**
   - Ensemble methods
   - Time-series based weighting
   - Market regime detection

3. **Real-time Optimization**
   - Dynamic parameter adjustment
   - Performance-based learning
   - A/B testing framework

4. **Enhanced Risk Integration**
   - Real-time risk monitoring
   - Dynamic position sizing
   - Portfolio-level risk controls

## API Reference

### SignalGenerator Methods

- `generate_signals(market_data, sentiment_data)` → `List[TradeSignal]`
- `update_config(new_config)` → `None`
- `get_performance_metrics()` → `Dict[str, Any]`
- `get_signal_history(hours)` → `List[Dict[str, Any]]`
- `reset()` → `None`

### TradeSignal Methods

- `validate()` → `bool`
- All dataclass fields are accessible as attributes

### Factory Functions

- `create_signal_generator(stat_arb, sentiment_analyzer, risk_manager, config)` → `SignalGenerator`

## Related Modules

- **Statistical Arbitrage** (`src.strategy.stat_arb`): Generates statistical signals
- **Sentiment Analysis** (`src.strategy.sentiment`): Provides sentiment signals
- **Risk Manager** (`src.execution.risk_manager`): Validates signals
- **Order Manager** (`src.execution.order_manager`): Executes signals
- **Configuration** (`src.utils.config_loader`): Loads settings
- **Logging** (`src.utils.logger`): Provides logging functionality 