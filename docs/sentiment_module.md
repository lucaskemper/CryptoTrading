# Sentiment Analysis Module Documentation

## Overview

The `src/strategy/sentiment.py` module provides a comprehensive, research-ready sentiment analysis system for the crypto trading bot. It transforms raw sentiment data into actionable, quantitative signals that can be integrated with trading strategies.

## Features

### Core Functionality

1. **Text Preprocessing**
   - URL removal
   - Special character cleaning (preserves hashtags and mentions)
   - Case normalization
   - Whitespace trimming

2. **Sentiment Scoring**
   - TextBlob integration (default)
   - Normalized scores (-1 to +1)
   - Extensible for other models (VADER, transformers, etc.)

3. **Batch Processing**
   - Efficient processing of multiple sentiment items
   - Automatic keyword extraction
   - Logging for transparency

4. **Aggregation**
   - Rolling window aggregation (mean, median)
   - Configurable window sizes
   - Time-series indexed output

5. **Signal Generation**
   - Threshold-based signal classification
   - Configurable thresholds
   - Returns 'positive', 'negative', 'neutral'

6. **Keyword Extraction**
   - Crypto-specific keyword detection
   - Hashtag extraction
   - Mention detection
   - Duplicate removal

## Data Structures

### SentimentData

```python
@dataclass
class SentimentData:
    timestamp: datetime
    source: str
    text: str
    sentiment_score: Optional[float] = None
    keywords: Optional[List[str]] = None
    url: Optional[str] = None
```

**Validation:**
- Non-empty text required
- Sentiment score must be between -1 and 1 (if provided)

### SentimentAnalyzer

```python
class SentimentAnalyzer:
    def __init__(self, model: Optional[Any] = None):
        # Initialize with optional custom model
```

## API Reference

### Core Methods

#### `analyze_sentiment(text: str) -> float`
Analyzes sentiment of a single text string.
- **Input:** Raw text string
- **Output:** Normalized sentiment score (-1 to +1)
- **Example:**
```python
analyzer = SentimentAnalyzer()
score = analyzer.analyze_sentiment("Bitcoin is mooning! 🚀")
# Returns: 0.550
```

#### `batch_analyze(data: List[SentimentData]) -> List[SentimentData]`
Processes multiple sentiment items in batch.
- **Input:** List of SentimentData objects
- **Output:** Updated SentimentData objects with scores and keywords
- **Example:**
```python
data = [SentimentData(timestamp=now(), source="reddit", text="BTC pumping!")]
analyzed = analyzer.batch_analyze(data)
# Returns: List with sentiment_score and keywords populated
```

#### `aggregate_sentiment(data: List[SentimentData], window: int = 10, method: str = 'mean') -> pd.Series`
Aggregates sentiment scores over a rolling window.
- **Input:** List of SentimentData, window size, aggregation method
- **Output:** Pandas Series indexed by timestamp
- **Example:**
```python
agg_series = analyzer.aggregate_sentiment(data, window=5, method='mean')
latest_sentiment = agg_series.iloc[-1]
```

#### `generate_sentiment_signal(score: float, thresholds: Dict[str, float]) -> str`
Generates trading signals based on sentiment thresholds.
- **Input:** Sentiment score, threshold dictionary
- **Output:** 'positive', 'negative', or 'neutral'
- **Example:**
```python
signal = analyzer.generate_sentiment_signal(0.5, {'positive': 0.3, 'negative': -0.3})
# Returns: 'positive'
```

#### `extract_keywords(text: str) -> List[str]`
Extracts crypto-relevant keywords from text.
- **Input:** Text string
- **Output:** List of keywords, hashtags, and mentions
- **Example:**
```python
keywords = analyzer.extract_keywords("SOL is pumping! #Solana #Moon")
# Returns: ['sol', 'pump', 'Solana', 'Moon']
```

## Usage Examples

### Basic Sentiment Analysis

```python
from src.strategy.sentiment import SentimentAnalyzer, SentimentData
from datetime import datetime

# Initialize analyzer
analyzer = SentimentAnalyzer()

# Analyze single text
score = analyzer.analyze_sentiment("Bitcoin is absolutely mooning! 🚀")
print(f"Sentiment score: {score}")  # 0.550

# Extract keywords
keywords = analyzer.extract_keywords("ETH and SOL are both doing well")
print(f"Keywords: {keywords}")  # ['eth', 'sol']
```

### Batch Processing

```python
# Create sample data
data = [
    SentimentData(timestamp=datetime.now(), source="reddit", text="BTC pumping!"),
    SentimentData(timestamp=datetime.now(), source="news", text="Crypto crashing"),
]

# Batch analyze
analyzed = analyzer.batch_analyze(data)
for item in analyzed:
    print(f"{item.source}: {item.sentiment_score:.3f} | {item.keywords}")
```

### Aggregation and Signal Generation

```python
# Aggregate sentiment
agg_series = analyzer.aggregate_sentiment(analyzed, window=5, method='mean')
latest_sentiment = agg_series.iloc[-1]

# Generate trading signal
thresholds = {'positive': 0.3, 'negative': -0.3}
signal = analyzer.generate_sentiment_signal(latest_sentiment, thresholds)

print(f"Latest sentiment: {latest_sentiment:.3f}")
print(f"Signal: {signal}")
```

### Trading Strategy Integration

```python
def generate_trading_signal(sentiment_data: List[SentimentData]) -> str:
    """Generate trading signal based on sentiment."""
    analyzer = SentimentAnalyzer()
    
    # Analyze sentiment
    analyzed = analyzer.batch_analyze(sentiment_data)
    
    # Aggregate over recent window
    agg_series = analyzer.aggregate_sentiment(analyzed, window=10, method='mean')
    latest_sentiment = agg_series.iloc[-1]
    
    # Generate signal
    thresholds = {'positive': 0.2, 'negative': -0.2}
    sentiment_signal = analyzer.generate_sentiment_signal(latest_sentiment, thresholds)
    
    # Map to trading action
    if sentiment_signal == 'positive':
        return 'BUY'
    elif sentiment_signal == 'negative':
        return 'SELL'
    else:
        return 'HOLD'
```

## Configuration

### Thresholds

Configure sentiment signal thresholds based on your strategy:

```python
# Conservative thresholds
conservative = {'positive': 0.5, 'negative': -0.5}

# Aggressive thresholds
aggressive = {'positive': 0.1, 'negative': -0.1}

# Balanced thresholds
balanced = {'positive': 0.3, 'negative': -0.3}
```

### Aggregation Windows

Choose appropriate window sizes for your trading frequency:

```python
# High-frequency trading (short windows)
short_window = 5  # 5 data points

# Daily trading (longer windows)
long_window = 50  # 50 data points

# Research/backtesting
research_window = 100  # 100 data points
```

## Integration with Data Collector

The sentiment module is fully integrated with the data collector:

```python
from src.data_collector import SentimentDataCollector

# The data collector automatically uses the new sentiment module
collector = SentimentDataCollector()
reddit_data = collector.get_reddit_sentiment()
news_data = collector.get_news_sentiment()

# All data comes with sentiment scores and keywords
for item in reddit_data:
    print(f"Score: {item.sentiment_score}, Keywords: {item.keywords}")
```

## Extensibility

### Adding New Sentiment Models

The module is designed for easy model swapping:

```python
# For VADER integration
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class VADERSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self):
        self.model = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text: str) -> float:
        scores = self.model.polarity_scores(text)
        return scores['compound']  # VADER compound score
```

### Custom Keyword Lists

Extend the crypto keywords for your specific needs:

```python
analyzer = SentimentAnalyzer()
analyzer.crypto_keywords.extend([
    'avalanche', 'avax', 'polygon', 'matic', 'cardano', 'ada'
])
```

## Performance Considerations

1. **Batch Processing:** Use `batch_analyze()` for large datasets
2. **Caching:** Consider caching sentiment scores for repeated text
3. **Async Processing:** For high-frequency applications, consider async processing
4. **Memory Management:** Large datasets may require streaming processing

## Testing

Run the comprehensive demo:

```bash
PYTHONPATH=. python examples/sentiment_demo.py
```

Run tests:

```bash
PYTHONPATH=. python -m pytest tests/test_data_collector.py::test_sentiment_analyzer -v
```

## Best Practices

1. **Logging:** All critical operations are logged for transparency
2. **Error Handling:** Graceful degradation when sentiment analysis fails
3. **Validation:** Data validation ensures quality inputs
4. **Modularity:** Easy to extend and customize for specific needs
5. **Documentation:** Comprehensive docstrings for all public methods

## Future Enhancements

1. **Transformer Models:** Integration with FinBERT, BERT, or GPT models
2. **Multi-language Support:** Support for non-English sentiment analysis
3. **Real-time Processing:** Async/streaming capabilities for live trading
4. **Advanced Features:** Sentiment intensity, emotion classification, topic modeling
5. **Model Comparison:** A/B testing framework for different sentiment models

## Summary

The sentiment analysis module provides a robust, research-ready foundation for sentiment-based trading strategies. It combines text preprocessing, sentiment scoring, aggregation, and signal generation in a modular, extensible package that integrates seamlessly with the existing trading bot architecture. 