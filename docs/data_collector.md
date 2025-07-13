# Data Collector Documentation

## Overview

The `data_collector.py` module provides comprehensive data collection capabilities for the crypto trading bot, including:

- **Market Data**: Real-time and historical price, volume, and order book data from multiple exchanges
- **Sentiment Data**: News, Reddit, and social media sentiment analysis
- **Data Storage**: SQLite database and CSV file storage with automatic organization
- **Error Handling**: Robust error handling with exponential backoff and logging
- **Configuration**: Environment-based configuration with secure API key management

## Features

### Market Data Collection

#### Supported Exchanges
- **Binance**: Primary exchange with deep liquidity for ETH and SOL
- **Kraken**: Secondary exchange with excellent API support and regulatory compliance
- **Solana DEXes**: Planned integration for native Solana ecosystem exposure

#### Data Types Collected
- **OHLCV Data**: Open, High, Low, Close, Volume data
- **Ticker Data**: Real-time price, bid/ask, volume information
- **Order Book Data**: Bid/ask depth for market microstructure analysis
- **Trade History**: Executed trades with timestamps and amounts

### Sentiment Data Collection

#### Sources
- **Reddit**: r/CryptoCurrency, r/ethtrader, r/solana subreddits
- **News APIs**: NewsAPI, CryptoPanic for crypto-related news
- **Social Media**: Twitter/X integration (planned)

#### Data Structure
- Text content with timestamps
- Source attribution
- Optional sentiment scoring
- Keywords and metadata

### Data Storage

#### Storage Options
- **SQLite Database**: Local database for querying and analysis
- **CSV Files**: Human-readable files for inspection and backup
- **Real-time Streaming**: In-memory data for low-latency trading
- **PostgreSQL**: Optional cloud database for production scaling

#### Data Organization
```
data/
├── market_data_ETH_USDT_binance.csv
├── market_data_SOL_USDT_binance.csv
├── sentiment_data_reddit.csv
├── sentiment_data_newsapi.csv
└── trading_bot.db (SQLite)
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file with your API keys:

```bash
# Exchange API Keys
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here
BINANCE_SANDBOX=false

KRAKEN_API_KEY=your_kraken_api_key_here
KRAKEN_API_SECRET=your_kraken_api_secret_here
KRAKEN_SANDBOX=false

# Sentiment API Keys
NEWS_API_KEY=your_newsapi_key_here
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
CRYPTOPANIC_API_KEY=your_cryptopanic_api_key_here

# Database Configuration
SQLITE_PATH=data/trading_bot.db
CSV_DIR=data/
```

### 3. Configuration File

The `config/config.yaml` file contains default settings:

```yaml
data_collection:
  market_data_interval: 30  # seconds
  sentiment_data_interval: 300  # seconds
  symbols:
    - ETH/USDT
    - SOL/USDT
    - BTC/USDT
```

## Usage

### Basic Usage

```python
from src.data_collector import data_collector

# Start data collection
data_collector.start()

# Get historical data
historical_data = data_collector.get_historical_data(
    symbol="ETH/USDT", 
    exchange="binance", 
    timeframe="1h", 
    limit=1000
)

# Get latest market data
latest_data = data_collector.get_latest_market_data("ETH/USDT", "binance")

# Stop data collection
data_collector.stop()
```

### Advanced Usage

```python
from src.data_collector import ExchangeDataCollector, SentimentDataCollector

# Initialize individual collectors
exchange_collector = ExchangeDataCollector()
sentiment_collector = SentimentDataCollector()

# Get OHLCV data
ohlcv_data = exchange_collector.get_ohlcv("ETH/USDT", "binance", "1m", 100)

# Get order book
order_book = exchange_collector.get_order_book("ETH/USDT", "binance")

# Get sentiment data
reddit_sentiment = sentiment_collector.get_reddit_sentiment()
news_sentiment = sentiment_collector.get_news_sentiment()
```

### WebSocket Real-time Data

```python
import asyncio

async def real_time_data():
    exchange_collector = ExchangeDataCollector()
    
    # Start websocket stream
    await exchange_collector.start_websocket_stream("ETHUSDT", "binance")
    
    # Process data from queue
    while True:
        if not exchange_collector.data_queue.empty():
            market_data = exchange_collector.data_queue.get()
            print(f"Real-time price: {market_data.price}")

# Run websocket
asyncio.run(real_time_data())
```

## Data Structures

### MarketData
```python
@dataclass
class MarketData:
    timestamp: datetime
    symbol: str
    exchange: str
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    close: Optional[float] = None
```

### OrderBookData
```python
@dataclass
class OrderBookData:
    timestamp: datetime
    symbol: str
    exchange: str
    bids: List[List[float]]  # [price, volume]
    asks: List[List[float]]  # [price, volume]
```

### SentimentData
```python
@dataclass
class SentimentData:
    timestamp: datetime
    source: str
    text: str
    sentiment_score: Optional[float] = None
    keywords: List[str] = None
    url: Optional[str] = None
```

## Error Handling

The data collector includes comprehensive error handling:

- **Automatic Retries**: Exponential backoff for network failures
- **Rate Limiting**: Respects exchange API rate limits
- **Graceful Degradation**: Continues operation even if some sources fail
- **Detailed Logging**: All errors and events are logged for debugging

## Performance Considerations

### Data Collection Frequency
- **Market Data**: 30-second intervals for REST API calls
- **Sentiment Data**: 5-minute intervals to avoid API rate limits
- **WebSocket**: Real-time streaming for low-latency requirements

### Storage Optimization
- **CSV Files**: Append-only for efficient writing
- **SQLite**: Indexed queries for fast retrieval
- **Data Retention**: Configurable cleanup policies

### Memory Management
- **Queue-based Processing**: Prevents memory overflow
- **Batch Processing**: Efficient database operations
- **Streaming**: Real-time data without storage overhead

## Monitoring and Logging

### Log Files
- **Console Output**: Real-time status updates
- **File Logs**: Detailed logs in `logs/trading_bot_YYYYMMDD.log`
- **Error Tracking**: Comprehensive error logging with stack traces

### Metrics
- **Data Collection Rate**: Records per second/minute
- **API Response Times**: Performance monitoring
- **Error Rates**: Success/failure tracking
- **Storage Usage**: Database and file size monitoring

## Security

### API Key Management
- **Environment Variables**: Secure storage of sensitive credentials
- **No Hardcoding**: API keys never stored in code
- **Sandbox Support**: Safe testing with exchange sandbox environments

### Data Privacy
- **Local Storage**: Data stored locally by default
- **Encryption**: Optional database encryption
- **Access Control**: Configurable data access permissions

## Troubleshooting

### Common Issues

1. **API Rate Limits**
   - Solution: Implement exponential backoff
   - Monitor: Check log files for rate limit errors

2. **Network Connectivity**
   - Solution: Automatic retry with increasing delays
   - Monitor: Connection timeout logs

3. **Database Errors**
   - Solution: Check file permissions and disk space
   - Monitor: SQLite error logs

4. **Missing API Keys**
   - Solution: Verify environment variables are set
   - Monitor: Configuration loading logs

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
from src.utils.logger import logger
logger.logger.setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features
- **Solana DEX Integration**: Direct blockchain data collection
- **Advanced Sentiment Analysis**: ML-based sentiment scoring
- **Data Compression**: Efficient storage for large datasets
- **Cloud Integration**: AWS S3, Google Cloud Storage support
- **Real-time Alerts**: Price and sentiment alerting system

### Performance Optimizations
- **Parallel Processing**: Multi-threaded data collection
- **Caching**: Redis integration for high-frequency data
- **Data Streaming**: Apache Kafka integration
- **Machine Learning**: Automated feature engineering 