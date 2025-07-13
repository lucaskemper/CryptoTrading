"""
Data Collector Module

Collects market data from multiple exchanges and sentiment data from various sources.
Supports real-time websocket connections and periodic REST API polling.
"""

import asyncio
import json
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
import ccxt
import requests
from dataclasses import dataclass
import websockets
import threading
from queue import Queue
import os
import re
from textblob import TextBlob
import numpy as np

from src.utils.logger import logger
from src.utils.config_loader import config


@dataclass
class MarketData:
    """Market data structure."""
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
    
    def validate(self) -> bool:
        """Validate market data for reasonable values."""
        if self.price <= 0 or self.volume < 0:
            return False
        if self.bid and self.ask and self.bid >= self.ask:
            return False
        if self.high and self.low and self.high < self.low:
            return False
        return True


@dataclass
class OrderBookData:
    """Order book data structure."""
    timestamp: datetime
    symbol: str
    exchange: str
    bids: List[List[float]]  # [price, volume]
    asks: List[List[float]]  # [price, volume]
    
    def validate(self) -> bool:
        """Validate order book data."""
        if not self.bids or not self.asks:
            return False
        # Check that bids are sorted descending and asks ascending
        bid_prices = [bid[0] for bid in self.bids]
        ask_prices = [ask[0] for ask in self.asks]
        if bid_prices != sorted(bid_prices, reverse=True):
            return False
        if ask_prices != sorted(ask_prices):
            return False
        return True


@dataclass
class SentimentData:
    """Sentiment data structure."""
    timestamp: datetime
    source: str
    text: str
    sentiment_score: Optional[float] = None
    keywords: List[str] = None
    url: Optional[str] = None
    
    def validate(self) -> bool:
        """Validate sentiment data."""
        if not self.text or len(self.text.strip()) == 0:
            return False
        if self.sentiment_score is not None and not (-1 <= self.sentiment_score <= 1):
            return False
        return True


class SentimentAnalyzer:
    """Handles sentiment analysis and keyword extraction."""
    
    def __init__(self):
        self.crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol', 'crypto', 
            'blockchain', 'defi', 'nft', 'altcoin', 'moon', 'pump', 'dump',
            'hodl', 'fomo', 'fud', 'bull', 'bear', 'mooning', 'crashing'
        ]
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment using TextBlob."""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract crypto-related keywords from text."""
        try:
            # Convert to lowercase and find keywords
            text_lower = text.lower()
            found_keywords = []
            
            for keyword in self.crypto_keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)
            
            # Also extract hashtags and mentions
            hashtags = re.findall(r'#(\w+)', text)
            mentions = re.findall(r'@(\w+)', text)
            
            found_keywords.extend(hashtags)
            found_keywords.extend(mentions)
            
            return list(set(found_keywords))  # Remove duplicates
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []


class ExchangeDataCollector:
    """Handles market data collection from cryptocurrency exchanges."""
    
    def __init__(self):
        self.exchanges = {}
        self.websocket_connections = {}
        self.data_queue = Queue()
        self.running = False
        self._setup_exchanges()
    
    def _setup_exchanges(self):
        """Initialize exchange connections."""
        exchanges_to_setup = ['binance', 'kraken']
        
        for exchange_name in exchanges_to_setup:
            try:
                exchange_config = config.get_exchange_config(exchange_name)
                if exchange_config.get('apiKey'):
                    exchange_class = getattr(ccxt, exchange_name)
                    self.exchanges[exchange_name] = exchange_class({
                        'apiKey': exchange_config['apiKey'],
                        'secret': exchange_config['secret'],
                        'sandbox': exchange_config['sandbox'],
                        'enableRateLimit': True
                    })
                    logger.info(f"Initialized {exchange_name} exchange")
                else:
                    logger.warning(f"No API credentials for {exchange_name}, using public endpoints only")
                    exchange_class = getattr(ccxt, exchange_name)
                    self.exchanges[exchange_name] = exchange_class({
                        'enableRateLimit': True
                    })
            except Exception as e:
                logger.error(f"Failed to initialize {exchange_name}: {e}")
    
    def get_ohlcv(self, symbol: str, exchange: str, timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
        """Get OHLCV data from exchange."""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not initialized")
            
            exchange_obj = self.exchanges[exchange]
            ohlcv = exchange_obj.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['exchange'] = exchange
            df['symbol'] = symbol
            
            # Validate data
            df = df.dropna()
            df = df[df['close'] > 0]
            df = df[df['volume'] >= 0]
            
            logger.debug(f"Retrieved {len(df)} OHLCV records for {symbol} from {exchange}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} from {exchange}: {e}")
            return pd.DataFrame()
    
    def get_order_book(self, symbol: str, exchange: str, limit: int = 20) -> Optional[OrderBookData]:
        """Get order book data from exchange."""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not initialized")
            
            exchange_obj = self.exchanges[exchange]
            order_book = exchange_obj.fetch_order_book(symbol, limit)
            
            order_book_data = OrderBookData(
                timestamp=datetime.now(),
                symbol=symbol,
                exchange=exchange,
                bids=order_book['bids'],
                asks=order_book['asks']
            )
            
            # Validate order book data
            if not order_book_data.validate():
                logger.warning(f"Invalid order book data for {symbol} from {exchange}")
                return None
            
            return order_book_data
            
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol} from {exchange}: {e}")
            return None
    
    def get_ticker(self, symbol: str, exchange: str) -> Optional[MarketData]:
        """Get current ticker data from exchange."""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not initialized")
            
            exchange_obj = self.exchanges[exchange]
            ticker = exchange_obj.fetch_ticker(symbol)
            
            # Validate ticker data before processing
            if not ticker or not isinstance(ticker, dict):
                logger.warning(f"Invalid ticker response for {symbol} from {exchange}")
                return None
            
            # Check for required fields and handle None values
            required_fields = ['last', 'baseVolume']
            for field in required_fields:
                if field not in ticker or ticker[field] is None:
                    logger.warning(f"Missing or None field '{field}' in ticker for {symbol} from {exchange}")
                    return None
            
            # Handle timestamp conversion safely - use current time if timestamp is missing
            if 'timestamp' in ticker and ticker['timestamp'] is not None:
                try:
                    timestamp = datetime.fromtimestamp(ticker['timestamp'] / 1000)
                except (TypeError, ValueError, OSError) as e:
                    logger.warning(f"Invalid timestamp in ticker for {symbol} from {exchange}: {e}")
                    timestamp = datetime.now()
            else:
                # Use current time if timestamp is missing (common with Kraken)
                timestamp = datetime.now()
                logger.debug(f"Using current time for {symbol} from {exchange} (no timestamp in response)")
            
            market_data = MarketData(
                timestamp=timestamp,
                symbol=symbol,
                exchange=exchange,
                price=ticker['last'],
                volume=ticker['baseVolume'],
                bid=ticker.get('bid'),
                ask=ticker.get('ask'),
                high=ticker.get('high'),
                low=ticker.get('low'),
                open=ticker.get('open'),
                close=ticker.get('close')
            )
            
            # Validate market data
            if not market_data.validate():
                logger.warning(f"Invalid market data for {symbol} from {exchange}")
                return None
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol} from {exchange}: {e}")
            return None
    
    async def start_websocket_stream(self, symbol: str, exchange: str):
        """Start websocket stream for real-time data."""
        if exchange == 'binance':
            await self._binance_websocket(symbol)
        elif exchange == 'kraken':
            await self._kraken_websocket(symbol)
    
    async def _binance_websocket(self, symbol: str):
        """Binance websocket implementation."""
        uri = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker"
        
        try:
            async with websockets.connect(uri) as websocket:
                logger.info(f"Connected to Binance websocket for {symbol}")
                
                while self.running:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        try:
                            # Safely extract data with error handling
                            market_data = MarketData(
                                timestamp=datetime.now(),
                                symbol=symbol,
                                exchange='binance',
                                price=float(data.get('c', 0)),
                                volume=float(data.get('v', 0)),
                                bid=float(data.get('b', 0)) if data.get('b') else None,
                                ask=float(data.get('a', 0)) if data.get('a') else None,
                                high=float(data.get('h', 0)) if data.get('h') else None,
                                low=float(data.get('l', 0)) if data.get('l') else None,
                                open=float(data.get('o', 0)) if data.get('o') else None,
                                close=float(data.get('c', 0))
                            )
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error processing Binance websocket data for {symbol}: {e}")
                            continue
                        
                        # Validate before adding to queue
                        if market_data.validate():
                            self.data_queue.put(market_data)
                        
                    except Exception as e:
                        logger.error(f"Error processing Binance websocket message: {e}")
                        
        except Exception as e:
            logger.error(f"Binance websocket error: {e}")
    
    async def _kraken_websocket(self, symbol: str):
        """Kraken websocket implementation."""
        try:
            # Kraken uses different symbol format (XBTUSD instead of BTC/USD)
            kraken_symbol = symbol.replace('/', '').replace('BTC', 'XBT').replace('ETH', 'XETH')
            
            # Kraken websocket endpoint
            uri = "wss://ws.kraken.com"
            
            async with websockets.connect(uri) as websocket:
                logger.info(f"Connected to Kraken websocket for {symbol}")
                
                # Subscribe to ticker
                subscribe_message = {
                    "event": "subscribe",
                    "pair": [kraken_symbol],
                    "subscription": {
                        "name": "ticker"
                    }
                }
                await websocket.send(json.dumps(subscribe_message))
                
                while self.running:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        # Handle subscription confirmation
                        if data.get('event') == 'subscriptionStatus':
                            logger.info(f"Kraken subscription confirmed: {data}")
                            continue
                        
                        # Handle ticker data
                        if isinstance(data, list) and len(data) > 1:
                            ticker_data = data[1]
                            
                            if isinstance(ticker_data, dict) and 'c' in ticker_data:
                                try:
                                    # Safely extract price data with bounds checking
                                    price_array = ticker_data.get('c', [])
                                    volume_array = ticker_data.get('v', [])
                                    high_array = ticker_data.get('h', [])
                                    low_array = ticker_data.get('l', [])
                                    
                                    # Check if arrays have sufficient elements
                                    if (len(price_array) > 0 and len(volume_array) > 1 and 
                                        len(high_array) > 1 and len(low_array) > 1):
                                        
                                        price = float(price_array[0])  # Current price
                                        volume = float(volume_array[1])  # 24h volume
                                        high = float(high_array[1])   # 24h high
                                        low = float(low_array[1])    # 24h low
                                        
                                        market_data = MarketData(
                                            timestamp=datetime.now(),
                                            symbol=symbol,
                                            exchange='kraken',
                                            price=price,
                                            volume=volume,
                                            high=high,
                                            low=low
                                        )
                                        
                                        # Validate before adding to queue
                                        if market_data.validate():
                                            self.data_queue.put(market_data)
                                    else:
                                        logger.warning(f"Insufficient data in Kraken websocket ticker for {symbol}")
                                        
                                except (ValueError, TypeError, IndexError) as e:
                                    logger.warning(f"Error processing Kraken websocket data for {symbol}: {e}")
                        
                    except Exception as e:
                        logger.error(f"Error processing Kraken websocket message: {e}")
                        
        except Exception as e:
            logger.error(f"Kraken websocket error: {e}")


class SentimentDataCollector:
    """Handles sentiment data collection from various sources."""
    
    def __init__(self):
        self.sentiment_config = config.get_sentiment_config()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'CryptoBot/1.0'})
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def get_reddit_sentiment(self, subreddits: List[str] = None, limit: int = 25) -> List[SentimentData]:
        """Collect sentiment data from Reddit."""
        if not subreddits:
            subreddits = ['CryptoCurrency', 'ethtrader', 'solana']
        
        sentiment_data = []
        
        for subreddit in subreddits:
            try:
                url = f"https://www.reddit.com/r/{subreddit}/hot.json"
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                posts = data['data']['children']
                
                for post in posts[:limit]:
                    post_data = post['data']
                    text = post_data['title'] + " " + (post_data.get('selftext', '')[:200])
                    
                    # Analyze sentiment and extract keywords
                    sentiment_score = self.sentiment_analyzer.analyze_sentiment(text)
                    keywords = self.sentiment_analyzer.extract_keywords(text)
                    
                    sentiment_data.append(SentimentData(
                        timestamp=datetime.fromtimestamp(post_data['created_utc']),
                        source=f"reddit/r/{subreddit}",
                        text=text,
                        sentiment_score=sentiment_score,
                        keywords=keywords,
                        url=f"https://reddit.com{post_data['permalink']}"
                    ))
                
                logger.debug(f"Collected {len(posts)} posts from r/{subreddit}")
                
            except Exception as e:
                logger.error(f"Error collecting Reddit data from r/{subreddit}: {e}")
        
        return sentiment_data
    
    def get_news_sentiment(self, keywords: List[str] = None) -> List[SentimentData]:
        """Collect sentiment data from news APIs."""
        if not keywords:
            keywords = ['bitcoin', 'ethereum', 'solana', 'crypto', 'blockchain']
        
        sentiment_data = []
        
        # NewsAPI implementation
        news_api_key = self.sentiment_config.get('news_api_key')
        if news_api_key:
            try:
                for keyword in keywords:
                    url = "https://newsapi.org/v2/everything"
                    params = {
                        'q': keyword,
                        'apiKey': news_api_key,
                        'language': 'en',
                        'sortBy': 'publishedAt',
                        'pageSize': 20
                    }
                    
                    response = self.session.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    
                    data = response.json()
                    articles = data.get('articles', [])
                    
                    for article in articles:
                        text = article['title'] + " " + (article.get('description', '')[:200])
                        
                        # Analyze sentiment and extract keywords
                        sentiment_score = self.sentiment_analyzer.analyze_sentiment(text)
                        keywords = self.sentiment_analyzer.extract_keywords(text)
                        
                        sentiment_data.append(SentimentData(
                            timestamp=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                            source="newsapi",
                            text=text,
                            sentiment_score=sentiment_score,
                            keywords=keywords,
                            url=article.get('url')
                        ))
                    
                    logger.debug(f"Collected {len(articles)} articles for keyword '{keyword}'")
                    
            except Exception as e:
                logger.error(f"Error collecting news data: {e}")
        
        return sentiment_data
    
    def get_cryptopanic_sentiment(self) -> List[SentimentData]:
        """Collect sentiment data from CryptoPanic API."""
        cryptopanic_key = self.sentiment_config.get('cryptopanic_api_key')
        sentiment_data = []
        
        if cryptopanic_key:
            try:
                url = "https://cryptopanic.com/api/v1/posts/"
                params = {
                    'auth_token': cryptopanic_key,
                    'filter': 'hot',
                    'currencies': 'BTC,ETH,SOL'
                }
                
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                posts = data.get('results', [])
                
                for post in posts:
                    text = post['title'] + " " + (post.get('metadata', {}).get('description', '')[:200])
                    
                    # Analyze sentiment and extract keywords
                    sentiment_score = self.sentiment_analyzer.analyze_sentiment(text)
                    keywords = self.sentiment_analyzer.extract_keywords(text)
                    
                    sentiment_data.append(SentimentData(
                        timestamp=datetime.fromisoformat(post['published_at']),
                        source="cryptopanic",
                        text=text,
                        sentiment_score=sentiment_score,
                        keywords=keywords,
                        url=post.get('url')
                    ))
                
                logger.debug(f"Collected {len(posts)} posts from CryptoPanic")
                
            except Exception as e:
                logger.error(f"Error collecting CryptoPanic data: {e}")
        
        return sentiment_data


class DataStorage:
    """Handles data storage to CSV, SQLite, and PostgreSQL."""
    
    def __init__(self):
        self.db_config = config.get_database_config()
        self._setup_storage()
    
    def _setup_storage(self):
        """Setup storage directories and database."""
        # Create data directories
        os.makedirs(self.db_config['csv_dir'], exist_ok=True)
        os.makedirs(os.path.dirname(self.db_config['sqlite_path']), exist_ok=True)
        
        # Initialize SQLite database
        self._init_sqlite()
    
    def _init_sqlite(self):
        """Initialize SQLite database with tables."""
        try:
            conn = sqlite3.connect(self.db_config['sqlite_path'])
            cursor = conn.cursor()
            
            # Market data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    exchange TEXT,
                    price REAL,
                    volume REAL,
                    bid REAL,
                    ask REAL,
                    high REAL,
                    low REAL,
                    open REAL,
                    close REAL
                )
            ''')
            
            # Order book table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS order_book (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    exchange TEXT,
                    bids TEXT,
                    asks TEXT
                )
            ''')
            
            # Sentiment data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    source TEXT,
                    text TEXT,
                    sentiment_score REAL,
                    keywords TEXT,
                    url TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("SQLite database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {e}")
    
    def save_market_data(self, market_data: MarketData):
        """Save market data to storage."""
        try:
            # Validate data before saving
            if not market_data.validate():
                logger.warning(f"Invalid market data, skipping save: {market_data}")
                return
            
            # Save to SQLite
            conn = sqlite3.connect(self.db_config['sqlite_path'])
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO market_data 
                (timestamp, symbol, exchange, price, volume, bid, ask, high, low, open, close)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                market_data.timestamp,
                market_data.symbol,
                market_data.exchange,
                market_data.price,
                market_data.volume,
                market_data.bid,
                market_data.ask,
                market_data.high,
                market_data.low,
                market_data.open,
                market_data.close
            ))
            
            conn.commit()
            conn.close()
            
            # Save to CSV (append mode)
            csv_path = f"{self.db_config['csv_dir']}/market_data_{market_data.symbol}_{market_data.exchange}.csv"
            df = pd.DataFrame([{
                'timestamp': market_data.timestamp,
                'symbol': market_data.symbol,
                'exchange': market_data.exchange,
                'price': market_data.price,
                'volume': market_data.volume,
                'bid': market_data.bid,
                'ask': market_data.ask,
                'high': market_data.high,
                'low': market_data.low,
                'open': market_data.open,
                'close': market_data.close
            }])
            # Ensure directory exists
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
            
        except Exception as e:
            logger.error(f"Error saving market data: {e}")
    
    def save_order_book(self, order_book: OrderBookData):
        """Save order book data to storage."""
        try:
            # Validate data before saving
            if not order_book.validate():
                logger.warning(f"Invalid order book data, skipping save: {order_book}")
                return
            
            # Save to SQLite
            conn = sqlite3.connect(self.db_config['sqlite_path'])
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO order_book (timestamp, symbol, exchange, bids, asks)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                order_book.timestamp,
                order_book.symbol,
                order_book.exchange,
                json.dumps(order_book.bids),
                json.dumps(order_book.asks)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving order book data: {e}")
    
    def save_sentiment_data(self, sentiment_data: SentimentData):
        """Save sentiment data to storage."""
        try:
            # Validate data before saving
            if not sentiment_data.validate():
                logger.warning(f"Invalid sentiment data, skipping save: {sentiment_data}")
                return
            
            # Save to SQLite
            conn = sqlite3.connect(self.db_config['sqlite_path'])
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sentiment_data (timestamp, source, text, sentiment_score, keywords, url)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                sentiment_data.timestamp,
                sentiment_data.source,
                sentiment_data.text,
                sentiment_data.sentiment_score,
                json.dumps(sentiment_data.keywords) if sentiment_data.keywords else None,
                sentiment_data.url
            ))
            
            conn.commit()
            conn.close()
            
            # Save to CSV
            csv_path = f"{self.db_config['csv_dir']}/sentiment_data_{sentiment_data.source}.csv"
            df = pd.DataFrame([{
                'timestamp': sentiment_data.timestamp,
                'source': sentiment_data.source,
                'text': sentiment_data.text,
                'sentiment_score': sentiment_data.sentiment_score,
                'keywords': json.dumps(sentiment_data.keywords) if sentiment_data.keywords else None,
                'url': sentiment_data.url
            }])
            # Ensure directory exists
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
            
        except Exception as e:
            logger.error(f"Error saving sentiment data: {e}")


class DataCollector:
    """Main data collector class that orchestrates all data collection."""
    
    def __init__(self):
        self.exchange_collector = ExchangeDataCollector()
        self.sentiment_collector = SentimentDataCollector()
        self.storage = DataStorage()
        self.running = False
        self.collection_thread = None
    
    def start(self):
        """Start the data collection process."""
        self.running = True
        self.collection_thread = threading.Thread(target=self._run_collection_loop)
        self.collection_thread.start()
        logger.info("Data collector started")
    
    def stop(self):
        """Stop the data collection process."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join()
        logger.info("Data collector stopped")
    
    def _run_collection_loop(self):
        """Main collection loop."""
        while self.running:
            try:
                # Collect market data
                self._collect_market_data()
                
                # Collect sentiment data (less frequent)
                if datetime.now().minute % 5 == 0:  # Every 5 minutes
                    self._collect_sentiment_data()
                
                time.sleep(30)  # 30-second intervals
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _collect_market_data(self):
        """Collect market data from all exchanges."""
        symbols = ['ETH/USDT', 'SOL/USDT', 'BTC/USDT']
        
        for exchange_name in self.exchange_collector.exchanges:
            for symbol in symbols:
                try:
                    # Get ticker data
                    ticker = self.exchange_collector.get_ticker(symbol, exchange_name)
                    if ticker:
                        self.storage.save_market_data(ticker)
                    
                    # Get order book data
                    order_book = self.exchange_collector.get_order_book(symbol, exchange_name)
                    if order_book:
                        self.storage.save_order_book(order_book)
                    
                except Exception as e:
                    logger.error(f"Error collecting data for {symbol} from {exchange_name}: {e}")
    
    def _collect_sentiment_data(self):
        """Collect sentiment data from all sources."""
        try:
            # Collect from Reddit
            reddit_data = self.sentiment_collector.get_reddit_sentiment()
            for data in reddit_data:
                self.storage.save_sentiment_data(data)
            
            # Collect from news APIs
            news_data = self.sentiment_collector.get_news_sentiment()
            for data in news_data:
                self.storage.save_sentiment_data(data)
            
            # Collect from CryptoPanic
            cryptopanic_data = self.sentiment_collector.get_cryptopanic_sentiment()
            for data in cryptopanic_data:
                self.storage.save_sentiment_data(data)
                
        except Exception as e:
            logger.error(f"Error collecting sentiment data: {e}")
    
    def get_historical_data(self, symbol: str, exchange: str, timeframe: str = '1h', limit: int = 1000) -> pd.DataFrame:
        """Get historical market data."""
        return self.exchange_collector.get_ohlcv(symbol, exchange, timeframe, limit)
    
    def get_latest_market_data(self, symbol: str, exchange: str) -> Optional[MarketData]:
        """Get latest market data for a symbol."""
        return self.exchange_collector.get_ticker(symbol, exchange)


# Global data collector instance
data_collector = DataCollector()
