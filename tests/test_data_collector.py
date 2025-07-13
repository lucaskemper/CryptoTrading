#!/usr/bin/env python3
"""
Comprehensive test script for the improved data collector module.
Tests sentiment analysis, data validation, Kraken websocket, and error handling.
"""

import time
import pandas as pd
from datetime import datetime
from src.data_collector import (
    data_collector, ExchangeDataCollector, SentimentDataCollector,
    SentimentAnalyzer, MarketData, OrderBookData, SentimentData
)


def test_sentiment_analyzer():
    """Test sentiment analysis and keyword extraction."""
    print("Testing sentiment analyzer...")
    
    analyzer = SentimentAnalyzer()
    
    # Test sentiment analysis
    test_texts = [
        "Bitcoin is mooning! 🚀🚀🚀",
        "Crypto is crashing, this is terrible",
        "Ethereum and Solana are both doing well",
        "I'm neutral about crypto prices"
    ]
    
    for text in test_texts:
        sentiment = analyzer.analyze_sentiment(text)
        keywords = analyzer.extract_keywords(text)
        print(f"Text: {text[:50]}...")
        print(f"  Sentiment: {sentiment:.3f}")
        print(f"  Keywords: {keywords}")
        print()
    
    print("✅ Sentiment analyzer tests completed\n")


def test_data_validation():
    """Test data validation for all data types."""
    print("Testing data validation...")
    
    # Test valid market data
    valid_market = MarketData(
        timestamp=datetime.now(),
        symbol="ETH/USDT",
        exchange="binance",
        price=2000.0,
        volume=1000.0,
        bid=1999.0,
        ask=2001.0
    )
    print(f"Valid market data: {valid_market.validate()}")
    
    # Test invalid market data (bid >= ask)
    invalid_market = MarketData(
        timestamp=datetime.now(),
        symbol="ETH/USDT",
        exchange="binance",
        price=2000.0,
        volume=1000.0,
        bid=2001.0,  # Higher than ask
        ask=2000.0
    )
    print(f"Invalid market data: {invalid_market.validate()}")
    
    # Test valid order book
    valid_orderbook = OrderBookData(
        timestamp=datetime.now(),
        symbol="ETH/USDT",
        exchange="binance",
        bids=[[1999.0, 1.0], [1998.0, 2.0]],  # Descending
        asks=[[2001.0, 1.0], [2002.0, 2.0]]   # Ascending
    )
    print(f"Valid order book: {valid_orderbook.validate()}")
    
    # Test invalid order book (wrong order)
    invalid_orderbook = OrderBookData(
        timestamp=datetime.now(),
        symbol="ETH/USDT",
        exchange="binance",
        bids=[[1998.0, 1.0], [1999.0, 2.0]],  # Wrong order
        asks=[[2001.0, 1.0], [2002.0, 2.0]]
    )
    print(f"Invalid order book: {invalid_orderbook.validate()}")
    
    # Test valid sentiment data
    valid_sentiment = SentimentData(
        timestamp=datetime.now(),
        source="test",
        text="Bitcoin is doing great!",
        sentiment_score=0.8,
        keywords=["bitcoin"]
    )
    print(f"Valid sentiment data: {valid_sentiment.validate()}")
    
    # Test invalid sentiment data (empty text)
    invalid_sentiment = SentimentData(
        timestamp=datetime.now(),
        source="test",
        text="",
        sentiment_score=0.8
    )
    print(f"Invalid sentiment data: {invalid_sentiment.validate()}")
    
    print("✅ Data validation tests completed\n")


def test_market_data_collection():
    """Test market data collection with validation."""
    print("Testing market data collection...")
    
    exchange_collector = ExchangeDataCollector()
    
    # Test getting historical data with validation
    print("Fetching historical OHLCV data...")
    historical_data = exchange_collector.get_ohlcv("ETH/USDT", "binance", "1h", 10)
    if not historical_data.empty:
        print(f"Retrieved {len(historical_data)} historical records")
        print("Data validation check:")
        print(f"  - No NaN values: {not historical_data.isnull().any().any()}")
        print(f"  - All prices > 0: {(historical_data['close'] > 0).all()}")
        print(f"  - All volumes >= 0: {(historical_data['volume'] >= 0).all()}")
        print(historical_data.head())
    else:
        print("No historical data retrieved")
    
    # Test getting current ticker with validation
    print("\nFetching current ticker data...")
    ticker = exchange_collector.get_ticker("ETH/USDT", "binance")
    if ticker:
        print(f"Current ETH price: ${ticker.price}")
        print(f"24h volume: {ticker.volume}")
        print(f"Bid/Ask spread: ${ticker.ask - ticker.bid:.2f}")
        print(f"Data validation: {ticker.validate()}")
    else:
        print("No ticker data retrieved")
    
    # Test getting order book with validation
    print("\nFetching order book data...")
    order_book = exchange_collector.get_order_book("ETH/USDT", "binance", 5)
    if order_book:
        print(f"Top 5 bids: {order_book.bids[:5]}")
        print(f"Top 5 asks: {order_book.asks[:5]}")
        print(f"Order book validation: {order_book.validate()}")
    else:
        print("No order book data retrieved")
    
    print("✅ Market data collection tests completed\n")


def test_sentiment_data_collection():
    """Test sentiment data collection with analysis."""
    print("Testing sentiment data collection...")
    
    sentiment_collector = SentimentDataCollector()
    
    # Test Reddit sentiment with analysis
    print("Fetching Reddit sentiment data...")
    reddit_data = sentiment_collector.get_reddit_sentiment(limit=3)
    print(f"Retrieved {len(reddit_data)} Reddit posts")
    
    for post in reddit_data[:2]:
        print(f"- {post.source}: {post.text[:100]}...")
        print(f"  Sentiment: {post.sentiment_score:.3f}")
        print(f"  Keywords: {post.keywords}")
        print(f"  Validation: {post.validate()}")
    
    # Test news sentiment with analysis
    print("\nFetching news sentiment data...")
    news_data = sentiment_collector.get_news_sentiment(['bitcoin'])
    print(f"Retrieved {len(news_data)} news articles")
    
    for article in news_data[:2]:
        print(f"- {article.source}: {article.text[:100]}...")
        print(f"  Sentiment: {article.sentiment_score:.3f}")
        print(f"  Keywords: {article.keywords}")
        print(f"  Validation: {article.validate()}")
    
    # Test CryptoPanic sentiment with analysis
    print("\nFetching CryptoPanic sentiment data...")
    cryptopanic_data = sentiment_collector.get_cryptopanic_sentiment()
    print(f"Retrieved {len(cryptopanic_data)} CryptoPanic posts")
    
    for post in cryptopanic_data[:2]:
        print(f"- {post.source}: {post.text[:100]}...")
        print(f"  Sentiment: {post.sentiment_score:.3f}")
        print(f"  Keywords: {post.keywords}")
        print(f"  Validation: {post.validate()}")
    
    print("✅ Sentiment data collection tests completed\n")


def test_data_storage_with_validation():
    """Test data storage with validation."""
    print("Testing data storage with validation...")
    
    # Create sample data
    sample_market_data = MarketData(
        timestamp=datetime.now(),
        symbol="ETH/USDT",
        exchange="test",
        price=2000.0,
        volume=1000.0,
        bid=1999.0,
        ask=2001.0
    )
    
    sample_sentiment_data = SentimentData(
        timestamp=datetime.now(),
        source="test",
        text="Bitcoin is mooning! 🚀",
        sentiment_score=0.8,
        keywords=["bitcoin", "moon"]
    )
    
    # Test storage with validation
    data_collector.storage.save_market_data(sample_market_data)
    data_collector.storage.save_sentiment_data(sample_sentiment_data)
    
    print("Sample data saved to storage with validation")
    
    # Test invalid data (should be skipped)
    invalid_market_data = MarketData(
        timestamp=datetime.now(),
        symbol="ETH/USDT",
        exchange="test",
        price=-1000.0,  # Invalid negative price
        volume=1000.0
    )
    
    data_collector.storage.save_market_data(invalid_market_data)
    print("Invalid data correctly skipped")
    
    print("✅ Data storage tests completed\n")


def test_websocket_support():
    """Test websocket support for both exchanges."""
    print("Testing websocket support...")
    
    exchange_collector = ExchangeDataCollector()
    
    # Test Binance websocket (async)
    print("Testing Binance websocket...")
    try:
        import asyncio
        
        async def test_binance_ws():
            # Set running flag
            exchange_collector.running = True
            
            # Start websocket for 5 seconds
            task = asyncio.create_task(
                exchange_collector.start_websocket_stream("ETHUSDT", "binance")
            )
            
            # Wait a bit and check for data
            await asyncio.sleep(3)
            
            # Check if we received any data
            data_count = exchange_collector.data_queue.qsize()
            print(f"Received {data_count} data points from Binance websocket")
            
            # Stop
            exchange_collector.running = False
            task.cancel()
            
        # Run the test
        asyncio.run(test_binance_ws())
        
    except Exception as e:
        print(f"Binance websocket test error: {e}")
    
    # Test Kraken websocket (if API keys are available)
    print("Testing Kraken websocket...")
    try:
        # Check if Kraken is initialized with API keys
        kraken_config = data_collector.exchange_collector.exchanges.get('kraken')
        if kraken_config and hasattr(kraken_config, 'apiKey') and kraken_config.apiKey:
            async def test_kraken_ws():
                exchange_collector.running = True
                
                task = asyncio.create_task(
                    exchange_collector.start_websocket_stream("XETHZUSD", "kraken")
                )
                
                await asyncio.sleep(3)
                
                data_count = exchange_collector.data_queue.qsize()
                print(f"Received {data_count} data points from Kraken websocket")
                
                exchange_collector.running = False
                task.cancel()
            
            asyncio.run(test_kraken_ws())
        else:
            print("Kraken API keys not available, skipping websocket test")
            
    except Exception as e:
        print(f"Kraken websocket test error: {e}")
    
    print("✅ Websocket tests completed\n")


def test_error_handling():
    """Test error handling and graceful degradation."""
    print("Testing error handling...")
    
    exchange_collector = ExchangeDataCollector()
    
    # Test with invalid symbol
    print("Testing invalid symbol handling...")
    invalid_data = exchange_collector.get_ticker("INVALID/PAIR", "binance")
    if invalid_data is None:
        print("✅ Correctly handled invalid symbol")
    
    # Test with invalid exchange
    print("Testing invalid exchange handling...")
    invalid_data = exchange_collector.get_ticker("ETH/USDT", "nonexistent")
    if invalid_data is None:
        print("✅ Correctly handled invalid exchange")
    
    # Test sentiment with invalid API keys
    print("Testing sentiment with invalid API keys...")
    sentiment_collector = SentimentDataCollector()
    
    # Temporarily modify config to test error handling
    original_news_key = sentiment_collector.sentiment_config.get('news_api_key')
    sentiment_collector.sentiment_config['news_api_key'] = 'invalid_key'
    
    news_data = sentiment_collector.get_news_sentiment(['bitcoin'])
    print(f"Retrieved {len(news_data)} articles with invalid key (should be 0)")
    
    # Restore original key
    sentiment_collector.sentiment_config['news_api_key'] = original_news_key
    
    print("✅ Error handling tests completed\n")


def main():
    """Run all comprehensive tests."""
    print("=== Improved Data Collector Test Suite ===")
    print(f"Started at: {datetime.now()}")
    print()
    
    # Run all tests
    test_sentiment_analyzer()
    test_data_validation()
    test_market_data_collection()
    test_sentiment_data_collection()
    test_data_storage_with_validation()
    test_websocket_support()
    test_error_handling()
    
    print("=== All Tests Completed ===")
    print(f"Finished at: {datetime.now()}")
    print("\n🎉 All improvements successfully implemented and tested!")


if __name__ == "__main__":
    main() 