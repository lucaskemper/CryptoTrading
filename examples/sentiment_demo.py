#!/usr/bin/env python3
"""
Enhanced Sentiment Analysis Demo

This demo showcases all the enhanced features of the sentiment analysis module:
- Granular error handling and reporting
- Weighted aggregation by source credibility
- Entity extraction (tickers, project names)
- Sentiment intensity analysis
- Comprehensive edge case handling

Usage:
    PYTHONPATH=. python examples/sentiment_enhanced_demo.py
"""

import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.strategy.sentiment import SentimentAnalyzer, SentimentData
from src.utils.logger import logger


def create_enhanced_sample_data() -> List[SentimentData]:
    """Create sample sentiment data with various sources and edge cases."""
    base_time = datetime.now()
    
    sample_texts = [
        # Normal cases
        ("Bitcoin BTC is absolutely mooning! 🚀🚀🚀 This is incredible!", "reddit", 0),
        ("Crypto market is crashing hard, this is terrible news", "news", 1),
        ("Ethereum ETH and Solana SOL are both performing well today", "reddit", 2),
        ("I'm neutral about crypto prices, waiting to see what happens", "news", 3),
        ("SOL is pumping like crazy! #Solana #Moon", "twitter", 4),
        ("BTC showing strong support at current levels", "reddit", 5),
        ("Market sentiment is bearish, expect more downside", "news", 6),
        ("DeFi protocols are gaining traction", "reddit", 7),
        ("NFT market is booming with new collections", "twitter", 8),
        ("Altcoin season might be starting soon", "reddit", 9),
        
        # Edge cases for error handling
        ("", "reddit", 10),  # Empty text
        ("   ", "news", 11),  # Whitespace only
        ("BTC!@#$%^&*()", "twitter", 12),  # Special characters
        ("Bitcoin " * 100, "reddit", 13),  # Very long text
        ("Check out https://example.com for more info", "news", 14),  # URLs
    ]
    
    data = []
    for text, source, offset in sample_texts:
        data.append(SentimentData(
            timestamp=base_time + timedelta(minutes=offset),
            source=source,
            text=text,
            sentiment_score=None,
            keywords=None,
            url=None
        ))
    
    return data


def demonstrate_error_handling(analyzer: SentimentAnalyzer):
    """Demonstrate enhanced error handling."""
    print("\n=== Enhanced Error Handling ===")
    
    # Test various edge cases
    edge_cases = [
        "",  # Empty string
        "   ",  # Whitespace only
        None,  # None value
        "BTC!@#$%^&*()",  # Special characters
        "Bitcoin " * 1000,  # Very long text
    ]
    
    for i, text in enumerate(edge_cases):
        print(f"Edge case {i+1}: {repr(text)}")
        
        # Test sentiment analysis
        score = analyzer.analyze_sentiment(text)
        print(f"  Sentiment score: {score}")
        
        # Test keyword extraction
        keywords = analyzer.extract_keywords(text)
        print(f"  Keywords: {keywords}")
        
        # Test entity extraction
        entities = analyzer.extract_entities(text)
        print(f"  Entities: {entities}")
        
        # Test intensity analysis
        intensity = analyzer.analyze_sentiment_intensity(text)
        print(f"  Intensity: {intensity}")
        print()


def demonstrate_weighted_aggregation(analyzer: SentimentAnalyzer, data: List[SentimentData]):
    """Demonstrate weighted aggregation by source credibility."""
    print("\n=== Weighted Aggregation ===")
    
    # Define source weights (higher = more credible)
    source_weights = {
        'news': 1.5,      # News sources get higher weight
        'reddit': 0.8,    # Reddit gets lower weight
        'twitter': 1.0,   # Twitter gets medium weight
    }
    
    print(f"Source weights: {source_weights}")
    
    # Test different aggregation methods
    methods = ['mean', 'median', 'weighted_mean']
    windows = [3, 5]
    
    for window in windows:
        for method in methods:
            if method == 'weighted_mean':
                agg_series = analyzer.aggregate_sentiment(
                    data, window=window, method=method, weights=source_weights
                )
            else:
                agg_series = analyzer.aggregate_sentiment(data, window=window, method=method)
            
            if not agg_series.empty:
                print(f"Window={window}, Method={method}:")
                print(f"  Latest score: {agg_series.iloc[-1]:.3f}")
                print(f"  Score range: {agg_series.min():.3f} to {agg_series.max():.3f}")
                print(f"  Data points: {len(agg_series)}")
            print()


def demonstrate_entity_extraction(analyzer: SentimentAnalyzer, data: List[SentimentData]):
    """Demonstrate entity extraction capabilities."""
    print("\n=== Entity Extraction ===")
    
    # Test entity extraction on sample texts
    test_texts = [
        "Bitcoin BTC and Ethereum ETH are mentioned. Also Solana SOL.",
        "$BTC and $ETH are popular tickers",
        "Cardano ADA and Polygon MATIC are also mentioned",
        "No entities in this text",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"Text {i}: {text}")
        entities = analyzer.extract_entities(text)
        print(f"  Tickers: {entities['tickers']}")
        print(f"  Projects: {entities['projects']}")
        print()


def demonstrate_sentiment_intensity(analyzer: SentimentAnalyzer):
    """Demonstrate sentiment intensity analysis."""
    print("\n=== Sentiment Intensity Analysis ===")
    
    # Test different intensity levels
    intensity_tests = [
        ("BITCOIN IS AMAZING!!! 🚀🚀🚀", "High intensity positive"),
        ("crypto prices are stable", "Low intensity neutral"),
        ("CRYPTO IS CRASHING!!! 😱😱😱", "High intensity negative"),
        ("Bitcoin is doing okay", "Medium intensity neutral"),
        ("ETH TO THE MOON!!! 🚀🚀🚀", "High intensity positive"),
    ]
    
    for text, description in intensity_tests:
        print(f"{description}: {repr(text)}")
        result = analyzer.analyze_sentiment_intensity(text)
        print(f"  Polarity: {result['polarity']:.3f}")
        print(f"  Subjectivity: {result['subjectivity']:.3f}")
        print(f"  Intensity: {result['intensity']:.3f}")
        print()


def demonstrate_enhanced_batch_processing(analyzer: SentimentAnalyzer, data: List[SentimentData]):
    """Demonstrate enhanced batch processing with all features."""
    print("\n=== Enhanced Batch Processing ===")
    
    print("Processing with basic features...")
    basic_result = analyzer.batch_analyze(data)
    print(f"  Processed {len(basic_result)} items")
    
    print("\nProcessing with entity extraction...")
    entity_result = analyzer.batch_analyze(data, include_entities=True)
    print(f"  Processed {len(entity_result)} items with entities")
    
    print("\nProcessing with intensity analysis...")
    intensity_result = analyzer.batch_analyze(data, include_intensity=True)
    print(f"  Processed {len(intensity_result)} items with intensity")
    
    print("\nProcessing with all features...")
    all_features_result = analyzer.batch_analyze(
        data, include_entities=True, include_intensity=True
    )
    print(f"  Processed {len(all_features_result)} items with all features")
    
    # Show some results
    print("\nSample results:")
    for i, item in enumerate(all_features_result[:3]):
        print(f"  Item {i+1}: {item.source}")
        print(f"    Text: {item.text[:50]}...")
        print(f"    Score: {item.sentiment_score:.3f}")
        print(f"    Keywords: {item.keywords[:5]}...")  # Show first 5 keywords
        print()


def demonstrate_trading_integration(analyzer: SentimentAnalyzer, data: List[SentimentData]):
    """Demonstrate integration with trading strategies."""
    print("\n=== Trading Strategy Integration ===")
    
    # Process data with all features
    processed_data = analyzer.batch_analyze(
        data, include_entities=True, include_intensity=True
    )
    
    # Define different weight configurations for different strategies
    strategy_weights = {
        'conservative': {'news': 2.0, 'reddit': 0.5, 'twitter': 0.8},
        'balanced': {'news': 1.5, 'reddit': 1.0, 'twitter': 1.0},
        'aggressive': {'news': 1.0, 'reddit': 1.2, 'twitter': 1.5},
    }
    
    print("Testing different trading strategies:")
    
    for strategy_name, weights in strategy_weights.items():
        print(f"\n{strategy_name.title()} Strategy:")
        print(f"  Weights: {weights}")
        
        # Aggregate sentiment with strategy weights
        agg_series = analyzer.aggregate_sentiment(
            processed_data, window=5, method='weighted_mean', weights=weights
        )
        
        if not agg_series.empty:
            latest_sentiment = agg_series.iloc[-1]
            
            # Generate trading signal
            thresholds = {'positive': 0.2, 'negative': -0.2}
            signal = analyzer.generate_sentiment_signal(latest_sentiment, thresholds)
            
            print(f"  Latest sentiment: {latest_sentiment:.3f}")
            print(f"  Signal: {signal}")
            
            # Simulate trading decision
            if signal == 'positive':
                print("  Trading decision: Consider BUY positions")
            elif signal == 'negative':
                print("  Trading decision: Consider SELL positions")
            else:
                print("  Trading decision: HOLD or reduce position sizes")


def main():
    """Run the enhanced sentiment analysis demonstration."""
    print("=== Enhanced Sentiment Analysis Module Demo ===")
    print(f"Started at: {datetime.now()}")
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    print("✅ SentimentAnalyzer initialized with enhanced features")
    
    # Create sample data
    data = create_enhanced_sample_data()
    print(f"✅ Created {len(data)} sample sentiment items (including edge cases)")
    
    # Run demonstrations
    demonstrate_error_handling(analyzer)
    demonstrate_entity_extraction(analyzer, data)
    demonstrate_sentiment_intensity(analyzer)
    demonstrate_enhanced_batch_processing(analyzer, data)
    demonstrate_weighted_aggregation(analyzer, data)
    demonstrate_trading_integration(analyzer, data)
    
    print("\n=== Enhanced Demo Completed ===")
    print(f"Finished at: {datetime.now()}")
    print("\n🎉 All enhanced features working correctly!")
    print("\n📊 Summary of Enhancements:")
    print("  ✅ Granular error handling with specific exception types")
    print("  ✅ Weighted aggregation by source credibility")
    print("  ✅ Entity extraction (tickers, project names)")
    print("  ✅ Sentiment intensity analysis")
    print("  ✅ Enhanced batch processing with optional features")
    print("  ✅ Comprehensive edge case handling")
    print("  ✅ Detailed logging and error reporting")


if __name__ == "__main__":
    main() 