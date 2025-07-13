"""
Signal Generator Demo

This demo shows how to use the SignalGenerator to combine statistical arbitrage
and sentiment signals for trading decisions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from src.strategy.signal_generator import SignalGenerator, TradeSignal, CombinationMethod
from src.strategy.stat_arb import StatisticalArbitrage, create_stat_arb_strategy
from src.strategy.sentiment import SentimentAnalyzer, SentimentData
from src.execution.risk_manager import RiskManager
from src.utils.config_loader import config
from src.utils.logger import logger


def create_mock_sentiment_data() -> List[SentimentData]:
    """Create mock sentiment data for demonstration."""
    sentiment_data = []
    
    # Mock data for different sources
    sources = ['reddit', 'twitter', 'news']
    assets = ['BTC', 'ETH', 'SOL']
    
    base_time = datetime.now()
    
    for i in range(20):
        source = sources[i % len(sources)]
        asset = assets[i % len(assets)]
        
        # Generate realistic sentiment scores
        if source == 'reddit':
            sentiment_score = np.random.normal(0.1, 0.3)  # Slightly positive
        elif source == 'twitter':
            sentiment_score = np.random.normal(-0.05, 0.4)  # More volatile
        else:  # news
            sentiment_score = np.random.normal(0.2, 0.2)  # More positive
        
        # Clamp to [-1, 1] range
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        sentiment_data.append(SentimentData(
            timestamp=base_time - timedelta(minutes=i*5),
            source=source,
            text=f"Mock sentiment text about {asset} from {source}",
            sentiment_score=sentiment_score,
            keywords=[asset.lower(), 'crypto', 'trading']
        ))
    
    return sentiment_data


def demo_basic_signal_generation():
    """Demonstrate basic signal generation."""
    print("\n=== Basic Signal Generation Demo ===")
    
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
    
    # Generate mock sentiment data
    sentiment_data = create_mock_sentiment_data()
    
    # Generate signals
    signals = signal_gen.generate_signals(sentiment_data=sentiment_data)
    
    print(f"Generated {len(signals)} signals")
    for signal in signals:
        print(f"  {signal.symbol} {signal.side} qty={signal.quantity:.4f} "
              f"conf={signal.confidence:.3f} sources={signal.sources}")


def demo_combination_methods():
    """Demonstrate different signal combination methods."""
    print("\n=== Signal Combination Methods Demo ===")
    
    # Create components
    stat_arb = create_stat_arb_strategy()
    sentiment_analyzer = SentimentAnalyzer()
    
    methods = [
        ('consensus', 'Consensus - Both signals must agree'),
        ('weighted', 'Weighted - Weighted average of signals'),
        ('filter', 'Filter - Sentiment filters stat arb signals'),
        ('hybrid', 'Hybrid - Combination of consensus and weighted')
    ]
    
    sentiment_data = create_mock_sentiment_data()
    
    for method_name, description in methods:
        print(f"\n{description}:")
        
        # Create signal generator with specific method
        signal_gen = SignalGenerator(
            stat_arb=stat_arb,
            sentiment_analyzer=sentiment_analyzer,
            config={'combination_method': method_name}
        )
        
        signals = signal_gen.generate_signals(sentiment_data=sentiment_data)
        print(f"  Generated {len(signals)} signals using {method_name} method")
        
        if signals:
            avg_confidence = np.mean([s.confidence for s in signals])
            print(f"  Average confidence: {avg_confidence:.3f}")


def demo_risk_integration():
    """Demonstrate risk manager integration."""
    print("\n=== Risk Manager Integration Demo ===")
    
    # Create components
    stat_arb = create_stat_arb_strategy()
    sentiment_analyzer = SentimentAnalyzer()
    risk_manager = RiskManager()
    
    # Create signal generator with risk checks enabled
    signal_gen = SignalGenerator(
        stat_arb=stat_arb,
        sentiment_analyzer=sentiment_analyzer,
        risk_manager=risk_manager,
        config={
            'enable_risk_checks': True,
            'require_risk_approval': False  # Allow signals even if risk check fails
        }
    )
    
    sentiment_data = create_mock_sentiment_data()
    
    # Generate signals with risk checks
    signals = signal_gen.generate_signals(sentiment_data=sentiment_data)
    
    print(f"Generated {len(signals)} signals with risk checks")
    
    # Show risk check results
    risk_checked = [s for s in signals if s.risk_checked]
    not_risk_checked = [s for s in signals if not s.risk_checked]
    
    print(f"  Risk-checked signals: {len(risk_checked)}")
    print(f"  Non-risk-checked signals: {len(not_risk_checked)}")


def demo_configuration_options():
    """Demonstrate different configuration options."""
    print("\n=== Configuration Options Demo ===")
    
    # Create base components
    stat_arb = create_stat_arb_strategy()
    sentiment_analyzer = SentimentAnalyzer()
    
    # Test different confidence thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7]
    sentiment_data = create_mock_sentiment_data()
    
    for threshold in thresholds:
        print(f"\nMinimum confidence threshold: {threshold}")
        
        signal_gen = SignalGenerator(
            stat_arb=stat_arb,
            sentiment_analyzer=sentiment_analyzer,
            config={'min_confidence': threshold}
        )
        
        signals = signal_gen.generate_signals(sentiment_data=sentiment_data)
        print(f"  Generated {len(signals)} signals")
        
        if signals:
            avg_confidence = np.mean([s.confidence for s in signals])
            print(f"  Average confidence: {avg_confidence:.3f}")
    
    # Test different weights
    print(f"\nWeight combinations:")
    weight_configs = [
        {'stat_weight': 0.8, 'sentiment_weight': 0.2, 'name': 'Stat-heavy'},
        {'stat_weight': 0.5, 'sentiment_weight': 0.5, 'name': 'Balanced'},
        {'stat_weight': 0.2, 'sentiment_weight': 0.8, 'name': 'Sentiment-heavy'}
    ]
    
    for weight_config in weight_configs:
        name = weight_config.pop('name')
        print(f"  {name}:")
        
        signal_gen = SignalGenerator(
            stat_arb=stat_arb,
            sentiment_analyzer=sentiment_analyzer,
            config={'combination_method': 'weighted', **weight_config}
        )
        
        signals = signal_gen.generate_signals(sentiment_data=sentiment_data)
        print(f"    Generated {len(signals)} signals")


def demo_signal_history():
    """Demonstrate signal history tracking."""
    print("\n=== Signal History Demo ===")
    
    # Create signal generator
    signal_gen = SignalGenerator(
        stat_arb=create_stat_arb_strategy(),
        sentiment_analyzer=SentimentAnalyzer()
    )
    
    sentiment_data = create_mock_sentiment_data()
    
    # Generate signals multiple times
    for i in range(3):
        print(f"\nBatch {i+1}:")
        signals = signal_gen.generate_signals(sentiment_data=sentiment_data)
        print(f"  Generated {len(signals)} signals")
    
    # Get signal history
    history = signal_gen.get_signal_history(hours=1)
    print(f"\nSignal history (last hour): {len(history)} signals")
    
    if history:
        # Group by symbol
        by_symbol = {}
        for signal in history:
            symbol = signal['symbol']
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(signal)
        
        for symbol, signals in by_symbol.items():
            buy_count = len([s for s in signals if s['side'] == 'buy'])
            sell_count = len([s for s in signals if s['side'] == 'sell'])
            avg_conf = np.mean([s['confidence'] for s in signals])
            print(f"  {symbol}: {buy_count} buy, {sell_count} sell, avg conf={avg_conf:.3f}")


def demo_performance_metrics():
    """Demonstrate performance metrics."""
    print("\n=== Performance Metrics Demo ===")
    
    # Create signal generator
    signal_gen = SignalGenerator(
        stat_arb=create_stat_arb_strategy(),
        sentiment_analyzer=SentimentAnalyzer()
    )
    
    sentiment_data = create_mock_sentiment_data()
    
    # Generate signals multiple times
    for i in range(5):
        signals = signal_gen.generate_signals(sentiment_data=sentiment_data)
        print(f"Batch {i+1}: {len(signals)} signals")
    
    # Get performance metrics
    metrics = signal_gen.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


def demo_custom_trade_signal():
    """Demonstrate creating custom trade signals."""
    print("\n=== Custom Trade Signal Demo ===")
    
    # Create a custom trade signal
    custom_signal = TradeSignal(
        symbol="BTC-ETH",
        side="buy",
        quantity=0.1,
        order_type="market",
        price=None,
        confidence=0.75,
        sources=["custom", "manual"],
        metadata={
            'reason': 'Manual override',
            'user': 'trader',
            'notes': 'Strong technical breakout'
        },
        timestamp=datetime.now(),
        signal_type="entry_long",
        risk_checked=False
    )
    
    print(f"Custom signal created:")
    print(f"  Symbol: {custom_signal.symbol}")
    print(f"  Side: {custom_signal.side}")
    print(f"  Quantity: {custom_signal.quantity}")
    print(f"  Confidence: {custom_signal.confidence}")
    print(f"  Sources: {custom_signal.sources}")
    print(f"  Valid: {custom_signal.validate()}")


def main():
    """Run all demos."""
    print("Signal Generator Demo Suite")
    print("=" * 50)
    
    try:
        # Run all demos
        demo_basic_signal_generation()
        demo_combination_methods()
        demo_risk_integration()
        demo_configuration_options()
        demo_signal_history()
        demo_performance_metrics()
        demo_custom_trade_signal()
        
        print("\n" + "=" * 50)
        print("All demos completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    main() 