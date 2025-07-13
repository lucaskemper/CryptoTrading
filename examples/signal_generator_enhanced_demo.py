"""
Enhanced Signal Generator Demo

This demo showcases the new features of the enhanced signal generator:
- Multi-asset portfolio signal generation
- Real-time asynchronous signal streaming
- Advanced analytics and performance tracking
- Portfolio-level decision making
- Signal-to-trade conversion tracking
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import threading

from src.strategy.signal_generator import (
    SignalGenerator, TradeSignal, PortfolioSignal, SignalAnalytics,
    SignalScope, CombinationMethod, SignalType, SignalSource
)
from src.strategy.stat_arb import StatisticalArbitrage, Signal as StatSignal
from src.strategy.sentiment import SentimentAnalyzer, SentimentData
from src.execution.risk_manager import RiskManager
from src.execution.position_manager import PositionManager
from src.utils.logger import logger


class MockStatisticalArbitrage:
    """Mock statistical arbitrage strategy for demo."""
    
    def __init__(self):
        self.logger = logger
        self.price_data = {
            'BTC': [50000, 51000, 52000, 51500, 53000],
            'ETH': [3000, 3100, 3200, 3150, 3300],
            'SOL': [100, 105, 110, 108, 115]
        }
    
    def generate_signals(self) -> List[StatSignal]:
        """Generate mock statistical arbitrage signals."""
        signals = []
        
        # Mock BTC-ETH pair signal
        btc_eth_signal = StatSignal(
            pair="BTC-ETH",
            signal_type="entry_long",
            confidence=0.75,
            z_score=2.1,
            spread_value=0.05,
            asset1="BTC",
            asset2="ETH",
            action1="buy",
            action2="sell",
            size1=0.1,
            size2=1.5,
            timestamp=datetime.now()
        )
        signals.append(btc_eth_signal)
        
        # Mock SOL-BTC pair signal
        sol_btc_signal = StatSignal(
            pair="SOL-BTC",
            signal_type="entry_short",
            confidence=0.65,
            z_score=-1.8,
            spread_value=-0.03,
            asset1="SOL",
            asset2="BTC",
            action1="sell",
            action2="buy",
            size1=10.0,
            size2=0.05,
            timestamp=datetime.now()
        )
        signals.append(sol_btc_signal)
        
        return signals


class MockSentimentAnalyzer:
    """Mock sentiment analyzer for demo."""
    
    def __init__(self):
        self.logger = logger
    
    def aggregate_sentiment(self, sentiment_data, window=10, method='mean') -> Dict[str, float]:
        """Aggregate sentiment scores by asset."""
        # Mock sentiment scores
        return {
            'BTC': 0.3,   # Positive sentiment
            'ETH': 0.1,   # Slightly positive
            'SOL': -0.2,  # Negative sentiment
            'AVAX': 0.0,  # Neutral
            'USDT': 0.0   # Neutral (stablecoin)
        }


class MockPositionManager:
    """Mock position manager for demo."""
    
    def __init__(self):
        self.logger = logger
        self.positions = {}
    
    def get_portfolio_metrics(self):
        """Get mock portfolio metrics."""
        from dataclasses import dataclass
        
        @dataclass
        class MockPortfolioMetrics:
            total_market_value: float
            total_unrealized_pnl: float
            total_realized_pnl: float
            total_pnl: float
            exposure_by_asset: Dict[str, float]
            exposure_by_sector: Dict[str, float]
        
        return MockPortfolioMetrics(
            total_market_value=10000.0,
            total_unrealized_pnl=500.0,
            total_realized_pnl=200.0,
            total_pnl=700.0,
            exposure_by_asset={
                'BTC': 4000.0,  # 40%
                'ETH': 3000.0,  # 30%
                'SOL': 2000.0,  # 20%
                'USDT': 1000.0  # 10%
            },
            exposure_by_sector={
                'Layer1': 9000.0,  # 90% (BTC, ETH, SOL)
                'Stablecoin': 1000.0  # 10% (USDT)
            }
        )


def signal_callback(signals: List[TradeSignal]):
    """Callback function for real-time signal processing."""
    logger.info(f"Real-time signal callback received {len(signals)} signals")
    for signal in signals:
        logger.info(f"  - {signal.symbol} {signal.side} qty={signal.quantity:.4f} "
                   f"conf={signal.confidence:.3f}")


def demo_basic_signal_generation():
    """Demo basic signal generation functionality."""
    logger.info("=== Basic Signal Generation Demo ===")
    
    # Initialize components
    stat_arb = MockStatisticalArbitrage()
    sentiment_analyzer = MockSentimentAnalyzer()
    risk_manager = RiskManager()
    position_manager = MockPositionManager()
    
    # Create signal generator
    config = {
        'combination_method': 'weighted',
        'stat_weight': 0.6,
        'sentiment_weight': 0.4,
        'min_confidence': 0.3,
        'enable_risk_checks': True
    }
    
    signal_generator = SignalGenerator(
        stat_arb=stat_arb,
        sentiment_analyzer=sentiment_analyzer,
        risk_manager=risk_manager,
        position_manager=position_manager,
        config=config
    )
    
    # Generate signals
    signals = signal_generator.generate_signals()
    
    logger.info(f"Generated {len(signals)} signals:")
    for signal in signals:
        logger.info(f"  - {signal.symbol} {signal.side} qty={signal.quantity:.4f} "
                   f"conf={signal.confidence:.3f} scope={signal.scope.value}")
    
    return signal_generator, signals


def demo_portfolio_signals():
    """Demo portfolio-level signal generation."""
    logger.info("\n=== Portfolio Signal Generation Demo ===")
    
    # Create signal generator with portfolio features
    stat_arb = MockStatisticalArbitrage()
    sentiment_analyzer = MockSentimentAnalyzer()
    risk_manager = RiskManager()
    position_manager = MockPositionManager()
    
    config = {
        'portfolio_rebalance_threshold': 0.1,
        'max_portfolio_deviation': 0.2,
        'correlation_threshold': 0.7
    }
    
    signal_generator = SignalGenerator(
        stat_arb=stat_arb,
        sentiment_analyzer=sentiment_analyzer,
        risk_manager=risk_manager,
        position_manager=position_manager,
        config=config
    )
    
    # Generate portfolio signals
    portfolio_signals = signal_generator._generate_portfolio_signals()
    
    logger.info(f"Generated {len(portfolio_signals)} portfolio signals:")
    for signal in portfolio_signals:
        logger.info(f"  - Portfolio {signal.portfolio_id}: {signal.signal_type}")
        logger.info(f"    Confidence: {signal.confidence:.3f}")
        logger.info(f"    Actions: {len(signal.rebalance_actions)}")
        
        for action in signal.rebalance_actions:
            if 'asset' in action:
                logger.info(f"      {action['asset']}: {action['action']} "
                           f"({action['adjustment']:.3f})")
            elif 'sector' in action:
                logger.info(f"      {action['sector']}: reduce by {action['reduction_needed']:.2f}")
    
    return signal_generator, portfolio_signals


def demo_real_time_streaming():
    """Demo real-time signal streaming."""
    logger.info("\n=== Real-Time Signal Streaming Demo ===")
    
    # Create signal generator with streaming enabled
    stat_arb = MockStatisticalArbitrage()
    sentiment_analyzer = MockSentimentAnalyzer()
    risk_manager = RiskManager()
    position_manager = MockPositionManager()
    
    config = {
        'enable_streaming': True,
        'stream_interval': 5,  # 5 seconds for demo
        'combination_method': 'hybrid'
    }
    
    signal_generator = SignalGenerator(
        stat_arb=stat_arb,
        sentiment_analyzer=sentiment_analyzer,
        risk_manager=risk_manager,
        position_manager=position_manager,
        config=config
    )
    
    # Start streaming
    signal_generator.start_streaming(callback=signal_callback)
    
    logger.info("Signal streaming started. Collecting signals for 15 seconds...")
    
    # Collect signals for 15 seconds
    start_time = time.time()
    collected_signals = []
    
    while time.time() - start_time < 15:
        # Get signals from queue
        streaming_signals = signal_generator.get_streaming_signals()
        collected_signals.extend(streaming_signals)
        
        if streaming_signals:
            logger.info(f"Collected {len(streaming_signals)} signals from stream")
        
        time.sleep(1)
    
    # Stop streaming
    signal_generator.stop_streaming()
    
    logger.info(f"Streaming stopped. Total signals collected: {len(collected_signals)}")
    
    return signal_generator, collected_signals


def demo_advanced_analytics():
    """Demo advanced analytics and performance tracking."""
    logger.info("\n=== Advanced Analytics Demo ===")
    
    # Create signal generator
    stat_arb = MockStatisticalArbitrage()
    sentiment_analyzer = MockSentimentAnalyzer()
    risk_manager = RiskManager()
    position_manager = MockPositionManager()
    
    signal_generator = SignalGenerator(
        stat_arb=stat_arb,
        sentiment_analyzer=sentiment_analyzer,
        risk_manager=risk_manager,
        position_manager=position_manager
    )
    
    # Generate some signals
    signals = signal_generator.generate_signals()
    
    # Simulate signal execution and performance
    for signal in signals:
        # Simulate execution
        signal_generator.update_signal_execution(
            signal.signal_id, 
            executed=True, 
            execution_price=signal.price or 50000.0
        )
        
        # Simulate signal close with profit/loss
        if signal.side == 'buy':
            exit_price = (signal.price or 50000.0) * 1.05  # 5% profit
        else:
            exit_price = (signal.price or 50000.0) * 0.95  # 5% loss
        
        signal_generator.close_signal(
            signal.signal_id,
            exit_price=exit_price,
            exit_time=datetime.now()
        )
    
    # Get analytics report
    analytics_report = signal_generator.get_analytics_report()
    
    logger.info("Analytics Report:")
    logger.info(f"  Total Signals: {analytics_report['total_signals']}")
    logger.info(f"  Executed Signals: {analytics_report['executed_signals']}")
    
    # Safely access conversion rates
    conversion_rates = analytics_report.get('conversion_rates', {})
    overall_rate = conversion_rates.get('overall', 0.0)
    logger.info(f"  Conversion Rate: {overall_rate:.2%}")
    
    if 'performance_metrics' in analytics_report:
        metrics = analytics_report['performance_metrics']
        logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        logger.info(f"  Average PnL: {metrics.get('avg_pnl', 0):.2%}")
        logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    
    logger.info(f"  Total PnL: {analytics_report['total_pnl']:.2%}")
    logger.info(f"  Signal Distribution: {analytics_report['signal_distribution']}")
    
    return signal_generator, analytics_report


def demo_signal_combination_methods():
    """Demo different signal combination methods."""
    logger.info("\n=== Signal Combination Methods Demo ===")
    
    stat_arb = MockStatisticalArbitrage()
    sentiment_analyzer = MockSentimentAnalyzer()
    risk_manager = RiskManager()
    position_manager = MockPositionManager()
    
    combination_methods = [
        'consensus',
        'weighted', 
        'filter',
        'hybrid',
        'portfolio_optimized'
    ]
    
    for method in combination_methods:
        logger.info(f"\n--- {method.upper()} Method ---")
        
        config = {
            'combination_method': method,
            'stat_weight': 0.6,
            'sentiment_weight': 0.4,
            'min_confidence': 0.3
        }
        
        signal_generator = SignalGenerator(
            stat_arb=stat_arb,
            sentiment_analyzer=sentiment_analyzer,
            risk_manager=risk_manager,
            position_manager=position_manager,
            config=config
        )
        
        signals = signal_generator.generate_signals()
        
        logger.info(f"Generated {len(signals)} signals using {method} method:")
        for signal in signals:
            logger.info(f"  - {signal.symbol} {signal.side} conf={signal.confidence:.3f}")


def demo_performance_tracking():
    """Demo comprehensive performance tracking."""
    logger.info("\n=== Performance Tracking Demo ===")
    
    # Create signal generator
    stat_arb = MockStatisticalArbitrage()
    sentiment_analyzer = MockSentimentAnalyzer()
    risk_manager = RiskManager()
    position_manager = MockPositionManager()
    
    signal_generator = SignalGenerator(
        stat_arb=stat_arb,
        sentiment_analyzer=sentiment_analyzer,
        risk_manager=risk_manager,
        position_manager=position_manager
    )
    
    # Simulate multiple trading sessions
    for session in range(3):
        logger.info(f"\n--- Trading Session {session + 1} ---")
        
        # Generate signals
        signals = signal_generator.generate_signals()
        
        # Simulate execution and performance
        for i, signal in enumerate(signals):
            # Simulate execution
            signal_generator.update_signal_execution(
                signal.signal_id, 
                executed=True, 
                execution_price=signal.price or 50000.0
            )
            
            # Simulate different outcomes
            if i % 2 == 0:  # Even signals are profitable
                exit_price = (signal.price or 50000.0) * 1.03  # 3% profit
            else:  # Odd signals are losses
                exit_price = (signal.price or 50000.0) * 0.98  # 2% loss
            
            signal_generator.close_signal(
                signal.signal_id,
                exit_price=exit_price,
                exit_time=datetime.now()
            )
        
        # Get performance metrics
        metrics = signal_generator.get_performance_metrics()
        logger.info(f"Session {session + 1} Performance:")
        logger.info(f"  Total Signals: {metrics['total_signals']}")
        logger.info(f"  Approved Signals: {metrics['approved_signals']}")
        logger.info(f"  Approval Rate: {metrics['approval_rate']:.2%}")
    
    # Final analytics report
    analytics_report = signal_generator.get_analytics_report()
    logger.info(f"\nFinal Analytics Report:")
    logger.info(f"  Total PnL: {analytics_report['total_pnl']:.2%}")
    logger.info(f"  Win Rate: {analytics_report['performance_metrics'].get('win_rate', 0):.2%}")
    logger.info(f"  Sharpe Ratio: {analytics_report['sharpe_ratio']:.3f}")


def main():
    """Run all demos."""
    logger.info("Enhanced Signal Generator Demo")
    logger.info("=" * 50)
    
    try:
        # Demo 1: Basic signal generation
        signal_generator, signals = demo_basic_signal_generation()
        
        # Demo 2: Portfolio signals
        portfolio_generator, portfolio_signals = demo_portfolio_signals()
        
        # Demo 3: Real-time streaming
        streaming_generator, streaming_signals = demo_real_time_streaming()
        
        # Demo 4: Advanced analytics
        analytics_generator, analytics_report = demo_advanced_analytics()
        
        # Demo 5: Signal combination methods
        demo_signal_combination_methods()
        
        # Demo 6: Performance tracking
        demo_performance_tracking()
        
        logger.info("\n" + "=" * 50)
        logger.info("All demos completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 