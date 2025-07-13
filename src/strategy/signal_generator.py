"""
Signal Generator Module

This module combines outputs from statistical arbitrage strategy and sentiment analysis
to produce actionable, risk-aware trading signals. It acts as the central logic layer
between strategy modules and execution systems.

Enhanced Features:
- Multi-asset portfolio signal generation
- Real-time asynchronous signal streaming
- Advanced analytics and performance tracking
- Portfolio-level decision making
- Signal-to-trade conversion tracking
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Callable, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import uuid
import time

from src.utils.logger import logger
from src.utils.config_loader import config
from src.strategy.stat_arb import StatisticalArbitrage, Signal as StatSignal
from src.strategy.sentiment import SentimentAnalyzer, SentimentData


class SignalType(Enum):
    """Signal type enumeration."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT = "exit"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    PORTFOLIO_HEDGE = "portfolio_hedge"


class SignalSource(Enum):
    """Signal source enumeration."""
    STATISTICAL_ARB = "statistical_arbitrage"
    SENTIMENT = "sentiment"
    COMBINED = "combined"
    RISK_MANAGER = "risk_manager"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"


class CombinationMethod(Enum):
    """Signal combination method enumeration."""
    CONSENSUS = "consensus"  # Both signals must agree
    WEIGHTED = "weighted"    # Weighted average of signals
    FILTER = "filter"        # Sentiment filters stat arb signals
    HYBRID = "hybrid"        # Combination of methods
    PORTFOLIO_OPTIMIZED = "portfolio_optimized"  # Portfolio-aware combination


class SignalScope(Enum):
    """Signal scope enumeration."""
    SINGLE_ASSET = "single_asset"
    PAIR_TRADE = "pair_trade"
    PORTFOLIO_LEVEL = "portfolio_level"
    SECTOR_LEVEL = "sector_level"


@dataclass
class PortfolioSignal:
    """
    Portfolio-level signal for multi-asset decisions.
    
    Attributes:
        portfolio_id: Unique portfolio identifier
        signal_type: Type of portfolio action
        target_allocation: Target allocation by asset
        current_allocation: Current allocation by asset
        rebalance_actions: List of individual trade actions
        confidence: Overall portfolio signal confidence
        metadata: Additional portfolio metadata
        timestamp: Signal generation timestamp
    """
    portfolio_id: str
    signal_type: str  # 'rebalance', 'hedge', 'diversify', 'concentrate'
    target_allocation: Dict[str, float]
    current_allocation: Dict[str, float]
    rebalance_actions: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def validate(self) -> bool:
        """Validate portfolio signal parameters."""
        if not self.portfolio_id or not self.signal_type:
            return False
        if self.confidence < 0 or self.confidence > 1:
            return False
        if not self.target_allocation or not self.current_allocation:
            return False
        # Check that allocations sum to approximately 1.0
        target_sum = sum(self.target_allocation.values())
        current_sum = sum(self.current_allocation.values())
        if abs(target_sum - 1.0) > 0.01 or abs(current_sum - 1.0) > 0.01:
            return False
        return True


@dataclass
class SignalPerformance:
    """
    Performance tracking for individual signals.
    
    Attributes:
        signal_id: Unique signal identifier
        symbol: Trading symbol
        signal_type: Type of signal
        entry_price: Entry price
        exit_price: Exit price (if closed)
        entry_time: Entry timestamp
        exit_time: Exit timestamp (if closed)
        pnl: Realized PnL
        pnl_percentage: PnL as percentage
        holding_time: Time held
        status: Signal status
    """
    signal_id: str
    symbol: str
    signal_type: str
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    holding_time: Optional[timedelta] = None
    status: str = "open"  # 'open', 'closed', 'cancelled'
    
    def close(self, exit_price: float, exit_time: Optional[datetime] = None):
        """Close the signal and calculate performance."""
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now()
        self.holding_time = self.exit_time - self.entry_time
        
        # Calculate PnL
        if self.signal_type in ['entry_long', 'buy']:
            self.pnl = (exit_price - self.entry_price) / self.entry_price
        elif self.signal_type in ['entry_short', 'sell']:
            self.pnl = (self.entry_price - exit_price) / self.entry_price
        else:
            self.pnl = 0.0
        
        self.pnl_percentage = self.pnl * 100
        self.status = "closed"


@dataclass
class TradeSignal:
    """
    Standardized trade signal data structure.
    
    Attributes:
        symbol: Trading pair symbol (e.g., 'BTC-ETH')
        side: Trade side ('buy' or 'sell')
        quantity: Position size
        order_type: Order type ('market' or 'limit')
        price: Limit price (None for market orders)
        confidence: Signal confidence score (0-1)
        sources: List of signal sources
        metadata: Additional signal metadata
        timestamp: Signal generation timestamp
        signal_type: Type of signal (entry, exit, etc.)
        risk_checked: Whether risk manager has validated the signal
        scope: Signal scope (single asset, pair, portfolio)
        signal_id: Unique signal identifier
    """
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: str  # 'market' or 'limit'
    price: Optional[float]
    confidence: float
    sources: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime
    signal_type: str  # 'entry_long', 'entry_short', 'exit'
    risk_checked: bool = False
    scope: SignalScope = SignalScope.SINGLE_ASSET
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def validate(self) -> bool:
        """Validate signal parameters."""
        if not self.symbol or not self.side or self.quantity <= 0:
            return False
        if self.confidence < 0 or self.confidence > 1:
            return False
        if self.side not in ['buy', 'sell']:
            return False
        if self.order_type not in ['market', 'limit']:
            return False
        if self.order_type == 'limit' and self.price is None:
            return False
        return True


class SignalAnalytics:
    """
    Advanced analytics for signal performance tracking.
    
    Features:
    - Signal-to-trade conversion rates
    - Post-trade performance analysis
    - Portfolio-level analytics
    - Risk-adjusted returns
    - Correlation analysis
    """
    
    def __init__(self):
        self.signals: Dict[str, SignalPerformance] = {}
        self.portfolio_signals: Dict[str, PortfolioSignal] = {}
        self.conversion_rates: Dict[str, float] = defaultdict(float)
        self.performance_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Performance tracking
        self.total_signals = 0
        self.executed_signals = 0
        self.profitable_signals = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        
        # Time series data
        self.daily_returns = deque(maxlen=252)  # One year
        self.signal_history = deque(maxlen=1000)
        
        logger.info("Signal Analytics initialized")
    
    def add_signal(self, signal: TradeSignal):
        """Add a new signal for tracking."""
        performance = SignalPerformance(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            signal_type=signal.signal_type,
            entry_price=signal.price or 0.0,
            entry_time=signal.timestamp
        )
        self.signals[signal.signal_id] = performance
        self.total_signals += 1
        self.signal_history.append({
            'signal_id': signal.signal_id,
            'symbol': signal.symbol,
            'signal_type': signal.signal_type,
            'confidence': signal.confidence,
            'timestamp': signal.timestamp
        })
    
    def update_signal_execution(self, signal_id: str, executed: bool, 
                               execution_price: Optional[float] = None):
        """Update signal execution status."""
        if signal_id not in self.signals:
            logger.warning(f"Signal {signal_id} not found for execution update")
            return
        
        signal = self.signals[signal_id]
        if executed:
            self.executed_signals += 1
            if execution_price:
                signal.entry_price = execution_price
    
    def close_signal(self, signal_id: str, exit_price: float, 
                    exit_time: Optional[datetime] = None):
        """Close a signal and calculate performance."""
        if signal_id not in self.signals:
            logger.warning(f"Signal {signal_id} not found for close")
            return
        
        signal = self.signals[signal_id]
        signal.close(exit_price, exit_time)
        
        # Update performance metrics
        if signal.pnl > 0:
            self.profitable_signals += 1
        
        self.total_pnl += signal.pnl
        self.daily_returns.append(signal.pnl)
        
        # Update conversion rates
        self._update_conversion_rates()
        
        # Update performance metrics
        self._update_performance_metrics()
    
    def _update_conversion_rates(self):
        """Update signal-to-trade conversion rates."""
        if self.total_signals > 0:
            self.conversion_rates['overall'] = self.executed_signals / self.total_signals
        
        # Calculate by signal type
        signal_types = defaultdict(int)
        executed_types = defaultdict(int)
        
        for signal in self.signals.values():
            signal_types[signal.signal_type] += 1
            if signal.status == "closed":
                executed_types[signal.signal_type] += 1
        
        for signal_type, total in signal_types.items():
            if total > 0:
                self.conversion_rates[signal_type] = executed_types[signal_type] / total
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        closed_signals = [s for s in self.signals.values() if s.status == "closed"]
        
        if not closed_signals:
            return
        
        # Calculate win rate
        if self.executed_signals > 0:
            self.performance_metrics['win_rate'] = self.profitable_signals / self.executed_signals
        
        # Calculate average PnL
        avg_pnl = np.mean([s.pnl for s in closed_signals])
        self.performance_metrics['avg_pnl'] = avg_pnl
        
        # Calculate Sharpe ratio
        if len(self.daily_returns) > 1:
            returns_array = np.array(list(self.daily_returns))
            if np.std(returns_array) > 0:
                self.sharpe_ratio = np.mean(returns_array) / np.std(returns_array)
                self.performance_metrics['sharpe_ratio'] = self.sharpe_ratio
        
        # Calculate max drawdown
        cumulative_returns = np.cumsum(list(self.daily_returns))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        self.max_drawdown = np.min(drawdown)
        self.performance_metrics['max_drawdown'] = self.max_drawdown
    
    def get_analytics_report(self) -> Dict[str, Any]:
        """Get comprehensive analytics report."""
        return {
            'total_signals': self.total_signals,
            'executed_signals': self.executed_signals,
            'conversion_rates': dict(self.conversion_rates),
            'performance_metrics': dict(self.performance_metrics),
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'recent_signals': list(self.signal_history)[-10:],  # Last 10 signals
            'signal_distribution': self._get_signal_distribution()
        }
    
    def _get_signal_distribution(self) -> Dict[str, int]:
        """Get distribution of signals by type."""
        distribution = defaultdict(int)
        for signal in self.signals.values():
            distribution[signal.signal_type] += 1
        return dict(distribution)


class SignalGenerator:
    """
    Enhanced Signal Generator - Central logic layer for combining trading signals.
    
    Features:
    - Multi-asset portfolio signal generation
    - Real-time asynchronous signal streaming
    - Advanced analytics and performance tracking
    - Portfolio-level decision making
    - Signal-to-trade conversion tracking
    - Risk manager integration for pre-trade validation
    - Comprehensive logging and signal tracking
    - Configurable thresholds and weights
    """
    
    def __init__(self, 
                 stat_arb: StatisticalArbitrage = None,
                 sentiment_analyzer: SentimentAnalyzer = None,
                 risk_manager = None,
                 position_manager = None,
                 config: Dict = None):
        """
        Initialize the signal generator.
        
        Args:
            stat_arb: Statistical arbitrage strategy instance
            sentiment_analyzer: Sentiment analyzer instance
            risk_manager: Risk manager instance
            position_manager: Position manager instance
            config: Configuration dictionary
        """
        self.stat_arb = stat_arb
        self.sentiment_analyzer = sentiment_analyzer
        self.risk_manager = risk_manager
        self.position_manager = position_manager
        self.config = config or self._load_default_config()
        self.logger = logger
        
        # Signal combination parameters
        self.combination_method = CombinationMethod(self.config.get('combination_method', 'consensus'))
        self.stat_weight = self.config.get('stat_weight', 0.6)
        self.sentiment_weight = self.config.get('sentiment_weight', 0.4)
        self.min_confidence = self.config.get('min_confidence', 0.3)
        self.sentiment_thresholds = self.config.get('sentiment_thresholds', {
            'positive': 0.2,
            'negative': -0.2,
            'neutral': 0.0
        })
        
        # Portfolio parameters
        self.portfolio_rebalance_threshold = self.config.get('portfolio_rebalance_threshold', 0.1)
        self.max_portfolio_deviation = self.config.get('max_portfolio_deviation', 0.2)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        
        # Risk management parameters
        self.enable_risk_checks = self.config.get('enable_risk_checks', True)
        self.require_risk_approval = self.config.get('require_risk_approval', False)
        
        # Real-time streaming parameters
        self.enable_streaming = self.config.get('enable_streaming', False)
        self.stream_interval = self.config.get('stream_interval', 30)  # seconds
        self.signal_queue = queue.Queue(maxsize=100)
        self.streaming_active = False
        
        # Analytics
        self.analytics = SignalAnalytics()
        
        # Signal tracking
        self.generated_signals: List[TradeSignal] = []
        self.signal_history: List[Dict[str, Any]] = []
        self.portfolio_signals: List[PortfolioSignal] = []
        
        # Performance metrics
        self.total_signals = 0
        self.approved_signals = 0
        self.rejected_signals = 0
        
        # Threading
        self._streaming_thread = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info("Enhanced Signal Generator initialized")
    
    def _load_default_config(self) -> Dict:
        """Load default configuration."""
        try:
            signal_config = config.get('strategy.signal_generator', {})
            return {
                'combination_method': signal_config.get('combination_method', 'consensus'),
                'stat_weight': signal_config.get('stat_weight', 0.6),
                'sentiment_weight': signal_config.get('sentiment_weight', 0.4),
                'min_confidence': signal_config.get('min_confidence', 0.3),
                'sentiment_thresholds': signal_config.get('sentiment_thresholds', {
                    'positive': 0.2,
                    'negative': -0.2,
                    'neutral': 0.0
                }),
                'enable_risk_checks': signal_config.get('enable_risk_checks', True),
                'require_risk_approval': signal_config.get('require_risk_approval', False),
                'max_signals_per_batch': signal_config.get('max_signals_per_batch', 10),
                'signal_timeout_seconds': signal_config.get('signal_timeout_seconds', 300),
                'enable_streaming': signal_config.get('enable_streaming', False),
                'stream_interval': signal_config.get('stream_interval', 30),
                'portfolio_rebalance_threshold': signal_config.get('portfolio_rebalance_threshold', 0.1),
                'max_portfolio_deviation': signal_config.get('max_portfolio_deviation', 0.2),
                'correlation_threshold': signal_config.get('correlation_threshold', 0.7)
            }
        except Exception as e:
            self.logger.error(f"Error loading signal generator config: {e}")
            return {}
    
    def generate_signals(self, 
                        market_data: Optional[Dict] = None,
                        sentiment_data: Optional[List[SentimentData]] = None) -> List[TradeSignal]:
        """
        Generate combined trading signals from statistical arbitrage and sentiment analysis.
        
        Args:
            market_data: Market data for statistical arbitrage
            sentiment_data: Sentiment data for analysis
            
        Returns:
            List of validated trade signals
        """
        self.logger.info("Starting signal generation process")
        
        # Get statistical arbitrage signals
        stat_signals = self._get_statistical_signals()
        
        # Get sentiment signals
        sentiment_signals = self._get_sentiment_signals(sentiment_data)
        
        # Combine signals
        combined_signals = self._combine_signals(stat_signals, sentiment_signals)
        
        # Generate portfolio-level signals
        portfolio_signals = self._generate_portfolio_signals()
        
        # Validate and filter signals
        validated_signals = self._validate_signals(combined_signals)
        
        # Apply risk checks if enabled
        if self.enable_risk_checks:
            validated_signals = self._apply_risk_checks(validated_signals)
        
        # Add to analytics tracking
        for signal in validated_signals:
            self.analytics.add_signal(signal)
        
        # Log signal generation summary
        self._log_signal_summary(validated_signals)
        
        return validated_signals
    
    def start_streaming(self, callback: Optional[Callable] = None):
        """
        Start real-time signal streaming.
        
        Args:
            callback: Optional callback function for signal processing
        """
        if self.streaming_active:
            self.logger.warning("Signal streaming already active")
            return
        
        self.streaming_active = True
        self._streaming_thread = threading.Thread(
            target=self._stream_signals,
            args=(callback,),
            daemon=True
        )
        self._streaming_thread.start()
        self.logger.info("Signal streaming started")
    
    def stop_streaming(self):
        """Stop real-time signal streaming."""
        self.streaming_active = False
        if self._streaming_thread:
            self._streaming_thread.join(timeout=5)
        self.logger.info("Signal streaming stopped")
    
    def _stream_signals(self, callback: Optional[Callable] = None):
        """Internal method for signal streaming."""
        while self.streaming_active:
            try:
                # Generate signals
                signals = self.generate_signals()
                
                # Put signals in queue
                for signal in signals:
                    try:
                        self.signal_queue.put(signal, timeout=1)
                    except queue.Full:
                        self.logger.warning("Signal queue full, dropping signal")
                
                # Call callback if provided
                if callback and signals:
                    try:
                        callback(signals)
                    except Exception as e:
                        self.logger.error(f"Error in signal callback: {e}")
                
                # Wait for next interval
                time.sleep(self.stream_interval)
                
            except Exception as e:
                self.logger.error(f"Error in signal streaming: {e}")
                time.sleep(5)  # Wait before retrying
    
    def get_streaming_signals(self) -> List[TradeSignal]:
        """Get signals from the streaming queue."""
        signals = []
        while not self.signal_queue.empty():
            try:
                signal = self.signal_queue.get_nowait()
                signals.append(signal)
            except queue.Empty:
                break
        return signals
    
    def _get_statistical_signals(self) -> List[StatSignal]:
        """Get signals from statistical arbitrage strategy."""
        if not self.stat_arb:
            self.logger.warning("Statistical arbitrage strategy not available")
            return []
        
        try:
            signals = self.stat_arb.generate_signals()
            self.logger.info(f"Generated {len(signals)} statistical arbitrage signals")
            return signals
        except Exception as e:
            self.logger.error(f"Error generating statistical signals: {e}")
            return []
    
    def _get_sentiment_signals(self, sentiment_data: Optional[List[SentimentData]]) -> Dict[str, float]:
        """Get sentiment signals for relevant assets."""
        if not self.sentiment_analyzer:
            self.logger.warning("Sentiment analyzer not available")
            return {}
        
        if not sentiment_data:
            self.logger.warning("No sentiment data provided")
            return {}
        
        try:
            # Aggregate sentiment scores by asset
            sentiment_scores = self.sentiment_analyzer.aggregate_sentiment(
                sentiment_data, 
                window=self.config.get('sentiment_window', 10),
                method=self.config.get('sentiment_aggregation', 'mean')
            )
            
            # Convert to asset-specific scores
            asset_sentiment = {}
            for asset in ['BTC', 'ETH', 'SOL']:  # Add more assets as needed
                if asset in sentiment_scores.index:
                    asset_sentiment[asset] = sentiment_scores[asset]
                else:
                    asset_sentiment[asset] = 0.0
            
            self.logger.info(f"Generated sentiment scores for {len(asset_sentiment)} assets")
            return asset_sentiment
            
        except Exception as e:
            self.logger.error(f"Error generating sentiment signals: {e}")
            return {}
    
    def _combine_signals(self, 
                        stat_signals: List[StatSignal], 
                        sentiment_scores: Dict[str, float]) -> List[TradeSignal]:
        """
        Combine statistical arbitrage and sentiment signals using the configured method.
        
        Args:
            stat_signals: Statistical arbitrage signals
            sentiment_scores: Sentiment scores by asset
            
        Returns:
            List of combined trade signals
        """
        combined_signals = []
        
        for stat_signal in stat_signals:
            # Extract asset from pair (assuming format like 'BTC-ETH')
            assets = stat_signal.pair.split('-')
            if len(assets) != 2:
                self.logger.warning(f"Invalid pair format: {stat_signal.pair}")
                continue
            
            asset1, asset2 = assets[0], assets[1]
            
            # Get sentiment scores for both assets
            sentiment1 = sentiment_scores.get(asset1, 0.0)
            sentiment2 = sentiment_scores.get(asset2, 0.0)
            
            # Generate sentiment signal
            sentiment_signal = self._generate_sentiment_signal(sentiment1, sentiment2)
            
            # Combine signals based on method
            if self.combination_method == CombinationMethod.CONSENSUS:
                combined = self._combine_consensus(stat_signal, sentiment_signal)
            elif self.combination_method == CombinationMethod.WEIGHTED:
                combined = self._combine_weighted(stat_signal, sentiment_signal)
            elif self.combination_method == CombinationMethod.FILTER:
                combined = self._combine_filter(stat_signal, sentiment_signal)
            elif self.combination_method == CombinationMethod.HYBRID:
                combined = self._combine_hybrid(stat_signal, sentiment_signal)
            elif self.combination_method == CombinationMethod.PORTFOLIO_OPTIMIZED:
                combined = self._combine_portfolio_optimized(stat_signal, sentiment_signal)
            else:
                self.logger.warning(f"Unknown combination method: {self.combination_method}")
                continue
            
            if combined:
                combined_signals.append(combined)
        
        self.logger.info(f"Combined {len(combined_signals)} signals using {self.combination_method.value} method")
        return combined_signals
    
    def _generate_sentiment_signal(self, sentiment1: float, sentiment2: float) -> Dict[str, Any]:
        """Generate sentiment signal from sentiment scores."""
        # Average sentiment for the pair
        avg_sentiment = (sentiment1 + sentiment2) / 2
        
        # Determine sentiment direction
        if avg_sentiment > self.sentiment_thresholds['positive']:
            sentiment_direction = 'positive'
        elif avg_sentiment < self.sentiment_thresholds['negative']:
            sentiment_direction = 'negative'
        else:
            sentiment_direction = 'neutral'
        
        return {
            'direction': sentiment_direction,
            'score': avg_sentiment,
            'confidence': abs(avg_sentiment),
            'asset1_sentiment': sentiment1,
            'asset2_sentiment': sentiment2
        }
    
    def _combine_consensus(self, stat_signal: StatSignal, sentiment_signal: Dict[str, Any]) -> Optional[TradeSignal]:
        """Combine signals using consensus logic - both must agree."""
        # Map stat signal to sentiment direction
        if stat_signal.signal_type == 'entry_long':
            stat_direction = 'positive'
        elif stat_signal.signal_type == 'entry_short':
            stat_direction = 'negative'
        else:
            # Exit signals don't need sentiment consensus
            return self._create_trade_signal(stat_signal, sentiment_signal, 'consensus')
        
        # Check consensus
        if stat_direction == sentiment_signal['direction']:
            return self._create_trade_signal(stat_signal, sentiment_signal, 'consensus')
        else:
            self.logger.debug(f"No consensus: stat={stat_direction}, sentiment={sentiment_signal['direction']}")
            return None
    
    def _combine_weighted(self, stat_signal: StatSignal, sentiment_signal: Dict[str, Any]) -> Optional[TradeSignal]:
        """Combine signals using weighted average."""
        # Calculate weighted confidence
        weighted_confidence = (
            self.stat_weight * stat_signal.confidence +
            self.sentiment_weight * sentiment_signal['confidence']
        )
        
        # Only generate signal if weighted confidence meets minimum threshold
        if weighted_confidence >= self.min_confidence:
            return self._create_trade_signal(stat_signal, sentiment_signal, 'weighted', weighted_confidence)
        else:
            self.logger.debug(f"Weighted confidence {weighted_confidence:.3f} below threshold {self.min_confidence}")
            return None
    
    def _combine_filter(self, stat_signal: StatSignal, sentiment_signal: Dict[str, Any]) -> Optional[TradeSignal]:
        """Use sentiment as a filter for statistical signals."""
        # Strong negative sentiment blocks long positions
        if (stat_signal.signal_type == 'entry_long' and 
            sentiment_signal['direction'] == 'negative' and 
            abs(sentiment_signal['score']) > 0.3):
            self.logger.debug("Sentiment filter blocked long position")
            return None
        
        # Strong positive sentiment blocks short positions
        if (stat_signal.signal_type == 'entry_short' and 
            sentiment_signal['direction'] == 'positive' and 
            abs(sentiment_signal['score']) > 0.3):
            self.logger.debug("Sentiment filter blocked short position")
            return None
        
        # Adjust confidence based on sentiment alignment
        if stat_signal.signal_type == 'entry_long' and sentiment_signal['direction'] == 'positive':
            adjusted_confidence = min(1.0, stat_signal.confidence * 1.2)
        elif stat_signal.signal_type == 'entry_short' and sentiment_signal['direction'] == 'negative':
            adjusted_confidence = min(1.0, stat_signal.confidence * 1.2)
        else:
            adjusted_confidence = stat_signal.confidence * 0.8
        
        return self._create_trade_signal(stat_signal, sentiment_signal, 'filter', adjusted_confidence)
    
    def _combine_hybrid(self, stat_signal: StatSignal, sentiment_signal: Dict[str, Any]) -> Optional[TradeSignal]:
        """Combine signals using hybrid approach (consensus + weighted)."""
        # First check consensus
        consensus_signal = self._combine_consensus(stat_signal, sentiment_signal)
        if consensus_signal:
            return consensus_signal
        
        # If no consensus, try weighted approach
        return self._combine_weighted(stat_signal, sentiment_signal)
    
    def _combine_portfolio_optimized(self, stat_signal: StatSignal, sentiment_signal: Dict[str, Any]) -> Optional[TradeSignal]:
        """Combine signals using portfolio-optimized approach."""
        # This is a placeholder for a more sophisticated portfolio optimization
        # In a real scenario, you would use a portfolio optimization library
        # to determine the optimal allocation based on multiple signals.
        # For now, we'll just use a weighted average of confidence.
        
        # Calculate weighted confidence
        weighted_confidence = (
            self.stat_weight * stat_signal.confidence +
            self.sentiment_weight * sentiment_signal['confidence']
        )
        
        # Only generate signal if weighted confidence meets minimum threshold
        if weighted_confidence >= self.min_confidence:
            return self._create_trade_signal(stat_signal, sentiment_signal, 'portfolio_optimized', weighted_confidence)
        else:
            self.logger.debug(f"Portfolio optimized confidence {weighted_confidence:.3f} below threshold {self.min_confidence}")
            return None
    
    def _generate_portfolio_signals(self) -> List[PortfolioSignal]:
        """Generate portfolio-level signals for multi-asset decisions."""
        if not self.position_manager:
            return []
        
        try:
            # Get current portfolio metrics
            portfolio_metrics = self.position_manager.get_portfolio_metrics()
            
            # Check for rebalancing opportunities
            rebalance_signals = self._check_rebalancing_needs(portfolio_metrics)
            
            # Check for hedging opportunities
            hedge_signals = self._check_hedging_needs(portfolio_metrics)
            
            # Check for diversification opportunities
            diversify_signals = self._check_diversification_needs(portfolio_metrics)
            
            return rebalance_signals + hedge_signals + diversify_signals
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio signals: {e}")
            return []
    
    def _check_rebalancing_needs(self, portfolio_metrics) -> List[PortfolioSignal]:
        """Check if portfolio rebalancing is needed."""
        signals = []
        
        # Calculate target allocation based on current performance
        current_allocation = portfolio_metrics.exposure_by_asset
        total_value = portfolio_metrics.total_market_value
        
        if total_value == 0:
            return signals
        
        # Normalize allocation
        normalized_allocation = {
            asset: value / total_value 
            for asset, value in current_allocation.items()
        }
        
        # Calculate target allocation (equal weight for now)
        target_allocation = {}
        num_assets = len(normalized_allocation)
        if num_assets > 0:
            equal_weight = 1.0 / num_assets
            target_allocation = {asset: equal_weight for asset in normalized_allocation.keys()}
        
        # Check if rebalancing is needed
        max_deviation = 0.0
        for asset in normalized_allocation:
            deviation = abs(normalized_allocation[asset] - target_allocation.get(asset, 0))
            max_deviation = max(max_deviation, deviation)
        
        if max_deviation > self.portfolio_rebalance_threshold:
            # Generate rebalancing actions
            rebalance_actions = []
            for asset in normalized_allocation:
                current_weight = normalized_allocation[asset]
                target_weight = target_allocation.get(asset, 0)
                
                if abs(current_weight - target_weight) > 0.01:  # 1% threshold
                    if current_weight > target_weight:
                        action = {
                            'asset': asset,
                            'action': 'reduce',
                            'current_weight': current_weight,
                            'target_weight': target_weight,
                            'adjustment': target_weight - current_weight
                        }
                    else:
                        action = {
                            'asset': asset,
                            'action': 'increase',
                            'current_weight': current_weight,
                            'target_weight': target_weight,
                            'adjustment': target_weight - current_weight
                        }
                    rebalance_actions.append(action)
            
            if rebalance_actions:
                signal = PortfolioSignal(
                    portfolio_id="main_portfolio",
                    signal_type="rebalance",
                    target_allocation=target_allocation,
                    current_allocation=normalized_allocation,
                    rebalance_actions=rebalance_actions,
                    confidence=min(0.8, max_deviation),  # Higher deviation = higher confidence
                    metadata={
                        'max_deviation': max_deviation,
                        'threshold': self.portfolio_rebalance_threshold,
                        'total_value': total_value
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _check_hedging_needs(self, portfolio_metrics) -> List[PortfolioSignal]:
        """Check if portfolio hedging is needed."""
        signals = []
        
        # Check for high correlation exposure
        exposure_by_asset = portfolio_metrics.exposure_by_asset
        total_value = portfolio_metrics.total_market_value
        
        if total_value == 0:
            return signals
        
        # Calculate correlation-based risk
        high_correlation_assets = self._identify_correlated_assets(exposure_by_asset)
        
        if high_correlation_assets:
            # Generate hedging signal
            hedge_actions = []
            for asset_group in high_correlation_assets:
                total_exposure = sum(exposure_by_asset.get(asset, 0) for asset in asset_group)
                if total_exposure > total_value * self.max_portfolio_deviation:
                    # Suggest hedging with negatively correlated asset
                    hedge_asset = self._find_hedge_asset(asset_group)
                    if hedge_asset:
                        action = {
                            'hedge_assets': asset_group,
                            'hedge_with': hedge_asset,
                            'exposure': total_exposure,
                            'suggested_hedge_size': total_exposure * 0.5  # 50% hedge
                        }
                        hedge_actions.append(action)
            
            if hedge_actions:
                signal = PortfolioSignal(
                    portfolio_id="main_portfolio",
                    signal_type="hedge",
                    target_allocation={},  # Will be calculated based on hedge actions
                    current_allocation=exposure_by_asset,
                    rebalance_actions=hedge_actions,
                    confidence=0.7,
                    metadata={
                        'correlation_threshold': self.correlation_threshold,
                        'hedge_actions': hedge_actions
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _check_diversification_needs(self, portfolio_metrics) -> List[PortfolioSignal]:
        """Check if portfolio diversification is needed."""
        signals = []
        
        # Check sector concentration
        exposure_by_sector = portfolio_metrics.exposure_by_sector
        total_value = portfolio_metrics.total_market_value
        
        if total_value == 0:
            return signals
        
        # Check for over-concentration in any sector
        max_sector_exposure = max(exposure_by_sector.values()) if exposure_by_sector else 0
        sector_limit = total_value * 0.4  # 40% max per sector
        
        if max_sector_exposure > sector_limit:
            # Generate diversification signal
            over_concentrated_sectors = [
                sector for sector, exposure in exposure_by_sector.items()
                if exposure > sector_limit
            ]
            
            diversify_actions = []
            for sector in over_concentrated_sectors:
                action = {
                    'sector': sector,
                    'current_exposure': exposure_by_sector[sector],
                    'target_exposure': sector_limit,
                    'reduction_needed': exposure_by_sector[sector] - sector_limit
                }
                diversify_actions.append(action)
            
            if diversify_actions:
                signal = PortfolioSignal(
                    portfolio_id="main_portfolio",
                    signal_type="diversify",
                    target_allocation={},  # Will be calculated based on diversification
                    current_allocation=exposure_by_sector,
                    rebalance_actions=diversify_actions,
                    confidence=0.8,
                    metadata={
                        'sector_limit': sector_limit,
                        'diversify_actions': diversify_actions
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _identify_correlated_assets(self, exposure_by_asset: Dict[str, float]) -> List[List[str]]:
        """Identify groups of highly correlated assets."""
        # This is a simplified implementation
        # In practice, you would calculate actual correlations from price data
        
        # Define known correlated asset groups
        correlated_groups = [
            ['BTC', 'ETH'],  # Major cryptocurrencies
            ['SOL', 'AVAX'],  # Layer 1 alternatives
            ['USDT', 'USDC'],  # Stablecoins
        ]
        
        # Check which groups have significant exposure
        high_exposure_groups = []
        for group in correlated_groups:
            group_exposure = sum(exposure_by_asset.get(asset, 0) for asset in group)
            if group_exposure > 0:  # Has some exposure
                high_exposure_groups.append(group)
        
        return high_exposure_groups
    
    def _find_hedge_asset(self, asset_group: List[str]) -> Optional[str]:
        """Find a suitable hedge asset for the given asset group."""
        # Simplified hedge asset mapping
        hedge_mapping = {
            'BTC': 'USDT',
            'ETH': 'USDT',
            'SOL': 'USDT',
            'AVAX': 'USDT',
        }
        
        # Return the first available hedge asset
        for asset in asset_group:
            if asset in hedge_mapping:
                return hedge_mapping[asset]
        
        return None
    
    def _create_trade_signal(self, 
                           stat_signal: StatSignal, 
                           sentiment_signal: Dict[str, Any],
                           combination_method: str,
                           confidence: Optional[float] = None) -> TradeSignal:
        """Create a standardized trade signal."""
        # Determine side based on signal type
        if stat_signal.signal_type == 'entry_long':
            side = 'buy'
        elif stat_signal.signal_type == 'entry_short':
            side = 'sell'
        elif stat_signal.signal_type == 'exit':
            # For exit signals, determine side based on current position
            side = 'sell'  # Default to sell for exits
        else:
            self.logger.warning(f"Unknown signal type: {stat_signal.signal_type}")
            side = 'buy'
        
        # Use provided confidence or calculate from components
        if confidence is None:
            confidence = (stat_signal.confidence + sentiment_signal['confidence']) / 2
        
        # Create metadata
        metadata = {
            'z_score': stat_signal.z_score,
            'spread_value': stat_signal.spread_value,
            'sentiment_score': sentiment_signal['score'],
            'sentiment_direction': sentiment_signal['direction'],
            'combination_method': combination_method,
            'stat_confidence': stat_signal.confidence,
            'sentiment_confidence': sentiment_signal['confidence']
        }
        
        # Create sources list
        sources = [SignalSource.STATISTICAL_ARB.value, SignalSource.SENTIMENT.value]
        
        return TradeSignal(
            symbol=stat_signal.pair,
            side=side,
            quantity=stat_signal.size1,  # Use size from stat signal
            order_type='market',  # Default to market orders
            price=None,
            confidence=confidence,
            sources=sources,
            metadata=metadata,
            timestamp=datetime.now(),
            signal_type=stat_signal.signal_type,
            risk_checked=False,
            scope=SignalScope.SINGLE_ASSET
        )
    
    def _validate_signals(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Validate and filter signals."""
        validated_signals = []
        
        for signal in signals:
            if signal.validate():
                validated_signals.append(signal)
            else:
                self.logger.warning(f"Invalid signal rejected: {signal}")
        
        self.logger.info(f"Validated {len(validated_signals)} out of {len(signals)} signals")
        return validated_signals
    
    def _apply_risk_checks(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Apply risk manager checks to signals."""
        if not self.risk_manager:
            self.logger.warning("Risk manager not available, skipping risk checks")
            return signals
        
        approved_signals = []
        
        for signal in signals:
            try:
                # Check with risk manager
                allowed, reason = self.risk_manager.check_order_risk(signal)
                
                if allowed:
                    signal.risk_checked = True
                    approved_signals.append(signal)
                    self.logger.debug(f"Risk check passed for {signal.symbol}")
                else:
                    self.logger.warning(f"Risk check failed for {signal.symbol}: {reason}")
                    # If risk check fails, still allow signal if not required
                    if not self.require_risk_approval:
                        approved_signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"Error during risk check for {signal.symbol}: {e}")
                # If risk check fails, still allow signal if not required
                if not self.require_risk_approval:
                    approved_signals.append(signal)
        
        self.logger.info(f"Risk manager approved {len(approved_signals)} out of {len(signals)} signals")
        return approved_signals
    
    def _log_signal_summary(self, signals: List[TradeSignal]):
        """Log summary of generated signals."""
        self.total_signals += len(signals)
        
        if signals:
            # Group by side
            buy_signals = [s for s in signals if s.side == 'buy']
            sell_signals = [s for s in signals if s.side == 'sell']
            
            # Calculate average confidence
            avg_confidence = np.mean([s.confidence for s in signals])
            
            self.logger.info(f"Signal generation complete:")
            self.logger.info(f"  - Total signals: {len(signals)}")
            self.logger.info(f"  - Buy signals: {len(buy_signals)}")
            self.logger.info(f"  - Sell signals: {len(sell_signals)}")
            self.logger.info(f"  - Average confidence: {avg_confidence:.3f}")
            
            # Log individual signals
            for signal in signals:
                self.logger.debug(f"Generated signal: {signal.symbol} {signal.side} "
                                f"qty={signal.quantity:.4f} conf={signal.confidence:.3f}")
        else:
            self.logger.info("No signals generated in this batch")
    
    def get_signal_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get signal history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_signals = [s for s in self.generated_signals if s.timestamp >= cutoff_time]
        
        history = []
        for signal in recent_signals:
            history.append({
                'symbol': signal.symbol,
                'side': signal.side,
                'quantity': signal.quantity,
                'confidence': signal.confidence,
                'timestamp': signal.timestamp.isoformat(),
                'sources': signal.sources,
                'risk_checked': signal.risk_checked
            })
        
        return history
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get signal generator performance metrics."""
        return {
            'total_signals': self.total_signals,
            'approved_signals': self.approved_signals,
            'rejected_signals': self.rejected_signals,
            'approval_rate': self.approved_signals / self.total_signals if self.total_signals > 0 else 0,
            'combination_method': self.combination_method.value,
            'min_confidence': self.min_confidence
        }
    
    def update_config(self, new_config: Dict):
        """Update signal generator configuration."""
        self.config.update(new_config)
        
        # Update instance variables
        if 'combination_method' in new_config:
            self.combination_method = CombinationMethod(new_config['combination_method'])
        if 'stat_weight' in new_config:
            self.stat_weight = new_config['stat_weight']
        if 'sentiment_weight' in new_config:
            self.sentiment_weight = new_config['sentiment_weight']
        if 'min_confidence' in new_config:
            self.min_confidence = new_config['min_confidence']
        if 'sentiment_thresholds' in new_config:
            self.sentiment_thresholds.update(new_config['sentiment_thresholds'])
        
        self.logger.info("Signal generator configuration updated")
    
    def reset(self):
        """Reset signal generator state."""
        self.generated_signals.clear()
        self.signal_history.clear()
        self.total_signals = 0
        self.approved_signals = 0
        self.rejected_signals = 0
        self.logger.info("Signal generator state reset")

    def get_analytics_report(self) -> Dict[str, Any]:
        """Get comprehensive analytics report."""
        return self.analytics.get_analytics_report()
    
    def update_signal_execution(self, signal_id: str, executed: bool, 
                               execution_price: Optional[float] = None):
        """Update signal execution status for analytics."""
        self.analytics.update_signal_execution(signal_id, executed, execution_price)
    
    def close_signal(self, signal_id: str, exit_price: float, 
                    exit_time: Optional[datetime] = None):
        """Close a signal and update performance analytics."""
        self.analytics.close_signal(signal_id, exit_price, exit_time)


def create_signal_generator(stat_arb: StatisticalArbitrage = None,
                          sentiment_analyzer: SentimentAnalyzer = None,
                          risk_manager = None,
                          config: Dict = None) -> SignalGenerator:
    """
    Factory function to create a signal generator instance.
    
    Args:
        stat_arb: Statistical arbitrage strategy instance
        sentiment_analyzer: Sentiment analyzer instance
        risk_manager: Risk manager instance
        config: Configuration dictionary
        
    Returns:
        Configured SignalGenerator instance
    """
    return SignalGenerator(
        stat_arb=stat_arb,
        sentiment_analyzer=sentiment_analyzer,
        risk_manager=risk_manager,
        config=config
    )