"""
Statistical Arbitrage Strategy Module

This module implements statistical arbitrage strategies for crypto trading,
including pair selection, spread calculation, signal generation, and risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from scipy import stats
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
import logging

from src.utils.logger import logger
from src.utils.config_loader import config

@dataclass
class PairData:
    """Data structure for asset pair information."""
    asset1: str
    asset2: str
    correlation: float
    cointegration_pvalue: float
    is_cointegrated: bool
    spread_mean: float
    spread_std: float
    hedge_ratio: float  # <-- NEW: OLS hedge ratio
    last_update: datetime


@dataclass
class Signal:
    """Trade signal data structure."""
    timestamp: datetime
    pair: str
    signal_type: str  # 'entry_long', 'entry_short', 'exit'
    asset1: str
    asset2: str
    action1: str  # 'buy' or 'sell'
    action2: str  # 'buy' or 'sell'
    size1: float
    size2: float
    z_score: float
    spread_value: float
    confidence: float


@dataclass
class Position:
    """Position tracking data structure."""
    pair: str
    asset1: str
    asset2: str
    size1: float
    size2: float
    entry_price1: float
    entry_price2: float
    entry_time: datetime
    current_pnl: float
    status: str  # 'open', 'closed'


class PerformanceMetrics:
    """Performance tracking for statistical arbitrage strategy."""
    
    def __init__(self):
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.returns = []
        self.trade_history = []
    
    def update(self, trade_pnl: float, trade_return: float):
        """Update performance metrics with new trade."""
        self.total_trades += 1
        self.total_pnl += trade_pnl
        self.returns.append(trade_return)
        
        if trade_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        if len(self.returns) > 1:
            returns_array = np.array(self.returns)
            self.sharpe_ratio = np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = np.cumprod(1 + np.array(self.returns))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        current_max_drawdown = np.min(drawdown)
        self.max_drawdown = min(self.max_drawdown, current_max_drawdown)
    
    def get_summary(self) -> Dict[str, float]:
        """Get performance summary."""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio
        }


class StatisticalArbitrage:
    """
    Statistical Arbitrage Strategy Implementation
    
    Features:
    - OLS hedge ratio spread calculation (default)
    - Dynamic z-score thresholds (optional, based on volatility or performance)
    - Multiple spread models (future: Kalman filter, rolling OLS)
    - Order routing stub for execution integration
    - Simulated slippage for backtesting (configurable)
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the statistical arbitrage strategy.
        
        Args:
            config: Configuration dictionary with strategy parameters
            Supported config keys:
                - z_score_threshold: float or 'dynamic'
                - dynamic_threshold_window: int (rolling window for dynamic threshold)
                - spread_model: 'ols' (default), 'kalman', 'rolling_ols'
                - slippage: float (fractional, e.g. 0.001 for 0.1% per trade)
        """
        self.config = config or self._load_default_config()
        self.logger = logger
        
        # Strategy parameters
        self.z_score_threshold = self.config.get('z_score_threshold', 2.0)
        self.dynamic_threshold_window = self.config.get('dynamic_threshold_window', 100)
        self.spread_model = self.config.get('spread_model', 'ols')
        self.slippage = self.config.get('slippage', 0.0)
        self.lookback_period = self.config.get('lookback_period', 100)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.cointegration_pvalue_threshold = self.config.get('cointegration_pvalue_threshold', 0.05)
        self.min_spread_std = self.config.get('min_spread_std', 0.001)
        self.position_size_limit = self.config.get('position_size_limit', 0.1)
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.05)
        self.take_profit_pct = self.config.get('take_profit_pct', 0.1)
        
        # Data storage
        self.pairs_data: Dict[str, PairData] = {}
        self.price_data: Dict[str, pd.Series] = {}
        self.spread_data: Dict[str, pd.Series] = {}
        self.positions: Dict[str, Position] = {}
        self.signals: List[Signal] = []
        
        # Performance tracking
        self.performance = PerformanceMetrics()
        
        # State tracking
        self.is_initialized = False
        self.last_update = None
        
        self.logger.info("Statistical Arbitrage strategy initialized")
    
    def _load_default_config(self) -> Dict:
        """Load default configuration from config file."""
        try:
            strategy_config = config.get('strategy.statistical_arbitrage', {})
            return {
                'z_score_threshold': strategy_config.get('z_score_threshold', 2.0),
                'lookback_period': strategy_config.get('cointegration_lookback', 100),
                'correlation_threshold': 0.7,
                'cointegration_pvalue_threshold': 0.05,
                'min_spread_std': 0.001,
                'position_size_limit': 0.1,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.1,
                'max_positions': 5,
                'rebalance_frequency': 300  # seconds
            }
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
    
    def update_price_data(self, asset: str, price: float, timestamp: datetime):
        """
        Update price data for an asset.
        
        Args:
            asset: Asset symbol
            price: Current price
            timestamp: Timestamp of the price
        """
        if asset not in self.price_data:
            self.price_data[asset] = pd.Series(dtype=float)
        
        self.price_data[asset][timestamp] = price
        
        # Keep only recent data
        cutoff_time = timestamp - timedelta(days=30)
        self.price_data[asset] = self.price_data[asset][self.price_data[asset].index >= cutoff_time]
        
        self.last_update = timestamp
    
    def find_cointegrated_pairs(self, assets: List[str]) -> List[Tuple[str, str]]:
        """
        Find cointegrated pairs from a list of assets.
        
        Args:
            assets: List of asset symbols
            
        Returns:
            List of cointegrated asset pairs
        """
        cointegrated_pairs = []
        
        for i, asset1 in enumerate(assets):
            for asset2 in assets[i+1:]:
                if asset1 not in self.price_data or asset2 not in self.price_data:
                    continue
                
                # Align price data
                price1 = self.price_data[asset1]
                price2 = self.price_data[asset2]
                
                # Find common timestamps
                common_times = price1.index.intersection(price2.index)
                if len(common_times) < self.lookback_period:
                    continue
                
                price1_aligned = price1[common_times]
                price2_aligned = price2[common_times]
                
                # Calculate correlation
                correlation = price1_aligned.corr(price2_aligned)
                
                if abs(correlation) < self.correlation_threshold:
                    continue
                
                # Test for cointegration
                try:
                    score, pvalue, _ = coint(price1_aligned, price2_aligned)
                    
                    if pvalue < self.cointegration_pvalue_threshold:
                        # Calculate hedge ratio using OLS regression
                        ols_model = OLS(price1_aligned.values, price2_aligned.values)
                        ols_result = ols_model.fit()
                        hedge_ratio = ols_result.params[0]
                        # Calculate spread using hedge ratio
                        spread = price1_aligned - hedge_ratio * price2_aligned
                        spread_mean = spread.mean()
                        spread_std = spread.std()
                        
                        if spread_std > self.min_spread_std:
                            pair_key = f"{asset1}_{asset2}"
                            self.pairs_data[pair_key] = PairData(
                                asset1=asset1,
                                asset2=asset2,
                                correlation=correlation,
                                cointegration_pvalue=pvalue,
                                is_cointegrated=True,
                                spread_mean=spread_mean,
                                spread_std=spread_std,
                                hedge_ratio=hedge_ratio,
                                last_update=datetime.now()
                            )
                            cointegrated_pairs.append((asset1, asset2))
                            self.logger.info(f"Found cointegrated pair: {asset1}-{asset2} "
                                           f"(correlation: {correlation:.3f}, p-value: {pvalue:.3f}, hedge_ratio: {hedge_ratio:.4f})")
                
                except Exception as e:
                    self.logger.warning(f"Error testing cointegration for {asset1}-{asset2}: {e}")
        
        return cointegrated_pairs
    
    def calculate_spread(self, asset1: str, asset2: str) -> Optional[float]:
        """
        Calculate the current spread between two assets using the hedge ratio.
        
        Args:
            asset1: First asset symbol
            asset2: Second asset symbol
            
        Returns:
            Current spread value or None if insufficient data
        """
        if asset1 not in self.price_data or asset2 not in self.price_data:
            return None
        
        price1 = self.price_data[asset1]
        price2 = self.price_data[asset2]
        
        # Get latest prices
        if price1.empty or price2.empty:
            return None
        
        latest_price1 = price1.iloc[-1]
        latest_price2 = price2.iloc[-1]
        
        # Find pair_key and hedge ratio
        pair_key = f"{asset1}_{asset2}"
        hedge_ratio = 1.0
        if pair_key in self.pairs_data:
            hedge_ratio = self.pairs_data[pair_key].hedge_ratio
        
        return latest_price1 - hedge_ratio * latest_price2
    
    def calculate_z_score(self, pair_key: str) -> Optional[float]:
        """
        Calculate the z-score for a pair's spread.
        
        Args:
            pair_key: Pair identifier
            
        Returns:
            Z-score value or None if insufficient data
        """
        if pair_key not in self.pairs_data:
            return None
        
        pair_data = self.pairs_data[pair_key]
        current_spread = self.calculate_spread(pair_data.asset1, pair_data.asset2)
        
        if current_spread is None:
            return None
        
        z_score = (current_spread - pair_data.spread_mean) / pair_data.spread_std
        return z_score
    
    def _get_z_score_threshold(self, pair_key: str) -> float:
        """
        Get the z-score threshold for a pair (static or dynamic).
        """
        if self.z_score_threshold == 'dynamic':
            # Use rolling volatility to adapt threshold
            if pair_key in self.pairs_data:
                pair_data = self.pairs_data[pair_key]
                asset1, asset2 = pair_data.asset1, pair_data.asset2
                price1 = self.price_data.get(asset1)
                price2 = self.price_data.get(asset2)
                if price1 is not None and price2 is not None:
                    common_times = price1.index.intersection(price2.index)
                    if len(common_times) >= self.dynamic_threshold_window:
                        # Use rolling std of spread as threshold
                        hedge_ratio = pair_data.hedge_ratio
                        spread = price1[common_times] - hedge_ratio * price2[common_times]
                        rolling_std = spread.rolling(self.dynamic_threshold_window).std().iloc[-1]
                        return max(1.0, 2.0 * rolling_std / (pair_data.spread_std or 1e-6))
            return 2.0  # fallback
        else:
            return float(self.z_score_threshold)
    
    def generate_signals(self) -> List[Signal]:
        """
        Generate trading signals based on current market conditions.
        
        Returns:
            List of trading signals
        """
        signals = []
        current_time = datetime.now()
        
        for pair_key, pair_data in self.pairs_data.items():
            z_score = self.calculate_z_score(pair_key)
            
            if z_score is None:
                continue
            
            threshold = self._get_z_score_threshold(pair_key)
            # Check if we have an open position for this pair
            has_position = pair_key in self.positions
            
            # Generate entry signals
            if not has_position:
                if z_score > threshold:
                    # Asset1 is overvalued relative to Asset2
                    signal = Signal(
                        timestamp=current_time,
                        pair=pair_key,
                        signal_type='entry_short',
                        asset1=pair_data.asset1,
                        asset2=pair_data.asset2,
                        action1='sell',
                        action2='buy',
                        size1=self._calculate_position_size(pair_data.asset1),
                        size2=self._calculate_position_size(pair_data.asset2),
                        z_score=z_score,
                        spread_value=self.calculate_spread(pair_data.asset1, pair_data.asset2),
                        confidence=min(abs(z_score) / threshold, 1.0)
                    )
                    signals.append(signal)
                    self.logger.info(f"Generated short signal for {pair_key} (z-score: {z_score:.3f}, threshold: {threshold:.3f})")
                
                elif z_score < -threshold:
                    # Asset1 is undervalued relative to Asset2
                    signal = Signal(
                        timestamp=current_time,
                        pair=pair_key,
                        signal_type='entry_long',
                        asset1=pair_data.asset1,
                        asset2=pair_data.asset2,
                        action1='buy',
                        action2='sell',
                        size1=self._calculate_position_size(pair_data.asset1),
                        size2=self._calculate_position_size(pair_data.asset2),
                        z_score=z_score,
                        spread_value=self.calculate_spread(pair_data.asset1, pair_data.asset2),
                        confidence=min(abs(z_score) / threshold, 1.0)
                    )
                    signals.append(signal)
                    self.logger.info(f"Generated long signal for {pair_key} (z-score: {z_score:.3f}, threshold: {threshold:.3f})")
            
            # Generate exit signals for open positions
            elif has_position:
                position = self.positions[pair_key]
                
                # Exit if spread has mean-reverted
                if abs(z_score) < 0.5:  # Close to mean
                    signal = Signal(
                        timestamp=current_time,
                        pair=pair_key,
                        signal_type='exit',
                        asset1=pair_data.asset1,
                        asset2=pair_data.asset2,
                        action1='sell' if position.size1 > 0 else 'buy',
                        action2='buy' if position.size2 > 0 else 'sell',
                        size1=abs(position.size1),
                        size2=abs(position.size2),
                        z_score=z_score,
                        spread_value=self.calculate_spread(pair_data.asset1, pair_data.asset2),
                        confidence=1.0
                    )
                    signals.append(signal)
                    self.logger.info(f"Generated exit signal for {pair_key} (z-score: {z_score:.3f})")
                
                # Check stop-loss and take-profit
                elif self._should_close_position(position, pair_data):
                    signal = Signal(
                        timestamp=current_time,
                        pair=pair_key,
                        signal_type='exit',
                        asset1=pair_data.asset1,
                        asset2=pair_data.asset2,
                        action1='sell' if position.size1 > 0 else 'buy',
                        action2='buy' if position.size2 > 0 else 'sell',
                        size1=abs(position.size1),
                        size2=abs(position.size2),
                        z_score=z_score,
                        spread_value=self.calculate_spread(pair_data.asset1, pair_data.asset2),
                        confidence=1.0
                    )
                    signals.append(signal)
                    self.logger.info(f"Generated risk management exit signal for {pair_key}")
        
        self.signals.extend(signals)
        return signals
    
    def route_signal_to_execution(self, signal, order_manager=None):
        """
        Route a trading signal to the order execution module.

        Args:
            signal: Signal object to be executed
            order_manager: Optional order manager instance with a submit_order(signal) method

        Returns:
            Execution result or None

        Expected order_manager interface:
            - submit_order(signal: Signal) -> dict or OrderResult
            - Handles real-world trading, slippage, partial fills, and order status
        """
        if order_manager is not None and hasattr(order_manager, 'submit_order'):
            return order_manager.submit_order(signal)
        else:
            self.logger.info(f"[ROUTING STUB] Would route signal to execution: {signal}")
            return None
    
    def _calculate_position_size(self, asset: str) -> float:
        """
        Calculate position size based on volatility and risk limits.
        
        Args:
            asset: Asset symbol
            
        Returns:
            Position size as fraction of portfolio
        """
        if asset not in self.price_data:
            return self.position_size_limit
        
        # Calculate volatility
        prices = self.price_data[asset]
        if len(prices) < 20:
            return self.position_size_limit
        
        returns = prices.pct_change().dropna()
        volatility = returns.std()
        
        # Adjust position size based on volatility
        # Higher volatility = smaller position size
        volatility_factor = max(0.1, 1 - volatility * 10)
        position_size = self.position_size_limit * volatility_factor
        
        return min(position_size, self.position_size_limit)
    
    def _should_close_position(self, position: Position, pair_data: PairData) -> bool:
        """
        Check if position should be closed due to risk management rules.
        
        Args:
            position: Current position
            pair_data: Pair data
            
        Returns:
            True if position should be closed
        """
        current_prices = {}
        for asset in [position.asset1, position.asset2]:
            if asset in self.price_data and not self.price_data[asset].empty:
                current_prices[asset] = self.price_data[asset].iloc[-1]
        
        if len(current_prices) != 2:
            return False
        
        # Calculate current PnL
        pnl1 = (current_prices[position.asset1] - position.entry_price1) * position.size1
        pnl2 = (current_prices[position.asset2] - position.entry_price2) * position.size2
        total_pnl = pnl1 + pnl2
        total_pnl_pct = total_pnl / (abs(position.entry_price1 * position.size1) + 
                                    abs(position.entry_price2 * position.size2))
        
        # Check stop-loss
        if total_pnl_pct < -self.stop_loss_pct:
            return True
        
        # Check take-profit
        if total_pnl_pct > self.take_profit_pct:
            return True
        
        return False
    
    def update_positions(self, signals: List[Signal]):
        """
        Update positions based on executed signals.
        Applies slippage to PnL if configured.
        
        Args:
            signals: List of executed signals
        """
        for signal in signals:
            pair_key = signal.pair
            
            if signal.signal_type.startswith('entry'):
                # Open new position
                position = Position(
                    pair=pair_key,
                    asset1=signal.asset1,
                    asset2=signal.asset2,
                    size1=signal.size1 if signal.action1 == 'buy' else -signal.size1,
                    size2=signal.size2 if signal.action2 == 'buy' else -signal.size2,
                    entry_price1=self.price_data[signal.asset1].iloc[-1],
                    entry_price2=self.price_data[signal.asset2].iloc[-1],
                    entry_time=signal.timestamp,
                    current_pnl=0.0,
                    status='open'
                )
                self.positions[pair_key] = position
                self.logger.info(f"Opened position for {pair_key}")
            
            elif signal.signal_type == 'exit':
                # Close position
                if pair_key in self.positions:
                    position = self.positions[pair_key]
                    
                    # Calculate final PnL
                    current_prices = {}
                    for asset in [position.asset1, position.asset2]:
                        if asset in self.price_data and not self.price_data[asset].empty:
                            current_prices[asset] = self.price_data[asset].iloc[-1]
                    
                    if len(current_prices) == 2:
                        pnl1 = (current_prices[position.asset1] - position.entry_price1) * position.size1
                        pnl2 = (current_prices[position.asset2] - position.entry_price2) * position.size2
                        final_pnl = pnl1 + pnl2
                        
                        # Apply slippage (simulate cost on both entry and exit)
                        total_investment = (abs(position.entry_price1 * position.size1) + 
                                          abs(position.entry_price2 * position.size2))
                        slippage_cost = self.slippage * total_investment
                        final_pnl -= slippage_cost
                        
                        trade_return = final_pnl / total_investment if total_investment > 0 else 0
                        
                        self.performance.update(final_pnl, trade_return)
                        
                        self.logger.info(f"Closed position for {pair_key}, PnL: {final_pnl:.4f} (slippage: {slippage_cost:.4f})")
                    
                    del self.positions[pair_key]
    
    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get performance summary for the strategy.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.performance.get_summary()
    
    def get_open_positions(self) -> Dict[str, Position]:
        """
        Get currently open positions.
        
        Returns:
            Dictionary of open positions
        """
        return self.positions.copy()
    
    def get_recent_signals(self, hours: int = 24) -> List[Signal]:
        """
        Get recent signals within specified time window.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent signals
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [signal for signal in self.signals if signal.timestamp >= cutoff_time]
    
    def reset(self):
        """Reset the strategy state."""
        self.pairs_data.clear()
        self.price_data.clear()
        self.spread_data.clear()
        self.positions.clear()
        self.signals.clear()
        self.performance = PerformanceMetrics()
        self.is_initialized = False
        self.logger.info("Statistical arbitrage strategy reset")


# Convenience function for creating strategy instance
def create_stat_arb_strategy(config: Dict = None) -> StatisticalArbitrage:
    """
    Create a new statistical arbitrage strategy instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        StatisticalArbitrage instance
    """
    return StatisticalArbitrage(config)

# NOTE: For edge case testing, see tests/test_stat_arb.py for coverage of:
# - Missing data
# - Sudden volatility spikes
# - Changing hedge ratios
# - Slippage impact
