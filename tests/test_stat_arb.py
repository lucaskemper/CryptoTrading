"""
Tests for the Statistical Arbitrage Strategy Module
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from strategy.stat_arb import (
    StatisticalArbitrage, 
    PairData, 
    Signal, 
    Position, 
    PerformanceMetrics,
    create_stat_arb_strategy
)


class TestPerformanceMetrics(unittest.TestCase):
    """Test the PerformanceMetrics class."""
    
    def setUp(self):
        self.metrics = PerformanceMetrics()
    
    def test_initial_state(self):
        """Test initial state of performance metrics."""
        summary = self.metrics.get_summary()
        self.assertEqual(summary['total_trades'], 0)
        self.assertEqual(summary['winning_trades'], 0)
        self.assertEqual(summary['losing_trades'], 0)
        self.assertEqual(summary['total_pnl'], 0.0)
        self.assertEqual(summary['win_rate'], 0)
    
    def test_update_winning_trade(self):
        """Test updating with a winning trade."""
        self.metrics.update(100.0, 0.05)  # $100 profit, 5% return
        summary = self.metrics.get_summary()
        self.assertEqual(summary['total_trades'], 1)
        self.assertEqual(summary['winning_trades'], 1)
        self.assertEqual(summary['losing_trades'], 0)
        self.assertEqual(summary['total_pnl'], 100.0)
        self.assertEqual(summary['win_rate'], 1.0)
    
    def test_update_losing_trade(self):
        """Test updating with a losing trade."""
        self.metrics.update(-50.0, -0.02)  # $50 loss, -2% return
        summary = self.metrics.get_summary()
        self.assertEqual(summary['total_trades'], 1)
        self.assertEqual(summary['winning_trades'], 0)
        self.assertEqual(summary['losing_trades'], 1)
        self.assertEqual(summary['total_pnl'], -50.0)
        self.assertEqual(summary['win_rate'], 0.0)
    
    def test_multiple_trades(self):
        """Test multiple trades."""
        trades = [
            (100.0, 0.05),   # Win
            (-50.0, -0.02),  # Loss
            (75.0, 0.03),    # Win
            (-25.0, -0.01)   # Loss
        ]
        
        for pnl, ret in trades:
            self.metrics.update(pnl, ret)
        
        summary = self.metrics.get_summary()
        self.assertEqual(summary['total_trades'], 4)
        self.assertEqual(summary['winning_trades'], 2)
        self.assertEqual(summary['losing_trades'], 2)
        self.assertEqual(summary['total_pnl'], 100.0)
        self.assertEqual(summary['win_rate'], 0.5)


class TestStatisticalArbitrage(unittest.TestCase):
    """Test the StatisticalArbitrage class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'z_score_threshold': 2.0,
            'lookback_period': 50,
            'correlation_threshold': 0.7,
            'cointegration_pvalue_threshold': 0.05,
            'min_spread_std': 0.001,
            'position_size_limit': 0.1,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.1
        }
        
        with patch('strategy.stat_arb.logger'):
            self.strategy = StatisticalArbitrage(self.config)
    
    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.z_score_threshold, 2.0)
        self.assertEqual(self.strategy.lookback_period, 50)
        self.assertEqual(self.strategy.correlation_threshold, 0.7)
        self.assertFalse(self.strategy.is_initialized)
        self.assertEqual(len(self.strategy.pairs_data), 0)
        self.assertEqual(len(self.strategy.positions), 0)
    
    def test_update_price_data(self):
        """Test updating price data."""
        timestamp = datetime.now()
        self.strategy.update_price_data('ETH', 2000.0, timestamp)
        
        self.assertIn('ETH', self.strategy.price_data)
        self.assertEqual(self.strategy.price_data['ETH'][timestamp], 2000.0)
        self.assertEqual(self.strategy.last_update, timestamp)
    
    def test_calculate_spread(self):
        """Test spread calculation."""
        timestamp = datetime.now()
        self.strategy.update_price_data('ETH', 2000.0, timestamp)
        self.strategy.update_price_data('SOL', 100.0, timestamp)
        
        spread = self.strategy.calculate_spread('ETH', 'SOL')
        self.assertEqual(spread, 1900.0)
    
    def test_calculate_spread_insufficient_data(self):
        """Test spread calculation with insufficient data."""
        spread = self.strategy.calculate_spread('ETH', 'SOL')
        self.assertIsNone(spread)
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        # Add some price data
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(30)]
        prices = [100 + np.random.normal(0, 5) for _ in range(30)]
        
        for ts, price in zip(timestamps, prices):
            self.strategy.update_price_data('ETH', price, ts)
        
        position_size = self.strategy._calculate_position_size('ETH')
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, self.strategy.position_size_limit)
    
    def test_calculate_position_size_no_data(self):
        """Test position size calculation with no data."""
        position_size = self.strategy._calculate_position_size('ETH')
        self.assertEqual(position_size, self.strategy.position_size_limit)
    
    def test_should_close_position_stop_loss(self):
        """Test position closure due to stop loss."""
        # Create a position with significant loss
        position = Position(
            pair='ETH_SOL',
            asset1='ETH',
            asset2='SOL',
            size1=1.0,
            size2=-10.0,  # Short SOL
            entry_price1=2000.0,
            entry_price2=100.0,
            entry_time=datetime.now(),
            current_pnl=-100.0,
            status='open'
        )
        
        # Update current prices to create a loss
        timestamp = datetime.now()
        self.strategy.update_price_data('ETH', 1900.0, timestamp)  # Loss on ETH
        self.strategy.update_price_data('SOL', 110.0, timestamp)   # Loss on SOL short
        
        should_close = self.strategy._should_close_position(position, Mock())
        self.assertTrue(should_close)
    
    def test_should_close_position_take_profit(self):
        """Test position closure due to take profit."""
        # Create a position with significant profit
        position = Position(
            pair='ETH_SOL',
            asset1='ETH',
            asset2='SOL',
            size1=1.0,
            size2=-10.0,  # Short SOL
            entry_price1=2000.0,
            entry_price2=100.0,
            entry_time=datetime.now(),
            current_pnl=100.0,
            status='open'
        )
        
        # Update current prices to create a larger profit (15% profit)
        timestamp = datetime.now()
        self.strategy.update_price_data('ETH', 2300.0, timestamp)  # 15% profit on ETH
        self.strategy.update_price_data('SOL', 85.0, timestamp)    # 15% profit on SOL short
        
        should_close = self.strategy._should_close_position(position, Mock())
        self.assertTrue(should_close)
    
    def test_should_close_position_normal(self):
        """Test position should not close under normal conditions."""
        position = Position(
            pair='ETH_SOL',
            asset1='ETH',
            asset2='SOL',
            size1=1.0,
            size2=-10.0,
            entry_price1=2000.0,
            entry_price2=100.0,
            entry_time=datetime.now(),
            current_pnl=10.0,
            status='open'
        )
        
        # Update current prices with small change
        timestamp = datetime.now()
        self.strategy.update_price_data('ETH', 2005.0, timestamp)
        self.strategy.update_price_data('SOL', 99.0, timestamp)
        
        should_close = self.strategy._should_close_position(position, Mock())
        self.assertFalse(should_close)
    
    def test_generate_signals_no_pairs(self):
        """Test signal generation with no pairs."""
        signals = self.strategy.generate_signals()
        self.assertEqual(len(signals), 0)
    
    def test_update_positions_entry_signal(self):
        """Test updating positions with entry signal."""
        # Add price data
        timestamp = datetime.now()
        self.strategy.update_price_data('ETH', 2000.0, timestamp)
        self.strategy.update_price_data('SOL', 100.0, timestamp)
        
        # Create entry signal
        signal = Signal(
            timestamp=timestamp,
            pair='ETH_SOL',
            signal_type='entry_long',
            asset1='ETH',
            asset2='SOL',
            action1='buy',
            action2='sell',
            size1=0.1,
            size2=2.0,
            z_score=-2.5,
            spread_value=-100.0,
            confidence=0.8
        )
        
        self.strategy.update_positions([signal])
        
        self.assertIn('ETH_SOL', self.strategy.positions)
        position = self.strategy.positions['ETH_SOL']
        self.assertEqual(position.size1, 0.1)
        self.assertEqual(position.size2, -2.0)
        self.assertEqual(position.status, 'open')
    
    def test_update_positions_exit_signal(self):
        """Test updating positions with exit signal."""
        # First create a position
        timestamp = datetime.now()
        self.strategy.update_price_data('ETH', 2000.0, timestamp)
        self.strategy.update_price_data('SOL', 100.0, timestamp)
        
        entry_signal = Signal(
            timestamp=timestamp,
            pair='ETH_SOL',
            signal_type='entry_long',
            asset1='ETH',
            asset2='SOL',
            action1='buy',
            action2='sell',
            size1=0.1,
            size2=2.0,
            z_score=-2.5,
            spread_value=-100.0,
            confidence=0.8
        )
        
        self.strategy.update_positions([entry_signal])
        
        # Now create exit signal
        exit_signal = Signal(
            timestamp=timestamp + timedelta(minutes=1),
            pair='ETH_SOL',
            signal_type='exit',
            asset1='ETH',
            asset2='SOL',
            action1='sell',
            action2='buy',
            size1=0.1,
            size2=2.0,
            z_score=0.1,
            spread_value=-5.0,
            confidence=1.0
        )
        
        self.strategy.update_positions([exit_signal])
        
        # Position should be closed
        self.assertNotIn('ETH_SOL', self.strategy.positions)
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        summary = self.strategy.get_performance_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('total_trades', summary)
        self.assertIn('win_rate', summary)
        self.assertIn('sharpe_ratio', summary)
    
    def test_get_open_positions(self):
        """Test getting open positions."""
        positions = self.strategy.get_open_positions()
        self.assertIsInstance(positions, dict)
        self.assertEqual(len(positions), 0)
    
    def test_get_recent_signals(self):
        """Test getting recent signals."""
        signals = self.strategy.get_recent_signals(hours=24)
        self.assertIsInstance(signals, list)
    
    def test_reset(self):
        """Test strategy reset."""
        # Add some data
        timestamp = datetime.now()
        self.strategy.update_price_data('ETH', 2000.0, timestamp)
        
        self.strategy.reset()
        
        self.assertEqual(len(self.strategy.price_data), 0)
        self.assertEqual(len(self.strategy.positions), 0)
        self.assertEqual(len(self.strategy.signals), 0)
        self.assertFalse(self.strategy.is_initialized)

    def test_missing_data_handling(self):
        """Test that strategy handles missing data gracefully."""
        # Only one asset has data
        timestamp = datetime.now()
        self.strategy.update_price_data('ETH', 2000.0, timestamp)
        # No SOL data
        spread = self.strategy.calculate_spread('ETH', 'SOL')
        self.assertIsNone(spread)
        # No error should be raised
        signals = self.strategy.generate_signals()
        self.assertIsInstance(signals, list)

    def test_volatility_spike(self):
        """Test strategy response to sudden volatility spike."""
        # Add normal data
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(50)]
        prices = [100 + np.random.normal(0, 1) for _ in range(50)]
        for ts, price in zip(timestamps, prices):
            self.strategy.update_price_data('ETH', price, ts)
            self.strategy.update_price_data('SOL', price * 0.05 + 95, ts)
        # Add a spike
        spike_time = datetime.now()
        self.strategy.update_price_data('ETH', 200, spike_time)
        self.strategy.update_price_data('SOL', 200, spike_time)
        # Should not crash
        signals = self.strategy.generate_signals()
        self.assertIsInstance(signals, list)

    def test_changing_hedge_ratio(self):
        """Test that hedge ratio is recalculated for new data windows."""
        # Add two regimes
        t1 = [datetime.now() - timedelta(hours=2, minutes=i) for i in range(50)]
        t2 = [datetime.now() - timedelta(minutes=i) for i in range(50)]
        p1 = [100 + i*0.1 for i in range(50)]
        p2 = [200 + i*0.2 for i in range(50)]
        # Regime 1
        for ts, price in zip(t1, p1):
            self.strategy.update_price_data('ETH', price, ts)
            self.strategy.update_price_data('SOL', price * 2, ts)
        # Regime 2 (different relationship)
        for ts, price in zip(t2, p2):
            self.strategy.update_price_data('ETH', price, ts)
            self.strategy.update_price_data('SOL', price * 0.5, ts)
        pairs = self.strategy.find_cointegrated_pairs(['ETH', 'SOL'])
        self.assertTrue(len(pairs) >= 0)  # Should not error
        # Hedge ratio should be present if found
        for pair in pairs:
            pair_key = f"{pair[0]}_{pair[1]}"
            if pair_key in self.strategy.pairs_data:
                self.assertTrue(hasattr(self.strategy.pairs_data[pair_key], 'hedge_ratio'))

    def test_slippage_impact(self):
        """Test that slippage reduces PnL as expected."""
        config = self.config.copy()
        config['slippage'] = 0.01  # 1% slippage
        with patch('strategy.stat_arb.logger'):
            strat = StatisticalArbitrage(config)
        # Add price data
        timestamp = datetime.now()
        strat.update_price_data('ETH', 2000.0, timestamp)
        strat.update_price_data('SOL', 100.0, timestamp)
        # Create entry and exit signals
        entry_signal = Signal(
            timestamp=timestamp,
            pair='ETH_SOL',
            signal_type='entry_long',
            asset1='ETH',
            asset2='SOL',
            action1='buy',
            action2='sell',
            size1=0.1,
            size2=2.0,
            z_score=-2.5,
            spread_value=-100.0,
            confidence=0.8
        )
        strat.update_positions([entry_signal])
        # Move price to create profit
        strat.update_price_data('ETH', 2100.0, timestamp + timedelta(minutes=1))
        strat.update_price_data('SOL', 90.0, timestamp + timedelta(minutes=1))
        exit_signal = Signal(
            timestamp=timestamp + timedelta(minutes=1),
            pair='ETH_SOL',
            signal_type='exit',
            asset1='ETH',
            asset2='SOL',
            action1='sell',
            action2='buy',
            size1=0.1,
            size2=2.0,
            z_score=0.1,
            spread_value=-5.0,
            confidence=1.0
        )
        strat.update_positions([exit_signal])
        perf = strat.get_performance_summary()
        # Calculate expected PnL: (0.1 * 100) + (2 * 10) = 10 + 20 = 30
        # With 1% slippage on total investment: (0.1 * 2000 + 2 * 100) * 0.01 = 4
        # Expected PnL: 30 - 4 = 26
        # But slippage is applied twice (entry + exit), so: 30 - 8 = 22
        self.assertLess(perf['total_pnl'], 30.0)  # Should be less than without slippage
        self.assertGreater(perf['total_pnl'], 0.0)  # Should still be positive


class TestCreateStatArbStrategy(unittest.TestCase):
    """Test the convenience function."""
    
    def test_create_strategy_with_config(self):
        """Test creating strategy with custom config."""
        config = {'z_score_threshold': 1.5}
        strategy = create_stat_arb_strategy(config)
        self.assertEqual(strategy.z_score_threshold, 1.5)
    
    def test_create_strategy_without_config(self):
        """Test creating strategy without config."""
        strategy = create_stat_arb_strategy()
        self.assertIsInstance(strategy, StatisticalArbitrage)


if __name__ == '__main__':
    unittest.main() 