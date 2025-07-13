"""
Tests for the Data Integration Module

Tests the integration between statistical arbitrage strategy and data collector.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from strategy.data_integration import StrategyDataIntegration, create_strategy_integration


class TestStrategyDataIntegration(unittest.TestCase):
    """Test the StrategyDataIntegration class."""
    
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
        
        with patch('strategy.data_integration.DataCollector'), \
             patch('strategy.data_integration.logger'):
            self.integration = StrategyDataIntegration(self.config)
    
    def test_initialization(self):
        """Test integration initialization."""
        self.assertIsNotNone(self.integration.strategy)
        self.assertIsNotNone(self.integration.data_collector)
        self.assertFalse(self.integration.running)
        self.assertEqual(len(self.integration.symbols), 3)  # Default symbols
        self.assertEqual(len(self.integration.exchanges), 2)  # Default exchanges
    
    def test_extract_assets_from_symbols(self):
        """Test asset extraction from trading pairs."""
        assets = self.integration._extract_assets_from_symbols()
        expected_assets = ['ETH', 'SOL', 'BTC']
        self.assertEqual(set(assets), set(expected_assets))
    
    def test_validate_price(self):
        """Test price validation."""
        # Valid price
        self.assertTrue(self.integration._validate_price(100.0, 'ETH'))
        
        # Invalid price (negative)
        self.assertFalse(self.integration._validate_price(-100.0, 'ETH'))
        
        # Invalid price (zero)
        self.assertFalse(self.integration._validate_price(0.0, 'ETH'))
        
        # Test extreme price change
        self.integration.strategy.update_price_data('ETH', 100.0, datetime.now())
        # Small change - should be valid
        self.assertTrue(self.integration._validate_price(105.0, 'ETH'))
        # Large change - should be invalid
        self.assertFalse(self.integration._validate_price(200.0, 'ETH'))
    
    def test_update_strategy_with_dataframe(self):
        """Test updating strategy with DataFrame data."""
        # Create test DataFrame
        timestamps = pd.date_range(start=datetime.now() - timedelta(hours=10), 
                                 end=datetime.now(), freq='1H')
        data = {
            'timestamp': timestamps,
            'open': [100 + i for i in range(len(timestamps))],
            'high': [101 + i for i in range(len(timestamps))],
            'low': [99 + i for i in range(len(timestamps))],
            'close': [100.5 + i for i in range(len(timestamps))],
            'volume': [1000 + i * 10 for i in range(len(timestamps))]
        }
        df = pd.DataFrame(data)
        
        # Update strategy
        self.integration._update_strategy_with_dataframe(df, 'ETH/USDT', 'binance')
        
        # Check that data was added
        self.assertIn('ETH', self.integration.strategy.price_data)
        self.assertGreater(len(self.integration.strategy.price_data['ETH']), 0)
    
    def test_update_strategy_with_empty_dataframe(self):
        """Test updating strategy with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        # Should not raise an error
        self.integration._update_strategy_with_dataframe(empty_df, 'ETH/USDT', 'binance')
        
        # Strategy should not have data for this asset
        self.assertNotIn('ETH', self.integration.strategy.price_data)
    
    def test_collect_and_update_market_data(self):
        """Test market data collection and strategy update."""
        # Mock market data
        mock_market_data = Mock()
        mock_market_data.price = 2000.0
        mock_market_data.timestamp = datetime.now()
        mock_market_data.validate.return_value = True
        
        # Mock data collector
        self.integration.data_collector.get_latest_market_data = Mock(return_value=mock_market_data)
        
        # Test collection
        self.integration._collect_and_update_market_data()
        
        # Check that data was added to strategy
        self.assertIn('ETH', self.integration.strategy.price_data)
    
    def test_update_position_pnls(self):
        """Test position PnL updates."""
        # Create a mock position
        from strategy.stat_arb import Position
        position = Position(
            pair='ETH_SOL',
            asset1='ETH',
            asset2='SOL',
            size1=1.0,
            size2=-10.0,
            entry_price1=2000.0,
            entry_price2=100.0,
            entry_time=datetime.now(),
            current_pnl=0.0,
            status='open'
        )
        
        # Add position to strategy
        self.integration.strategy.positions['ETH_SOL'] = position
        
        # Add price data
        current_time = datetime.now()
        self.integration.strategy.update_price_data('ETH', 2100.0, current_time)
        self.integration.strategy.update_price_data('SOL', 90.0, current_time)
        
        # Update PnL
        self.integration._update_position_pnls()
        
        # Check that PnL was updated
        self.assertNotEqual(position.current_pnl, 0.0)
    
    def test_get_strategy_status(self):
        """Test getting strategy status."""
        status = self.integration.get_strategy_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('running', status)
        self.assertIn('pairs_count', status)
        self.assertIn('positions_count', status)
        self.assertIn('signals_count', status)
        self.assertIn('performance', status)
        self.assertIn('last_update', status)
    
    def test_get_open_positions(self):
        """Test getting open positions."""
        positions = self.integration.get_open_positions()
        self.assertIsInstance(positions, dict)
    
    def test_get_recent_signals(self):
        """Test getting recent signals."""
        signals = self.integration.get_recent_signals(hours=24)
        self.assertIsInstance(signals, list)
    
    def test_force_signal_generation(self):
        """Test forcing signal generation."""
        signals = self.integration.force_signal_generation()
        self.assertIsInstance(signals, list)
    
    @patch('strategy.data_integration.DataCollector')
    def test_start_and_stop(self, mock_data_collector):
        """Test starting and stopping the integration."""
        # Mock data collector
        mock_collector = Mock()
        mock_data_collector.return_value = mock_collector
        
        # Create integration
        integration = StrategyDataIntegration(self.config)
        
        # Test start
        integration.start()
        self.assertTrue(integration.running)
        mock_collector.start.assert_called_once()
        
        # Test stop
        integration.stop()
        self.assertFalse(integration.running)
        mock_collector.stop.assert_called_once()
    
    def test_initialize_strategy_with_historical_data(self):
        """Test strategy initialization with historical data."""
        # Mock historical data
        timestamps = pd.date_range(start=datetime.now() - timedelta(days=5), end=datetime.now(), freq='1H')
        n = len(timestamps)
        mock_df = pd.DataFrame({
            'timestamp': timestamps,
            'open': [100 + i for i in range(n)],
            'high': [101 + i for i in range(n)],
            'low': [99 + i for i in range(n)],
            'close': [100.5 + i for i in range(n)],
            'volume': [1000 + i * 10 for i in range(n)]
        })
        
        # Mock data collector
        self.integration.data_collector.get_historical_data = Mock(return_value=mock_df)
        
        # Test initialization
        self.integration._initialize_strategy_with_historical_data()
        
        # Check that data was loaded
        self.assertGreater(len(self.integration.strategy.price_data), 0)


class TestCreateStrategyIntegration(unittest.TestCase):
    """Test the convenience function."""
    
    def test_create_integration_with_config(self):
        """Test creating integration with custom config."""
        config = {'z_score_threshold': 1.5}
        integration = create_strategy_integration(config)
        self.assertEqual(integration.strategy.z_score_threshold, 1.5)
    
    def test_create_integration_without_config(self):
        """Test creating integration without config."""
        integration = create_strategy_integration()
        self.assertIsInstance(integration, StrategyDataIntegration)


if __name__ == '__main__':
    unittest.main() 