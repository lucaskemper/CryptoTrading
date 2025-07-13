"""
Tests for the Backtesting Module

Comprehensive tests for the backtesting framework including data simulation,
strategy execution, and performance analysis.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtesting.backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from backtesting.data_simulator import DataSimulator
from backtesting.performance_analyzer import PerformanceAnalyzer
from backtesting.strategy_runner import StrategyRunner


class TestBacktestConfig(unittest.TestCase):
    """Test the BacktestConfig dataclass."""
    
    def test_config_creation(self):
        """Test creating a backtest configuration."""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            symbols=['ETH/USDT', 'SOL/USDT'],
            initial_capital=100000.0
        )
        
        self.assertEqual(config.start_date, start_date)
        self.assertEqual(config.end_date, end_date)
        self.assertEqual(config.symbols, ['ETH/USDT', 'SOL/USDT'])
        self.assertEqual(config.initial_capital, 100000.0)
        self.assertEqual(config.data_frequency, '1h')  # Default value
        self.assertTrue(config.track_metrics)  # Default value


class TestDataSimulator(unittest.TestCase):
    """Test the DataSimulator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BacktestConfig(
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
            symbols=['ETH/USDT', 'SOL/USDT'],
            initial_capital=100000.0
        )
        self.simulator = DataSimulator(self.config)
    
    def test_initialization(self):
        """Test data simulator initialization."""
        self.assertIsNotNone(self.simulator)
        self.assertEqual(self.simulator.config, self.config)
        self.assertEqual(len(self.simulator.market_data), 0)
    
    def test_generate_market_data(self):
        """Test market data generation."""
        self.simulator._generate_market_data('ETH/USDT')
        
        self.assertIn('ETH/USDT', self.simulator.market_data)
        data = self.simulator.market_data['ETH/USDT']
        
        # Check data structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('timestamp', data.columns)
        self.assertIn('open', data.columns)
        self.assertIn('high', data.columns)
        self.assertIn('low', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)
        
        # Check data quality
        self.assertTrue(len(data) > 0)
        self.assertTrue((data['close'] > 0).all())
        self.assertTrue((data['volume'] >= 0).all())
    
    def test_generate_price_series(self):
        """Test price series generation."""
        timestamps = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            periods=100
        )
        
        price_series = self.simulator._generate_price_series('ETH/USDT', timestamps)
        
        self.assertIsInstance(price_series, pd.Series)
        self.assertEqual(len(price_series), len(timestamps))
        self.assertTrue((price_series > 0).all())
    
    def test_generate_volume_series(self):
        """Test volume series generation."""
        timestamps = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            periods=100
        )
        
        volume_series = self.simulator._generate_volume_series(timestamps)
        
        self.assertIsInstance(volume_series, pd.Series)
        self.assertEqual(len(volume_series), len(timestamps))
        self.assertTrue((volume_series >= 0).all())
    
    def test_generate_sentiment_data(self):
        """Test sentiment data generation."""
        self.simulator.market_data['ETH/USDT'] = pd.DataFrame({
            'timestamp': pd.date_range(self.config.start_date, self.config.end_date, periods=10),
            'close': np.random.uniform(1000, 3000, 10)
        })
        
        self.simulator._generate_sentiment_data()
        
        self.assertIn('overall', self.simulator.sentiment_data)
        sentiment_data = self.simulator.sentiment_data['overall']
        
        self.assertIsInstance(sentiment_data, pd.DataFrame)
        self.assertIn('timestamp', sentiment_data.columns)
        self.assertIn('sentiment_score', sentiment_data.columns)
        self.assertIn('confidence', sentiment_data.columns)
    
    def test_get_data_iterator(self):
        """Test data iterator creation."""
        # Generate some data first
        self.simulator._generate_market_data('ETH/USDT')
        self.simulator._generate_sentiment_data()
        self.simulator._initialize_data_iterator()
        
        iterator = self.simulator.get_data_iterator()
        
        # Test first few items
        count = 0
        for timestamp, market_data, sentiment_data in iterator:
            self.assertIsInstance(timestamp, datetime)
            self.assertIsInstance(market_data, dict)
            count += 1
            if count >= 5:  # Test first 5 items
                break
        
        self.assertTrue(count > 0)


class TestPerformanceAnalyzer(unittest.TestCase):
    """Test the PerformanceAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = PerformanceAnalyzer()
        
        # Create sample data
        self.equity_curve = pd.Series(
            [100000, 101000, 102000, 101500, 103000],
            index=pd.date_range('2023-01-01', periods=5, freq='D'),
            name='portfolio_value'
        )
        
        self.trade_history = [
            {'timestamp': '2023-01-01', 'pnl': 100, 'symbol': 'ETH/USDT'},
            {'timestamp': '2023-01-02', 'pnl': 200, 'symbol': 'SOL/USDT'},
            {'timestamp': '2023-01-03', 'pnl': -50, 'symbol': 'ETH/USDT'},
            {'timestamp': '2023-01-04', 'pnl': 150, 'symbol': 'BTC/USDT'}
        ]
        
        self.position_history = [
            {'timestamp': '2023-01-01', 'positions': {'ETH/USDT': {'quantity': 1}}},
            {'timestamp': '2023-01-02', 'positions': {'ETH/USDT': {'quantity': 1}, 'SOL/USDT': {'quantity': 2}}}
        ]
        
        self.signal_history = [
            {'timestamp': '2023-01-01', 'signal': {'strategy': 'stat_arb'}},
            {'timestamp': '2023-01-02', 'signal': {'strategy': 'sentiment'}}
        ]
    
    def test_initialization(self):
        """Test performance analyzer initialization."""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.risk_free_rate, 0.02)
    
    def test_analyze_performance(self):
        """Test performance analysis."""
        results = self.analyzer.analyze_performance(
            self.equity_curve,
            self.trade_history,
            self.position_history,
            self.signal_history
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('basic_metrics', results)
        self.assertIn('risk_metrics', results)
        self.assertIn('trade_analysis', results)
    
    def test_calculate_basic_metrics(self):
        """Test basic metrics calculation."""
        self.analyzer._calculate_basic_metrics(self.equity_curve)
        
        self.assertIn('total_return', self.analyzer.metrics)
        self.assertIn('annualized_return', self.analyzer.metrics)
        self.assertIn('volatility', self.analyzer.metrics)
        self.assertIn('max_drawdown', self.analyzer.metrics)
    
    def test_calculate_risk_metrics(self):
        """Test risk metrics calculation."""
        self.analyzer._calculate_risk_metrics(self.equity_curve)
        
        self.assertIn('sharpe_ratio', self.analyzer.risk_metrics)
        self.assertIn('sortino_ratio', self.analyzer.risk_metrics)
        self.assertIn('calmar_ratio', self.analyzer.risk_metrics)
        self.assertIn('var_95', self.analyzer.risk_metrics)
    
    def test_analyze_trades(self):
        """Test trade analysis."""
        self.analyzer._analyze_trades(self.trade_history)
        
        self.assertIn('total_trades', self.analyzer.trade_analysis)
        self.assertIn('winning_trades', self.analyzer.trade_analysis)
        self.assertIn('losing_trades', self.analyzer.trade_analysis)
        self.assertIn('win_rate', self.analyzer.trade_analysis)
        self.assertIn('profit_factor', self.analyzer.trade_analysis)
    
    def test_generate_report(self):
        """Test report generation."""
        results = self.analyzer.analyze_performance(
            self.equity_curve,
            self.trade_history,
            self.position_history,
            self.signal_history
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = self.analyzer.generate_report(results, temp_dir)
            
            self.assertIsInstance(report_path, str)
            self.assertTrue(Path(report_path).exists())
    
    def test_generate_plots(self):
        """Test plot generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_paths = self.analyzer.generate_plots(
                self.equity_curve,
                self.trade_history,
                temp_dir
            )
            
            self.assertIsInstance(plot_paths, list)
            self.assertTrue(len(plot_paths) > 0)
            
            for plot_path in plot_paths:
                self.assertTrue(Path(plot_path).exists())


class TestStrategyRunner(unittest.TestCase):
    """Test the StrategyRunner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BacktestConfig(
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
            symbols=['ETH/USDT', 'SOL/USDT'],
            initial_capital=100000.0
        )
        
        with patch('backtesting.strategy_runner.StatisticalArbitrage'), \
             patch('backtesting.strategy_runner.SentimentAnalyzer'), \
             patch('backtesting.strategy_runner.SignalGenerator'), \
             patch('backtesting.strategy_runner.RiskManager'), \
             patch('backtesting.strategy_runner.PositionManager'):
            self.runner = StrategyRunner(self.config)
    
    def test_initialization(self):
        """Test strategy runner initialization."""
        self.assertIsNotNone(self.runner)
        self.assertEqual(self.runner.config, self.config)
        self.assertEqual(self.runner.initial_capital, 100000.0)
        self.assertEqual(self.runner.max_position_size, 0.1)
    
    def test_initialize_strategies(self):
        """Test strategy initialization."""
        with patch('backtesting.strategy_runner.StatisticalArbitrage'), \
             patch('backtesting.strategy_runner.SentimentAnalyzer'), \
             patch('backtesting.strategy_runner.SignalGenerator'), \
             patch('backtesting.strategy_runner.RiskManager'), \
             patch('backtesting.strategy_runner.PositionManager'):
            
            self.runner.initialize_strategies()
            
            self.assertIsNotNone(self.runner.stat_arb)
            self.assertIsNotNone(self.runner.signal_generator)
            self.assertIsNotNone(self.runner.risk_manager)
            self.assertIsNotNone(self.runner.position_manager)
    
    def test_update_market_data(self):
        """Test market data update."""
        market_data = {
            'ETH/USDT': {'close': 2000.0, 'timestamp': datetime.now()},
            'SOL/USDT': {'close': 100.0, 'timestamp': datetime.now()}
        }
        timestamp = datetime.now()
        
        self.runner.update_market_data(market_data, timestamp)
        
        # Test that the method doesn't raise exceptions
        self.assertTrue(True)
    
    def test_generate_signals(self):
        """Test signal generation."""
        market_data = {
            'ETH/USDT': {'close': 2000.0},
            'SOL/USDT': {'close': 100.0}
        }
        
        # Mock strategy components
        self.runner.stat_arb = Mock()
        self.runner.stat_arb.generate_signals.return_value = [
            {'symbol': 'ETH/USDT', 'side': 'buy', 'confidence': 0.8}
        ]
        
        self.runner.signal_generator = Mock()
        self.runner.signal_generator.combine_signals.return_value = [
            {'symbol': 'ETH/USDT', 'side': 'buy', 'confidence': 0.8}
        ]
        
        signals = self.runner.generate_signals(market_data)
        
        self.assertIsInstance(signals, list)
        self.assertTrue(len(signals) > 0)
    
    def test_validate_signals(self):
        """Test signal validation."""
        signals = [
            {'symbol': 'ETH/USDT', 'side': 'buy', 'confidence': 0.8},
            {'symbol': 'SOL/USDT', 'side': 'sell', 'confidence': 0.6}
        ]
        
        # Mock risk manager
        self.runner.risk_manager = Mock()
        self.runner.risk_manager.validate_signal.return_value = True
        
        validated_signals = self.runner.validate_signals(signals)
        
        self.assertEqual(len(validated_signals), len(signals))
    
    def test_execute_trades(self):
        """Test trade execution."""
        signals = [
            {'symbol': 'ETH/USDT', 'side': 'buy', 'confidence': 0.8}
        ]
        
        market_data = {
            'ETH/USDT': {'close': 2000.0}
        }
        
        timestamp = datetime.now()
        
        executed_trades = self.runner.execute_trades(signals, market_data, timestamp)
        
        self.assertIsInstance(executed_trades, list)
        if len(executed_trades) > 0:
            trade = executed_trades[0]
            self.assertIn('timestamp', trade)
            self.assertIn('symbol', trade)
            self.assertIn('side', trade)
            self.assertIn('quantity', trade)
            self.assertIn('price', trade)
    
    def test_get_performance_metrics(self):
        """Test performance metrics calculation."""
        # Add some trade history
        self.runner.trade_history = [
            {'pnl': 100},
            {'pnl': -50},
            {'pnl': 200}
        ]
        
        metrics = self.runner.get_performance_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_trades', metrics)
        self.assertIn('winning_trades', metrics)
        self.assertIn('losing_trades', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('total_pnl', metrics)


class TestBacktestEngine(unittest.TestCase):
    """Test the BacktestEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BacktestConfig(
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
            symbols=['ETH/USDT', 'SOL/USDT'],
            initial_capital=100000.0
        )
        
        with patch('backtesting.backtest_engine.DataSimulator'), \
             patch('backtesting.backtest_engine.PerformanceAnalyzer'), \
             patch('backtesting.backtest_engine.StrategyRunner'):
            self.engine = BacktestEngine(self.config)
    
    def test_initialization(self):
        """Test backtest engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.config, self.config)
        self.assertEqual(self.engine.current_portfolio_value, 100000.0)
        self.assertEqual(self.engine.peak_portfolio_value, 100000.0)
    
    def test_initialize_strategies(self):
        """Test strategy initialization."""
        with patch('backtesting.backtest_engine.StatisticalArbitrage'), \
             patch('backtesting.backtest_engine.SentimentAnalyzer'), \
             patch('backtesting.backtest_engine.SignalGenerator'), \
             patch('backtesting.backtest_engine.RiskManager'), \
             patch('backtesting.backtest_engine.PositionManager'):
            
            self.engine._initialize_strategies()
            
            self.assertIsNotNone(self.engine.stat_arb)
            self.assertIsNotNone(self.engine.signal_generator)
            self.assertIsNotNone(self.engine.risk_manager)
            self.assertIsNotNone(self.engine.position_manager)
    
    def test_run_backtest(self):
        """Test running a complete backtest."""
        # Mock all components
        with patch('backtesting.backtest_engine.DataSimulator') as mock_simulator, \
             patch('backtesting.backtest_engine.PerformanceAnalyzer') as mock_analyzer, \
             patch('backtesting.backtest_engine.StrategyRunner') as mock_runner:
            
            # Mock data simulator
            mock_simulator_instance = Mock()
            mock_simulator_instance.initialize.return_value = None
            mock_simulator_instance.get_data_iterator.return_value = iter([
                (datetime.now(), {'ETH/USDT': {'close': 2000}}, None)
            ])
            mock_simulator.return_value = mock_simulator_instance
            
            # Mock strategy runner
            mock_runner_instance = Mock()
            mock_runner_instance.generate_signals.return_value = []
            mock_runner_instance.validate_signals.return_value = []
            mock_runner_instance.execute_trades.return_value = []
            mock_runner.return_value = mock_runner_instance
            
            # Mock performance analyzer
            mock_analyzer_instance = Mock()
            mock_analyzer_instance.analyze_performance.return_value = {}
            mock_analyzer.return_value = mock_analyzer_instance
            
            # Run backtest
            results = self.engine.run_backtest()
            
            self.assertIsInstance(results, BacktestResult)


if __name__ == '__main__':
    unittest.main() 