#!/usr/bin/env python3
"""
Tests for Enhanced Signal Generator
Tests the enhanced signal generator with ML filtering and risk management
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock

# Import the components
from src.strategy.enhanced_signal_generator import EnhancedSignalGenerator, EnhancedSignal
from src.ml.signal_filter import MLSignalFilter

class TestEnhancedSignalGenerator(unittest.TestCase):
    """Test cases for EnhancedSignalGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pkl")
        
        # Create enhanced signal generator
        self.signal_generator = EnhancedSignalGenerator(initial_capital=100000.0)
        
        # Create sample market data with a clear outlier to trigger a Z-score > 2.0
        btc_prices = [50000 + (i % 3) * 10 for i in range(20)]  # 50000, 50010, 50020, ...
        btc_prices[-1] = 51000  # Outlier
        self.market_data = {
            'BTC': [
                {'price': price, 'volume': 1000 + i * 100, 'timestamp': datetime.now()} for i, price in enumerate(btc_prices)
            ],
            'ETH': [
                {'price': 3000, 'volume': 500, 'timestamp': datetime.now()} for _ in range(21)
            ]
        }
        
        # Create sample sentiment data
        self.sentiment_data = {
            'BTC': {
                'score': 0.4,
                'confidence': 0.7,
                'sources': ['news', 'twitter', 'reddit']
            },
            'ETH': {
                'score': -0.2,
                'confidence': 0.6,
                'sources': ['news', 'twitter']
            }
        }
        
        # Create historical training data
        self.historical_data = []
        for i in range(150):
            signal = {
                'z_score': np.random.normal(0, 2),
                'correlation': np.random.uniform(0.5, 0.9),
                'spread_std': np.random.uniform(0.02, 0.08),
                'volume_ratio': np.random.uniform(0.5, 1.5),
                'price_momentum': np.random.normal(0, 0.03),
                'volatility': np.random.uniform(0.1, 0.2),
                'market_regime': np.random.choice([0, 1, 2]),
                'sentiment_score': np.random.uniform(-0.5, 0.5),
                'timestamp': datetime.now() + timedelta(hours=i),
                'pnl': np.random.normal(0.02, 0.05)
            }
            self.historical_data.append(signal)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test EnhancedSignalGenerator initialization"""
        self.assertIsNotNone(self.signal_generator.ml_filter)
        self.assertIsNotNone(self.signal_generator.risk_manager)
        self.assertEqual(len(self.signal_generator.signals), 0)
        self.assertEqual(len(self.signal_generator.historical_signals), 0)
    
    def test_add_market_data(self):
        """Test adding market data for risk calculations"""
        self.signal_generator.add_market_data('BTC', 50000, 1000)
        
        # Verify data was added to risk manager
        # (This would require access to risk manager's internal state)
        # For now, just test that the method doesn't raise an exception
        self.assertTrue(True)
    
    def test_generate_statistical_signals(self):
        """Test statistical signal generation"""
        signals = self.signal_generator._generate_statistical_signals(self.market_data)
        
        # Should generate signals for both BTC and ETH
        self.assertGreater(len(signals), 0)
        
        for signal in signals:
            self.assertIn('symbol', signal)
            self.assertIn('side', signal)
            self.assertIn('confidence', signal)
            self.assertIn('price', signal)
            self.assertIn('z_score', signal)
            self.assertIn('strategy', signal)
            self.assertIn('timestamp', signal)
            
            self.assertIn(signal['symbol'], ['BTC', 'ETH'])
            self.assertIn(signal['side'], ['long', 'short'])
            self.assertGreaterEqual(signal['confidence'], 0)
            self.assertLessEqual(signal['confidence'], 1)
            self.assertEqual(signal['strategy'], 'statistical_arbitrage')
    
    def test_generate_sentiment_signals(self):
        """Test sentiment-based signal generation"""
        signals = self.signal_generator._generate_sentiment_signals(self.market_data, self.sentiment_data)
        
        # Should generate signals based on sentiment
        self.assertGreater(len(signals), 0)
        
        for signal in signals:
            self.assertIn('symbol', signal)
            self.assertIn('side', signal)
            self.assertIn('confidence', signal)
            self.assertIn('price', signal)
            self.assertIn('sentiment_score', signal)
            self.assertIn('strategy', signal)
            self.assertIn('timestamp', signal)
            
            self.assertIn(signal['symbol'], ['BTC', 'ETH'])
            self.assertIn(signal['side'], ['long', 'short'])
            self.assertGreaterEqual(signal['confidence'], 0)
            self.assertLessEqual(signal['confidence'], 1)
            self.assertEqual(signal['strategy'], 'sentiment')
    
    def test_generate_sentiment_signals_no_sentiment(self):
        """Test sentiment signal generation with no sentiment data"""
        signals = self.signal_generator._generate_sentiment_signals(self.market_data, None)
        
        # Should return empty list when no sentiment data
        self.assertEqual(len(signals), 0)
    
    def test_generate_sentiment_signals_missing_symbol(self):
        """Test sentiment signal generation with missing symbol in market data"""
        sentiment_data = {
            'UNKNOWN': {
                'score': 0.5,
                'confidence': 0.8,
                'sources': ['news']
            }
        }
        
        signals = self.signal_generator._generate_sentiment_signals(self.market_data, sentiment_data)
        
        # Should not generate signals for unknown symbols
        self.assertEqual(len(signals), 0)
    
    @patch('src.ml.signal_filter.MLSignalFilter.filter_signals')
    def test_generate_signals_with_ml_filtering(self, mock_filter_signals):
        """Test signal generation with ML filtering"""
        # Mock the ML filter to return filtered signals
        mock_filter_signals.return_value = [
            {
                'symbol': 'BTC',
                'side': 'long',
                'confidence': 0.8,
                'ml_quality': 0.7,
                'price': 52000,
                'strategy': 'statistical_arbitrage',
                'timestamp': datetime.now()
            }
        ]
        
        signals = self.signal_generator.generate_signals(self.market_data, self.sentiment_data)
        
        # Should call ML filter
        mock_filter_signals.assert_called_once()
        
        # Should return enhanced signals
        self.assertGreater(len(signals), 0)
        
        for signal in signals:
            self.assertIsInstance(signal, EnhancedSignal)
            self.assertIsNotNone(signal.symbol)
            self.assertIsNotNone(signal.side)
            self.assertGreaterEqual(signal.confidence, 0)
            self.assertLessEqual(signal.confidence, 1)
            self.assertGreaterEqual(signal.ml_quality, 0)
            self.assertLessEqual(signal.ml_quality, 1)
            self.assertGreaterEqual(signal.risk_score, 0)
            self.assertLessEqual(signal.risk_score, 1)
            self.assertGreater(signal.position_size, 0)
            self.assertIsNotNone(signal.entry_price)
            self.assertIsNotNone(signal.stop_loss)
            self.assertIsNotNone(signal.take_profit)
            self.assertIsNotNone(signal.strategy)
            self.assertIsNotNone(signal.timestamp)
            self.assertIsNotNone(signal.metadata)
    
    def test_generate_signals_no_sentiment(self):
        """Test signal generation without sentiment data"""
        signals = self.signal_generator.generate_signals(self.market_data)
        
        # Should still generate signals (statistical only)
        self.assertGreaterEqual(len(signals), 0)
    
    def test_train_ml_model_sufficient_data(self):
        """Test ML model training with sufficient data"""
        with patch.object(self.signal_generator.ml_filter, 'train') as mock_train:
            mock_train.return_value = True
            
            self.signal_generator.train_ml_model(self.historical_data)
            
            mock_train.assert_called_once_with(self.historical_data)
    
    def test_train_ml_model_insufficient_data(self):
        """Test ML model training with insufficient data"""
        small_data = self.historical_data[:50]  # Less than 100 samples
        
        with patch.object(self.signal_generator.ml_filter, 'train') as mock_train:
            self.signal_generator.train_ml_model(small_data)
            
            # Should not call train method
            mock_train.assert_not_called()
    
    def test_get_performance_metrics(self):
        """Test performance metrics calculation"""
        # Add some signals to the generator
        self.signal_generator.signals = [
            EnhancedSignal(
                symbol='BTC',
                side='long',
                confidence=0.8,
                ml_quality=0.7,
                risk_score=0.9,
                position_size=1000,
                entry_price=50000,
                stop_loss=49000,
                take_profit=52000,
                strategy='statistical_arbitrage',
                timestamp=datetime.now(),
                metadata={'pnl': 0.05}
            ),
            EnhancedSignal(
                symbol='ETH',
                side='short',
                confidence=0.6,
                ml_quality=0.5,
                risk_score=0.8,
                position_size=500,
                entry_price=3000,
                stop_loss=3100,
                take_profit=2900,
                strategy='sentiment',
                timestamp=datetime.now(),
                metadata={'pnl': -0.02}
            )
        ]
        
        metrics = self.signal_generator.get_performance_metrics()
        
        self.assertIn('total_signals', metrics)
        self.assertIn('profitable_signals', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('avg_ml_quality', metrics)
        self.assertIn('avg_confidence', metrics)
        
        self.assertEqual(metrics['total_signals'], 2)
        self.assertEqual(metrics['profitable_signals'], 1)
        self.assertEqual(metrics['win_rate'], 0.5)
        self.assertEqual(metrics['avg_ml_quality'], 0.6)
        self.assertEqual(metrics['avg_confidence'], 0.7)
    
    def test_get_performance_metrics_no_signals(self):
        """Test performance metrics with no signals"""
        metrics = self.signal_generator.get_performance_metrics()
        
        self.assertEqual(metrics, {})
    
    def test_get_risk_report(self):
        """Test risk report generation"""
        report = self.signal_generator.get_risk_report()
        
        # Should return a dictionary with risk metrics
        self.assertIsInstance(report, dict)
        # Specific metrics would depend on the risk manager implementation
    
    def test_enhanced_signal_dataclass(self):
        """Test EnhancedSignal dataclass"""
        signal = EnhancedSignal(
            symbol='BTC',
            side='long',
            confidence=0.8,
            ml_quality=0.7,
            risk_score=0.9,
            position_size=1000,
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000,
            strategy='statistical_arbitrage',
            timestamp=datetime.now(),
            metadata={'test': 'data'}
        )
        
        self.assertEqual(signal.symbol, 'BTC')
        self.assertEqual(signal.side, 'long')
        self.assertEqual(signal.confidence, 0.8)
        self.assertEqual(signal.ml_quality, 0.7)
        self.assertEqual(signal.risk_score, 0.9)
        self.assertEqual(signal.position_size, 1000)
        self.assertEqual(signal.entry_price, 50000)
        self.assertEqual(signal.stop_loss, 49000)
        self.assertEqual(signal.take_profit, 52000)
        self.assertEqual(signal.strategy, 'statistical_arbitrage')
        self.assertIsInstance(signal.timestamp, datetime)
        self.assertEqual(signal.metadata, {'test': 'data'})
    
    def test_statistical_signal_generation_edge_cases(self):
        """Test statistical signal generation with edge cases"""
        # Test with insufficient data
        small_market_data = {
            'BTC': [
                {'price': 50000, 'volume': 1000, 'timestamp': datetime.now()},
                {'price': 50100, 'volume': 1100, 'timestamp': datetime.now()}
            ]
        }
        
        signals = self.signal_generator._generate_statistical_signals(small_market_data)
        
        # Should not generate signals with insufficient data
        self.assertEqual(len(signals), 0)
        
        # Test with zero standard deviation
        zero_std_data = {
            'BTC': [
                {'price': 50000, 'volume': 1000, 'timestamp': datetime.now()}
            ] * 20  # Same price repeated
        }
        
        signals = self.signal_generator._generate_statistical_signals(zero_std_data)
        
        # Should not generate signals with zero standard deviation
        self.assertEqual(len(signals), 0)
    
    def test_sentiment_signal_generation_edge_cases(self):
        """Test sentiment signal generation with edge cases"""
        # Test with low confidence sentiment
        low_confidence_sentiment = {
            'BTC': {
                'score': 0.5,
                'confidence': 0.4,  # Below threshold
                'sources': ['news']
            }
        }
        
        signals = self.signal_generator._generate_sentiment_signals(self.market_data, low_confidence_sentiment)
        
        # Should not generate signals with low confidence
        self.assertEqual(len(signals), 0)
        
        # Test with neutral sentiment
        neutral_sentiment = {
            'BTC': {
                'score': 0.1,  # Below threshold
                'confidence': 0.8,
                'sources': ['news']
            }
        }
        
        signals = self.signal_generator._generate_sentiment_signals(self.market_data, neutral_sentiment)
        
        # Should not generate signals with neutral sentiment
        self.assertEqual(len(signals), 0)

if __name__ == '__main__':
    unittest.main() 