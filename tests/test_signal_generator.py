"""
Unit tests for Signal Generator module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from src.strategy.signal_generator import (
    SignalGenerator, TradeSignal, CombinationMethod, 
    SignalType, SignalSource, create_signal_generator
)
from src.strategy.stat_arb import Signal as StatSignal
from src.strategy.sentiment import SentimentData


class TestTradeSignal(unittest.TestCase):
    """Test TradeSignal dataclass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_signal = TradeSignal(
            symbol="BTC-ETH",
            side="buy",
            quantity=0.1,
            order_type="market",
            price=None,
            confidence=0.75,
            sources=["stat_arb", "sentiment"],
            metadata={"z_score": 2.1, "sentiment_score": 0.3},
            timestamp=datetime.now(),
            signal_type="entry_long",
            risk_checked=False
        )
    
    def test_valid_signal(self):
        """Test that a valid signal passes validation."""
        self.assertTrue(self.valid_signal.validate())
    
    def test_invalid_symbol(self):
        """Test that empty symbol fails validation."""
        signal = self.valid_signal
        signal.symbol = ""
        self.assertFalse(signal.validate())
    
    def test_invalid_side(self):
        """Test that invalid side fails validation."""
        signal = self.valid_signal
        signal.side = "invalid"
        self.assertFalse(signal.validate())
    
    def test_invalid_quantity(self):
        """Test that zero or negative quantity fails validation."""
        signal = self.valid_signal
        signal.quantity = 0
        self.assertFalse(signal.validate())
        
        signal.quantity = -0.1
        self.assertFalse(signal.validate())
    
    def test_invalid_confidence(self):
        """Test that confidence outside [0,1] fails validation."""
        signal = self.valid_signal
        signal.confidence = 1.5
        self.assertFalse(signal.validate())
        
        signal.confidence = -0.1
        self.assertFalse(signal.validate())
    
    def test_invalid_order_type(self):
        """Test that invalid order type fails validation."""
        signal = self.valid_signal
        signal.order_type = "invalid"
        self.assertFalse(signal.validate())
    
    def test_limit_order_without_price(self):
        """Test that limit orders require price."""
        signal = self.valid_signal
        signal.order_type = "limit"
        signal.price = None
        self.assertFalse(signal.validate())
        
        signal.price = 50000.0
        self.assertTrue(signal.validate())


class TestSignalGenerator(unittest.TestCase):
    """Test SignalGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock components
        self.mock_stat_arb = Mock()
        self.mock_sentiment_analyzer = Mock()
        self.mock_risk_manager = Mock()
        
        # Create signal generator
        self.signal_gen = SignalGenerator(
            stat_arb=self.mock_stat_arb,
            sentiment_analyzer=self.mock_sentiment_analyzer,
            risk_manager=self.mock_risk_manager,
            config={
                'combination_method': 'consensus',
                'stat_weight': 0.6,
                'sentiment_weight': 0.4,
                'min_confidence': 0.3,
                'enable_risk_checks': True,
                'require_risk_approval': False
            }
        )
    
    def test_initialization(self):
        """Test signal generator initialization."""
        self.assertIsNotNone(self.signal_gen)
        self.assertEqual(self.signal_gen.combination_method, CombinationMethod.CONSENSUS)
        self.assertEqual(self.signal_gen.stat_weight, 0.6)
        self.assertEqual(self.signal_gen.sentiment_weight, 0.4)
        self.assertEqual(self.signal_gen.min_confidence, 0.3)
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        config = self.signal_gen._load_default_config()
        self.assertIsInstance(config, dict)
        self.assertIn('combination_method', config)
        self.assertIn('stat_weight', config)
        self.assertIn('sentiment_weight', config)
    
    @patch('src.strategy.signal_generator.config')
    def test_load_default_config_with_error(self, mock_config):
        """Test loading config when config module fails."""
        mock_config.get.side_effect = Exception("Config error")
        
        config = self.signal_gen._load_default_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(len(config), 0)
    
    def test_get_statistical_signals_success(self):
        """Test getting statistical signals successfully."""
        # Mock stat arb signals
        mock_signals = [
            StatSignal(
                timestamp=datetime.now(),
                pair="BTC-ETH",
                signal_type="entry_long",
                asset1="BTC",
                asset2="ETH",
                action1="buy",
                action2="sell",
                size1=0.1,
                size2=1.5,
                z_score=2.1,
                spread_value=0.05,
                confidence=0.8
            )
        ]
        self.mock_stat_arb.generate_signals.return_value = mock_signals
        
        signals = self.signal_gen._get_statistical_signals()
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].pair, "BTC-ETH")
    
    def test_get_statistical_signals_no_stat_arb(self):
        """Test getting signals when stat arb is not available."""
        signal_gen = SignalGenerator(stat_arb=None)
        signals = signal_gen._get_statistical_signals()
        self.assertEqual(len(signals), 0)
    
    def test_get_statistical_signals_error(self):
        """Test getting signals when stat arb raises error."""
        self.mock_stat_arb.generate_signals.side_effect = Exception("Stat arb error")
        
        signals = self.signal_gen._get_statistical_signals()
        self.assertEqual(len(signals), 0)
    
    def test_get_sentiment_signals_success(self):
        """Test getting sentiment signals successfully."""
        # Mock sentiment data
        sentiment_data = [
            SentimentData(
                timestamp=datetime.now(),
                source="reddit",
                text="BTC is going to the moon!",
                sentiment_score=0.8,
                keywords=["BTC", "moon"]
            )
        ]
        
        # Mock sentiment analyzer
        mock_scores = pd.Series({'BTC': 0.6, 'ETH': 0.3, 'SOL': 0.1})
        self.mock_sentiment_analyzer.aggregate_sentiment.return_value = mock_scores
        
        signals = self.signal_gen._get_sentiment_signals(sentiment_data)
        self.assertIn('BTC', signals)
        self.assertIn('ETH', signals)
        self.assertIn('SOL', signals)
    
    def test_get_sentiment_signals_no_analyzer(self):
        """Test getting sentiment signals when analyzer is not available."""
        signal_gen = SignalGenerator(sentiment_analyzer=None)
        signals = signal_gen._get_sentiment_signals([])
        self.assertEqual(len(signals), 0)
    
    def test_get_sentiment_signals_no_data(self):
        """Test getting sentiment signals with no data."""
        signals = self.signal_gen._get_sentiment_signals(None)
        self.assertEqual(len(signals), 0)
    
    def test_generate_sentiment_signal(self):
        """Test sentiment signal generation."""
        sentiment1, sentiment2 = 0.5, -0.2
        signal = self.signal_gen._generate_sentiment_signal(sentiment1, sentiment2)
        
        self.assertEqual(signal['direction'], 'neutral')  # 0.15 is between -0.2 and 0.2
        self.assertAlmostEqual(signal['score'], 0.15)  # (0.5 + -0.2) / 2
        self.assertIn('confidence', signal)
        self.assertIn('asset1_sentiment', signal)
        self.assertIn('asset2_sentiment', signal)
    
    def test_combine_consensus_agreement(self):
        """Test consensus combination when signals agree."""
        stat_signal = StatSignal(
            timestamp=datetime.now(),
            pair="BTC-ETH",
            signal_type="entry_long",
            asset1="BTC",
            asset2="ETH",
            action1="buy",
            action2="sell",
            size1=0.1,
            size2=1.5,
            z_score=2.1,
            spread_value=0.05,
            confidence=0.8
        )
        
        sentiment_signal = {
            'direction': 'positive',
            'score': 0.3,
            'confidence': 0.7,
            'asset1_sentiment': 0.4,
            'asset2_sentiment': 0.2
        }
        
        result = self.signal_gen._combine_consensus(stat_signal, sentiment_signal)
        self.assertIsNotNone(result)
        self.assertEqual(result.side, 'buy')
        self.assertEqual(result.symbol, 'BTC-ETH')
    
    def test_combine_consensus_disagreement(self):
        """Test consensus combination when signals disagree."""
        stat_signal = StatSignal(
            timestamp=datetime.now(),
            pair="BTC-ETH",
            signal_type="entry_long",
            asset1="BTC",
            asset2="ETH",
            action1="buy",
            action2="sell",
            size1=0.1,
            size2=1.5,
            z_score=2.1,
            spread_value=0.05,
            confidence=0.8
        )
        
        sentiment_signal = {
            'direction': 'negative',
            'score': -0.3,
            'confidence': 0.7,
            'asset1_sentiment': -0.4,
            'asset2_sentiment': -0.2
        }
        
        result = self.signal_gen._combine_consensus(stat_signal, sentiment_signal)
        self.assertIsNone(result)
    
    def test_combine_weighted(self):
        """Test weighted combination method."""
        stat_signal = StatSignal(
            timestamp=datetime.now(),
            pair="BTC-ETH",
            signal_type="entry_long",
            asset1="BTC",
            asset2="ETH",
            action1="buy",
            action2="sell",
            size1=0.1,
            size2=1.5,
            z_score=2.1,
            spread_value=0.05,
            confidence=0.8
        )
        
        sentiment_signal = {
            'direction': 'positive',
            'score': 0.3,
            'confidence': 0.7,
            'asset1_sentiment': 0.4,
            'asset2_sentiment': 0.2
        }
        
        result = self.signal_gen._combine_weighted(stat_signal, sentiment_signal)
        self.assertIsNotNone(result)
        
        # Check weighted confidence calculation
        expected_confidence = (0.6 * 0.8 + 0.4 * 0.7)
        self.assertAlmostEqual(result.confidence, expected_confidence)
    
    def test_combine_weighted_below_threshold(self):
        """Test weighted combination when confidence is below threshold."""
        self.signal_gen.min_confidence = 0.9
        
        stat_signal = StatSignal(
            timestamp=datetime.now(),
            pair="BTC-ETH",
            signal_type="entry_long",
            asset1="BTC",
            asset2="ETH",
            action1="buy",
            action2="sell",
            size1=0.1,
            size2=1.5,
            z_score=2.1,
            spread_value=0.05,
            confidence=0.3  # Low confidence
        )
        
        sentiment_signal = {
            'direction': 'positive',
            'score': 0.3,
            'confidence': 0.4,  # Low confidence
            'asset1_sentiment': 0.4,
            'asset2_sentiment': 0.2
        }
        
        result = self.signal_gen._combine_weighted(stat_signal, sentiment_signal)
        self.assertIsNone(result)
    
    def test_combine_filter_positive_sentiment(self):
        """Test filter combination with positive sentiment."""
        stat_signal = StatSignal(
            timestamp=datetime.now(),
            pair="BTC-ETH",
            signal_type="entry_long",
            asset1="BTC",
            asset2="ETH",
            action1="buy",
            action2="sell",
            size1=0.1,
            size2=1.5,
            z_score=2.1,
            spread_value=0.05,
            confidence=0.8
        )
        
        sentiment_signal = {
            'direction': 'positive',
            'score': 0.5,
            'confidence': 0.7,
            'asset1_sentiment': 0.6,
            'asset2_sentiment': 0.4
        }
        
        result = self.signal_gen._combine_filter(stat_signal, sentiment_signal)
        self.assertIsNotNone(result)
        # Should have increased confidence
        self.assertGreater(result.confidence, 0.8)
    
    def test_combine_filter_negative_sentiment_block(self):
        """Test filter combination blocking signal with strong negative sentiment."""
        stat_signal = StatSignal(
            timestamp=datetime.now(),
            pair="BTC-ETH",
            signal_type="entry_long",
            asset1="BTC",
            asset2="ETH",
            action1="buy",
            action2="sell",
            size1=0.1,
            size2=1.5,
            z_score=2.1,
            spread_value=0.05,
            confidence=0.8
        )
        
        sentiment_signal = {
            'direction': 'negative',
            'score': -0.5,  # Strong negative
            'confidence': 0.7,
            'asset1_sentiment': -0.6,
            'asset2_sentiment': -0.4
        }
        
        result = self.signal_gen._combine_filter(stat_signal, sentiment_signal)
        self.assertIsNone(result)
    
    def test_validate_signals(self):
        """Test signal validation."""
        valid_signal = TradeSignal(
            symbol="BTC-ETH",
            side="buy",
            quantity=0.1,
            order_type="market",
            price=None,
            confidence=0.75,
            sources=["stat_arb"],
            metadata={},
            timestamp=datetime.now(),
            signal_type="entry_long"
        )
        
        invalid_signal = TradeSignal(
            symbol="",  # Invalid
            side="buy",
            quantity=0.1,
            order_type="market",
            price=None,
            confidence=0.75,
            sources=["stat_arb"],
            metadata={},
            timestamp=datetime.now(),
            signal_type="entry_long"
        )
        
        signals = [valid_signal, invalid_signal]
        validated = self.signal_gen._validate_signals(signals)
        
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0], valid_signal)
    
    def test_apply_risk_checks_success(self):
        """Test applying risk checks successfully."""
        signal = TradeSignal(
            symbol="BTC-ETH",
            side="buy",
            quantity=0.1,
            order_type="market",
            price=None,
            confidence=0.75,
            sources=["stat_arb"],
            metadata={},
            timestamp=datetime.now(),
            signal_type="entry_long"
        )
        
        self.mock_risk_manager.check_order_risk.return_value = (True, "Approved")
        
        signals = [signal]
        result = self.signal_gen._apply_risk_checks(signals)
        
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].risk_checked)
    
    def test_apply_risk_checks_failure(self):
        """Test applying risk checks with failure."""
        signal = TradeSignal(
            symbol="BTC-ETH",
            side="buy",
            quantity=0.1,
            order_type="market",
            price=None,
            confidence=0.75,
            sources=["stat_arb"],
            metadata={},
            timestamp=datetime.now(),
            signal_type="entry_long"
        )
        
        self.mock_risk_manager.check_order_risk.return_value = (False, "Risk limit exceeded")
        
        signals = [signal]
        result = self.signal_gen._apply_risk_checks(signals)
        
        # Should still include signal since require_risk_approval is False
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0].risk_checked)
    
    def test_apply_risk_checks_no_risk_manager(self):
        """Test applying risk checks when risk manager is not available."""
        signal_gen = SignalGenerator(risk_manager=None)
        signal = TradeSignal(
            symbol="BTC-ETH",
            side="buy",
            quantity=0.1,
            order_type="market",
            price=None,
            confidence=0.75,
            sources=["stat_arb"],
            metadata={},
            timestamp=datetime.now(),
            signal_type="entry_long"
        )
        
        signals = [signal]
        result = signal_gen._apply_risk_checks(signals)
        
        self.assertEqual(len(result), 1)
    
    def test_get_signal_history(self):
        """Test getting signal history."""
        # Add some signals to history
        signal = TradeSignal(
            symbol="BTC-ETH",
            side="buy",
            quantity=0.1,
            order_type="market",
            price=None,
            confidence=0.75,
            sources=["stat_arb"],
            metadata={},
            timestamp=datetime.now(),
            signal_type="entry_long"
        )
        
        self.signal_gen.generated_signals.append(signal)
        
        history = self.signal_gen.get_signal_history(hours=1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['symbol'], 'BTC-ETH')
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        metrics = self.signal_gen.get_performance_metrics()
        
        self.assertIn('total_signals', metrics)
        self.assertIn('approved_signals', metrics)
        self.assertIn('rejected_signals', metrics)
        self.assertIn('approval_rate', metrics)
        self.assertIn('combination_method', metrics)
        self.assertIn('min_confidence', metrics)
    
    def test_update_config(self):
        """Test updating configuration."""
        new_config = {
            'combination_method': 'weighted',
            'stat_weight': 0.7,
            'sentiment_weight': 0.3,
            'min_confidence': 0.5
        }
        
        self.signal_gen.update_config(new_config)
        
        self.assertEqual(self.signal_gen.combination_method, CombinationMethod.WEIGHTED)
        self.assertEqual(self.signal_gen.stat_weight, 0.7)
        self.assertEqual(self.signal_gen.sentiment_weight, 0.3)
        self.assertEqual(self.signal_gen.min_confidence, 0.5)
    
    def test_reset(self):
        """Test resetting signal generator state."""
        # Add some data
        self.signal_gen.generated_signals.append(Mock())
        self.signal_gen.signal_history.append({})
        self.signal_gen.total_signals = 10
        self.signal_gen.approved_signals = 8
        self.signal_gen.rejected_signals = 2
        
        self.signal_gen.reset()
        
        self.assertEqual(len(self.signal_gen.generated_signals), 0)
        self.assertEqual(len(self.signal_gen.signal_history), 0)
        self.assertEqual(self.signal_gen.total_signals, 0)
        self.assertEqual(self.signal_gen.approved_signals, 0)
        self.assertEqual(self.signal_gen.rejected_signals, 0)


class TestCreateSignalGenerator(unittest.TestCase):
    """Test factory function."""
    
    def test_create_signal_generator(self):
        """Test creating signal generator with factory function."""
        mock_stat_arb = Mock()
        mock_sentiment_analyzer = Mock()
        mock_risk_manager = Mock()
        config = {'combination_method': 'consensus'}
        
        signal_gen = create_signal_generator(
            stat_arb=mock_stat_arb,
            sentiment_analyzer=mock_sentiment_analyzer,
            risk_manager=mock_risk_manager,
            config=config
        )
        
        self.assertIsInstance(signal_gen, SignalGenerator)
        self.assertEqual(signal_gen.stat_arb, mock_stat_arb)
        self.assertEqual(signal_gen.sentiment_analyzer, mock_sentiment_analyzer)
        self.assertEqual(signal_gen.risk_manager, mock_risk_manager)


if __name__ == '__main__':
    unittest.main() 