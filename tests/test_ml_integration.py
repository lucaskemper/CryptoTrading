#!/usr/bin/env python3
"""
Integration Tests for ML Components
Tests the complete ML pipeline including data flow, model training, and integration
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
import asyncio

# Import all ML-related components
from src.ml.signal_filter import MLSignalFilter
from src.strategy.enhanced_signal_generator import EnhancedSignalGenerator, EnhancedSignal
from src.strategy.stat_arb import StatisticalArbitrage
from src.strategy.sentiment import SentimentAnalyzer
from src.data_collector import DataCollector

class TestMLIntegration(unittest.TestCase):
    """Integration tests for the complete ML pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "integration_model.pkl")
        
        # Create ML components
        self.ml_filter = MLSignalFilter(model_path=self.model_path)
        self.signal_generator = EnhancedSignalGenerator(initial_capital=100000.0)
        
        # Create sample market data
        self.market_data = {
            'BTC': [
                {'price': 50000 + i*100, 'volume': 1000 + i*100, 'timestamp': datetime.now() + timedelta(hours=i)}
                for i in range(50)
            ],
            'ETH': [
                {'price': 3000 + i*10, 'volume': 500 + i*50, 'timestamp': datetime.now() + timedelta(hours=i)}
                for i in range(50)
            ],
            'SOL': [
                {'price': 100 + i*2, 'volume': 200 + i*20, 'timestamp': datetime.now() + timedelta(hours=i)}
                for i in range(50)
            ]
        }
        
        # Create sample sentiment data
        self.sentiment_data = {
            'BTC': {
                'score': 0.4,
                'confidence': 0.7,
                'sources': ['news', 'twitter', 'reddit'],
                'timestamp': datetime.now()
            },
            'ETH': {
                'score': -0.2,
                'confidence': 0.6,
                'sources': ['news', 'twitter'],
                'timestamp': datetime.now()
            },
            'SOL': {
                'score': 0.1,
                'confidence': 0.5,
                'sources': ['reddit'],
                'timestamp': datetime.now()
            }
        }
        
        # Create historical training data with realistic patterns
        self.historical_data = self._create_realistic_training_data()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_realistic_training_data(self):
        """Create realistic historical training data"""
        data = []
        
        # Create different market scenarios
        scenarios = [
            {'name': 'bull_market', 'pnl_mean': 0.03, 'pnl_std': 0.02, 'success_rate': 0.7},
            {'name': 'bear_market', 'pnl_mean': -0.02, 'pnl_std': 0.03, 'success_rate': 0.3},
            {'name': 'sideways_market', 'pnl_mean': 0.01, 'pnl_std': 0.015, 'success_rate': 0.5}
        ]
        
        for scenario in scenarios:
            for i in range(50):  # 50 samples per scenario
                # Determine if this signal would be profitable
                is_profitable = np.random.random() < scenario['success_rate']
                
                if is_profitable:
                    pnl = abs(np.random.normal(scenario['pnl_mean'], scenario['pnl_std']))
                else:
                    pnl = -abs(np.random.normal(scenario['pnl_mean'], scenario['pnl_std']))
                
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
                    'pnl': pnl,
                    'scenario': scenario['name']
                }
                data.append(signal)
        
        return data
    
    def test_complete_ml_pipeline(self):
        """Test the complete ML pipeline from data to predictions"""
        # Step 1: Train the ML model
        success = self.ml_filter.train(self.historical_data)
        self.assertTrue(success)
        self.assertTrue(self.ml_filter.is_trained)
        
        # Step 2: Generate signals using enhanced signal generator
        signals = self.signal_generator.generate_signals(self.market_data, self.sentiment_data)
        
        # Step 3: Verify signals have ML quality scores
        for signal in signals:
            self.assertIsInstance(signal, EnhancedSignal)
            self.assertGreaterEqual(signal.ml_quality, 0)
            self.assertLessEqual(signal.ml_quality, 1)
            self.assertGreaterEqual(signal.confidence, 0)
            self.assertLessEqual(signal.confidence, 1)
            self.assertGreater(signal.position_size, 0)
    
    def test_ml_model_persistence_and_reloading(self):
        """Test that ML model can be saved and reloaded correctly"""
        # Train model
        self.ml_filter.train(self.historical_data)
        
        # Create new instance and load model
        new_filter = MLSignalFilter(model_path=self.model_path)
        success = new_filter.load_model()
        
        self.assertTrue(success)
        self.assertTrue(new_filter.is_trained)
        
        # Test prediction consistency
        test_signal = {
            'z_score': 2.0,
            'correlation': 0.9,
            'spread_std': 0.04,
            'volume_ratio': 1.1,
            'price_momentum': 0.03,
            'volatility': 0.14,
            'market_regime': 1,
            'sentiment_score': 0.4,
            'timestamp': datetime.now()
        }
        
        quality1, confidence1 = self.ml_filter.predict_signal_quality(test_signal)
        quality2, confidence2 = new_filter.predict_signal_quality(test_signal)
        
        # Predictions should be consistent
        self.assertAlmostEqual(quality1, quality2, places=5)
        self.assertAlmostEqual(confidence1, confidence2, places=5)
    
    def test_signal_filtering_and_ranking(self):
        """Test signal filtering and ranking functionality"""
        # Train model
        self.ml_filter.train(self.historical_data)
        
        # Create test signals
        test_signals = [
            {
                'symbol': 'BTC',
                'side': 'long',
                'z_score': 2.0,
                'correlation': 0.9,
                'spread_std': 0.04,
                'volume_ratio': 1.1,
                'price_momentum': 0.03,
                'volatility': 0.14,
                'market_regime': 1,
                'sentiment_score': 0.4,
                'timestamp': datetime.now()
            },
            {
                'symbol': 'ETH',
                'side': 'short',
                'z_score': -1.5,
                'correlation': 0.7,
                'spread_std': 0.06,
                'volume_ratio': 0.9,
                'price_momentum': -0.01,
                'volatility': 0.16,
                'market_regime': 0,
                'sentiment_score': -0.2,
                'timestamp': datetime.now()
            },
            {
                'symbol': 'SOL',
                'side': 'long',
                'z_score': 1.2,
                'correlation': 0.8,
                'spread_std': 0.03,
                'volume_ratio': 1.0,
                'price_momentum': 0.02,
                'volatility': 0.15,
                'market_regime': 1,
                'sentiment_score': 0.3,
                'timestamp': datetime.now()
            }
        ]
        
        # Test filtering
        filtered_signals = self.ml_filter.filter_signals(test_signals, min_quality=0.5, min_confidence=0.3)
        
        # All original signals should have ML quality scores
        for signal in test_signals:
            self.assertIn('ml_quality', signal)
            self.assertIn('ml_confidence', signal)
        
        # Test ranking
        ranked_signals = self.ml_filter.rank_signals(test_signals)
        
        # Should be sorted by quality (descending)
        qualities = [signal.get('ml_quality', 0) for signal in ranked_signals]
        self.assertEqual(qualities, sorted(qualities, reverse=True))
    
    def test_enhanced_signal_generation_with_ml(self):
        """Test enhanced signal generation with ML integration"""
        # Train the ML model first
        self.ml_filter.train(self.historical_data)
        
        # Generate signals
        signals = self.signal_generator.generate_signals(self.market_data, self.sentiment_data)
        
        # Verify signal structure and ML integration
        for signal in signals:
            self.assertIsInstance(signal, EnhancedSignal)
            self.assertIsNotNone(signal.symbol)
            self.assertIsNotNone(signal.side)
            self.assertGreaterEqual(signal.ml_quality, 0)
            self.assertLessEqual(signal.ml_quality, 1)
            self.assertGreaterEqual(signal.confidence, 0)
            self.assertLessEqual(signal.confidence, 1)
            self.assertGreater(signal.position_size, 0)
            self.assertIsNotNone(signal.entry_price)
            self.assertIsNotNone(signal.stop_loss)
            self.assertIsNotNone(signal.take_profit)
            self.assertIsNotNone(signal.strategy)
            self.assertIsNotNone(signal.timestamp)
            self.assertIsNotNone(signal.metadata)
    
    def test_ml_model_performance_metrics(self):
        """Test ML model performance and metrics"""
        # Train model
        self.ml_filter.train(self.historical_data)
        
        # Test predictions on historical data
        predictions = []
        actual_outcomes = []
        
        for signal in self.historical_data[:50]:  # Test on subset
            quality, confidence = self.ml_filter.predict_signal_quality(signal)
            predictions.append(quality)
            actual_outcomes.append(1 if signal['pnl'] > 0 else 0)
        
        # Calculate basic metrics
        correct_predictions = sum(1 for p, a in zip(predictions, actual_outcomes) 
                                if (p > 0.5 and a == 1) or (p <= 0.5 and a == 0))
        accuracy = correct_predictions / len(predictions)
        
        # Should have reasonable accuracy (better than random)
        self.assertGreater(accuracy, 0.4)  # Better than 40% accuracy
        
        # Quality scores should be reasonable
        self.assertGreater(np.mean(predictions), 0.2)
        self.assertLess(np.mean(predictions), 0.8)
    
    def test_error_handling_and_robustness(self):
        """Test error handling and robustness of ML components"""
        # Test with invalid data
        invalid_signals = [
            {
                'z_score': 'invalid',
                'timestamp': datetime.now()
            },
            {
                'z_score': None,
                'timestamp': datetime.now()
            },
            {
                'z_score': np.inf,
                'timestamp': datetime.now()
            }
        ]
        
        # Should handle invalid data gracefully
        for signal in invalid_signals:
            quality, confidence = self.ml_filter.predict_signal_quality(signal)
            self.assertEqual(quality, 0.5)  # Default value
            self.assertEqual(confidence, 0.0)  # Default value
        
        # Test with empty data
        empty_signals = []
        filtered = self.ml_filter.filter_signals(empty_signals)
        self.assertEqual(filtered, [])
        
        ranked = self.ml_filter.rank_signals(empty_signals)
        self.assertEqual(ranked, [])
    
    def test_ml_integration_with_risk_management(self):
        """Test ML integration with risk management"""
        # Train model
        self.ml_filter.train(self.historical_data)
        
        # Generate signals with risk management
        signals = self.signal_generator.generate_signals(self.market_data, self.sentiment_data)
        
        # Verify risk management integration
        for signal in signals:
            self.assertGreaterEqual(signal.risk_score, 0)
            self.assertLessEqual(signal.risk_score, 1)
            self.assertGreater(signal.position_size, 0)
            self.assertIsNotNone(signal.stop_loss)
            self.assertIsNotNone(signal.take_profit)
            
            # Position size should be reasonable
            self.assertLess(signal.position_size, 100000)  # Should not exceed capital
    
    def test_ml_model_feature_importance(self):
        """Test ML model feature importance analysis"""
        # Train model
        self.ml_filter.train(self.historical_data)
        
        # Check feature importance
        if hasattr(self.ml_filter.model, 'feature_importances_'):
            importances = self.ml_filter.model.feature_importances_
            
            # Should have importance for each feature
            self.assertEqual(len(importances), len(self.ml_filter.feature_columns))
            
            # Importance should sum to 1
            self.assertAlmostEqual(np.sum(importances), 1.0, places=5)
            
            # All importances should be non-negative
            self.assertTrue(np.all(importances >= 0))
    
    def test_ml_model_hyperparameter_sensitivity(self):
        """Test ML model sensitivity to different hyperparameters"""
        # Test with different training data sizes
        small_data = self.historical_data[:50]
        large_data = self.historical_data[:200]
        
        # Should handle different data sizes
        success_small = self.ml_filter.train(small_data)
        self.assertFalse(success_small)  # Should fail with insufficient data
        
        # Create new instance for large data
        large_filter = MLSignalFilter(model_path=os.path.join(self.temp_dir, "large_model.pkl"))
        success_large = large_filter.train(large_data)
        self.assertTrue(success_large)
    
    def test_ml_model_temporal_consistency(self):
        """Test ML model consistency over time"""
        # Train model
        self.ml_filter.train(self.historical_data)
        
        # Test predictions on same signal over time
        test_signal = {
            'z_score': 2.0,
            'correlation': 0.9,
            'spread_std': 0.04,
            'volume_ratio': 1.1,
            'price_momentum': 0.03,
            'volatility': 0.14,
            'market_regime': 1,
            'sentiment_score': 0.4,
            'timestamp': datetime.now()
        }
        
        predictions = []
        for _ in range(10):
            quality, confidence = self.ml_filter.predict_signal_quality(test_signal)
            predictions.append((quality, confidence))
        
        # Predictions should be consistent (same input, same output)
        first_prediction = predictions[0]
        for prediction in predictions[1:]:
            self.assertAlmostEqual(first_prediction[0], prediction[0], places=5)
            self.assertAlmostEqual(first_prediction[1], prediction[1], places=5)

if __name__ == '__main__':
    unittest.main() 