#!/usr/bin/env python3
"""
Tests for ML Signal Filter
Tests the machine learning signal filtering and ranking functionality
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock

# Import the ML components
from src.ml.signal_filter import MLSignalFilter

class TestMLSignalFilter(unittest.TestCase):
    """Test cases for MLSignalFilter class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pkl")
        self.ml_filter = MLSignalFilter(model_path=self.model_path)
        
        # Create sample training data
        self.sample_signals = [
            {
                'z_score': 2.1,
                'correlation': 0.8,
                'spread_std': 0.05,
                'volume_ratio': 1.2,
                'price_momentum': 0.02,
                'volatility': 0.15,
                'market_regime': 1,  # bull market
                'sentiment_score': 0.3,
                'timestamp': datetime.now(),
                'pnl': 0.05  # 5% profit
            },
            {
                'z_score': -1.5,
                'correlation': 0.7,
                'spread_std': 0.03,
                'volume_ratio': 0.8,
                'price_momentum': -0.01,
                'volatility': 0.12,
                'market_regime': 0,  # bear market
                'sentiment_score': -0.2,
                'timestamp': datetime.now(),
                'pnl': -0.02  # 2% loss
            },
            {
                'z_score': 1.8,
                'correlation': 0.9,
                'spread_std': 0.04,
                'volume_ratio': 1.1,
                'price_momentum': 0.03,
                'volatility': 0.14,
                'market_regime': 1,
                'sentiment_score': 0.4,
                'timestamp': datetime.now(),
                'pnl': 0.08  # 8% profit
            }
        ]
        
        # Create larger training dataset
        self.large_training_data = []
        for i in range(150):  # More than minimum required
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
                'pnl': np.random.normal(0.02, 0.05)  # Mix of profits and losses
            }
            self.large_training_data.append(signal)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test MLSignalFilter initialization"""
        self.assertIsNone(self.ml_filter.model)
        self.assertFalse(self.ml_filter.is_trained)
        self.assertEqual(len(self.ml_filter.feature_columns), 10)
        self.assertEqual(self.ml_filter.model_path, self.model_path)
    
    def test_extract_features(self):
        """Test feature extraction from signal data"""
        signal_data = {
            'z_score': 2.0,
            'correlation': 0.8,
            'spread_std': 0.05,
            'volume_ratio': 1.2,
            'price_momentum': 0.02,
            'volatility': 0.15,
            'market_regime': 1,
            'sentiment_score': 0.3,
            'timestamp': datetime.now()
        }
        
        features = self.ml_filter.extract_features(signal_data)
        
        self.assertEqual(features.shape, (1, 10))
        self.assertEqual(features[0, 0], 2.0)  # z_score
        self.assertEqual(features[0, 1], 0.8)  # correlation
        self.assertEqual(features[0, 2], 0.05)  # spread_std
        self.assertEqual(features[0, 3], 1.2)  # volume_ratio
        self.assertEqual(features[0, 4], 0.02)  # price_momentum
        self.assertEqual(features[0, 5], 0.15)  # volatility
        self.assertEqual(features[0, 6], 1)  # market_regime
        self.assertEqual(features[0, 7], 0.3)  # sentiment_score
        self.assertGreaterEqual(features[0, 8], 0)  # time_of_day
        self.assertLessEqual(features[0, 8], 1)
        self.assertGreaterEqual(features[0, 9], 0)  # day_of_week
        self.assertLessEqual(features[0, 9], 1)
    
    def test_extract_features_with_missing_data(self):
        """Test feature extraction with missing data"""
        signal_data = {
            'z_score': 1.5,
            'timestamp': datetime.now()
            # Missing other features
        }
        
        features = self.ml_filter.extract_features(signal_data)
        
        self.assertEqual(features.shape, (1, 10))
        self.assertEqual(features[0, 0], 1.5)  # z_score
        self.assertEqual(features[0, 1], 0)  # correlation (default)
        self.assertEqual(features[0, 2], 0)  # spread_std (default)
        self.assertEqual(features[0, 3], 1)  # volume_ratio (default)
        # ... other defaults
    
    def test_extract_features_with_string_timestamp(self):
        """Test feature extraction with string timestamp"""
        signal_data = {
            'z_score': 1.0,
            'timestamp': datetime.now().isoformat()
        }
        
        features = self.ml_filter.extract_features(signal_data)
        
        self.assertEqual(features.shape, (1, 10))
        self.assertEqual(features[0, 0], 1.0)
    
    def test_prepare_training_data(self):
        """Test training data preparation"""
        X, y = self.ml_filter.prepare_training_data(self.sample_signals)
        
        self.assertEqual(X.shape, (3, 10))
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y[0], 1)  # First signal profitable
        self.assertEqual(y[1], 0)  # Second signal loss
        self.assertEqual(y[2], 1)  # Third signal profitable
    
    def test_train_with_sufficient_data(self):
        """Test model training with sufficient data"""
        success = self.ml_filter.train(self.large_training_data)
        
        self.assertTrue(success)
        self.assertTrue(self.ml_filter.is_trained)
        self.assertIsNotNone(self.ml_filter.model)
        self.assertIsNotNone(self.ml_filter.scaler)
        self.assertTrue(os.path.exists(self.model_path))
    
    def test_train_with_insufficient_data(self):
        """Test model training with insufficient data"""
        small_data = self.sample_signals[:2]  # Only 2 samples
        success = self.ml_filter.train(small_data)
        
        self.assertFalse(success)
        self.assertFalse(self.ml_filter.is_trained)
        self.assertIsNone(self.ml_filter.model)
    
    def test_load_model_success(self):
        """Test successful model loading"""
        # First train a model
        self.ml_filter.train(self.large_training_data)
        
        # Create new instance and load
        new_filter = MLSignalFilter(model_path=self.model_path)
        success = new_filter.load_model()
        
        self.assertTrue(success)
        self.assertTrue(new_filter.is_trained)
        self.assertIsNotNone(new_filter.model)
        self.assertIsNotNone(new_filter.scaler)
    
    def test_load_model_failure(self):
        """Test model loading when file doesn't exist"""
        success = self.ml_filter.load_model()
        
        self.assertFalse(success)
        self.assertFalse(self.ml_filter.is_trained)
        self.assertIsNone(self.ml_filter.model)
    
    def test_predict_signal_quality_trained_model(self):
        """Test signal quality prediction with trained model"""
        # Train the model first
        self.ml_filter.train(self.large_training_data)
        
        signal_data = {
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
        
        quality, confidence = self.ml_filter.predict_signal_quality(signal_data)
        
        self.assertGreaterEqual(quality, 0)
        self.assertLessEqual(quality, 1)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)
    
    def test_predict_signal_quality_untrained_model(self):
        """Test signal quality prediction with untrained model"""
        signal_data = {
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
        
        quality, confidence = self.ml_filter.predict_signal_quality(signal_data)
        
        self.assertEqual(quality, 0.5)  # Default value
        self.assertEqual(confidence, 0.0)  # Default value
    
    def test_filter_signals_trained_model(self):
        """Test signal filtering with trained model"""
        # Train the model first
        self.ml_filter.train(self.large_training_data)
        
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
                'z_score': -1.0,
                'correlation': 0.6,
                'spread_std': 0.06,
                'volume_ratio': 0.9,
                'price_momentum': -0.01,
                'volatility': 0.16,
                'market_regime': 0,
                'sentiment_score': -0.2,
                'timestamp': datetime.now()
            }
        ]
        
        filtered_signals = self.ml_filter.filter_signals(test_signals, min_quality=0.6, min_confidence=0.3)
        
        # All signals should have ml_quality and ml_confidence added
        for signal in test_signals:
            self.assertIn('ml_quality', signal)
            self.assertIn('ml_confidence', signal)
            self.assertGreaterEqual(signal['ml_quality'], 0)
            self.assertLessEqual(signal['ml_quality'], 1)
            self.assertGreaterEqual(signal['ml_confidence'], 0)
            self.assertLessEqual(signal['ml_confidence'], 1)
        
        # Filtered signals should meet criteria
        for signal in filtered_signals:
            self.assertGreaterEqual(signal['ml_quality'], 0.6)
            self.assertGreaterEqual(signal['ml_confidence'], 0.3)
    
    def test_filter_signals_untrained_model(self):
        """Test signal filtering with untrained model"""
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
            }
        ]
        
        filtered_signals = self.ml_filter.filter_signals(test_signals)
        
        # Should return all signals when model is not trained
        self.assertEqual(len(filtered_signals), len(test_signals))
    
    def test_rank_signals(self):
        """Test signal ranking"""
        # Train the model first
        self.ml_filter.train(self.large_training_data)
        
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
                'z_score': -1.0,
                'correlation': 0.6,
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
                'z_score': 1.5,
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
        
        ranked_signals = self.ml_filter.rank_signals(test_signals)
        
        # Should have same number of signals
        self.assertEqual(len(ranked_signals), len(test_signals))
        
        # Should be sorted by ml_quality (descending)
        qualities = [signal.get('ml_quality', 0) for signal in ranked_signals]
        self.assertEqual(qualities, sorted(qualities, reverse=True))
        
        # All signals should have ml_quality and ml_confidence
        for signal in ranked_signals:
            self.assertIn('ml_quality', signal)
            self.assertIn('ml_confidence', signal)
    
    def test_rank_signals_empty_list(self):
        """Test signal ranking with empty list"""
        ranked_signals = self.ml_filter.rank_signals([])
        self.assertEqual(ranked_signals, [])
    
    def test_rank_signals_untrained_model(self):
        """Test signal ranking with untrained model"""
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
            }
        ]
        
        ranked_signals = self.ml_filter.rank_signals(test_signals)
        
        # Should return signals with default ml_quality values
        self.assertEqual(len(ranked_signals), 1)
        self.assertIn('ml_quality', ranked_signals[0])
        self.assertIn('ml_confidence', ranked_signals[0])
    
    def test_error_handling_in_predict_signal_quality(self):
        """Test error handling in predict_signal_quality"""
        # Train the model first
        self.ml_filter.train(self.large_training_data)
        
        # Test with invalid data
        invalid_signal = {
            'z_score': 'invalid',  # Should cause error
            'timestamp': datetime.now()
        }
        
        quality, confidence = self.ml_filter.predict_signal_quality(invalid_signal)
        
        # Should return default values on error
        self.assertEqual(quality, 0.5)
        self.assertEqual(confidence, 0.0)
    
    def test_model_persistence(self):
        """Test that trained model can be saved and loaded"""
        # Train model
        self.ml_filter.train(self.large_training_data)
        
        # Create new instance
        new_filter = MLSignalFilter(model_path=self.model_path)
        
        # Load model
        success = new_filter.load_model()
        
        self.assertTrue(success)
        self.assertTrue(new_filter.is_trained)
        
        # Test prediction with loaded model
        signal_data = {
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
        
        quality, confidence = new_filter.predict_signal_quality(signal_data)
        
        self.assertGreaterEqual(quality, 0)
        self.assertLessEqual(quality, 1)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)

if __name__ == '__main__':
    unittest.main() 