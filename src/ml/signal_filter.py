#!/usr/bin/env python3
"""
Machine Learning Signal Filter
Uses ML models to filter and rank trading signals for better performance
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MLSignalFilter:
    """Machine Learning-based signal filtering and ranking"""
    
    def __init__(self, model_path: str = "models/signal_filter.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'z_score', 'correlation', 'spread_std', 'volume_ratio',
            'price_momentum', 'volatility', 'market_regime',
            'sentiment_score', 'time_of_day', 'day_of_week'
        ]
        self.is_trained = False
        
    def extract_features(self, signal_data: Dict) -> np.ndarray:
        """Extract features from signal data"""
        features = []
        
        # Technical features
        features.append(signal_data.get('z_score', 0))
        features.append(signal_data.get('correlation', 0))
        features.append(signal_data.get('spread_std', 0))
        features.append(signal_data.get('volume_ratio', 1))
        
        # Price momentum (last 5 periods)
        features.append(signal_data.get('price_momentum', 0))
        
        # Volatility (rolling std)
        features.append(signal_data.get('volatility', 0))
        
        # Market regime (bull/bear/sideways)
        features.append(signal_data.get('market_regime', 0))
        
        # Sentiment features
        features.append(signal_data.get('sentiment_score', 0))
        
        # Time features
        timestamp = signal_data.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        features.append(timestamp.hour / 24.0)  # Time of day (0-1)
        features.append(timestamp.weekday() / 7.0)  # Day of week (0-1)
        
        return np.array(features).reshape(1, -1)
    
    def prepare_training_data(self, historical_signals: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from historical signals"""
        X = []
        y = []
        
        for signal in historical_signals:
            features = self.extract_features(signal)
            X.append(features.flatten())
            
            # Label: 1 if profitable, 0 if not
            pnl = signal.get('pnl', 0)
            y.append(1 if pnl > 0 else 0)
        
        return np.array(X), np.array(y)
    
    def train(self, historical_signals: List[Dict], test_size: float = 0.2):
        """Train the ML model on historical signal data"""
        logger.info("Training ML signal filter...")
        
        # Prepare data
        X, y = self.prepare_training_data(historical_signals)
        
        if len(X) < 100:
            logger.warning("Insufficient training data. Need at least 100 signals.")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model (Random Forest for interpretability)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"Model accuracy: {accuracy:.3f}")
        logger.info(f"Feature importance: {dict(zip(self.feature_columns, self.model.feature_importances_))}")
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, self.model_path)
        
        self.is_trained = True
        return True
    
    def load_model(self) -> bool:
        """Load trained model from disk"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_columns = model_data['feature_columns']
                self.is_trained = True
                logger.info("ML model loaded successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
        
        return False
    
    def predict_signal_quality(self, signal_data: Dict) -> Tuple[float, float]:
        """Predict signal quality and confidence"""
        if not self.is_trained:
            return 0.5, 0.0  # Default values if not trained
        
        try:
            features = self.extract_features(signal_data)
            features_scaled = self.scaler.transform(features)
            
            # Get probability of profitable trade
            prob = self.model.predict_proba(features_scaled)[0]
            quality_score = prob[1]  # Probability of positive outcome
            
            # Confidence based on feature importance and data quality
            confidence = min(1.0, np.std(features) + 0.5)
            
            return quality_score, confidence
            
        except Exception as e:
            logger.error(f"Error predicting signal quality: {e}")
            return 0.5, 0.0
    
    def filter_signals(self, signals: List[Dict], 
                      min_quality: float = 0.6,
                      min_confidence: float = 0.3) -> List[Dict]:
        """Filter signals based on ML predictions"""
        if not self.is_trained:
            logger.warning("ML model not trained, returning all signals")
            return signals
        
        filtered_signals = []
        
        for signal in signals:
            quality, confidence = self.predict_signal_quality(signal)
            
            signal['ml_quality'] = quality
            signal['ml_confidence'] = confidence
            
            if quality >= min_quality and confidence >= min_confidence:
                filtered_signals.append(signal)
                logger.info(f"Signal approved: quality={quality:.3f}, confidence={confidence:.3f}")
            else:
                logger.info(f"Signal rejected: quality={quality:.3f}, confidence={confidence:.3f}")
        
        return filtered_signals
    
    def rank_signals(self, signals: List[Dict]) -> List[Dict]:
        """Rank signals by ML quality score"""
        if not signals:
            return signals
        
        # Add ML predictions if not present
        for signal in signals:
            if 'ml_quality' not in signal:
                quality, confidence = self.predict_signal_quality(signal)
                signal['ml_quality'] = quality
                signal['ml_confidence'] = confidence
        
        # Sort by quality score (descending)
        ranked_signals = sorted(signals, key=lambda x: x.get('ml_quality', 0), reverse=True)
        
        return ranked_signals

# Example usage
if __name__ == "__main__":
    # Create sample training data
    sample_signals = [
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
        }
    ]
    
    # Initialize and train
    ml_filter = MLSignalFilter()
    ml_filter.train(sample_signals)
    
    # Test filtering
    test_signals = [
        {'z_score': 2.0, 'correlation': 0.9, 'spread_std': 0.04, 'volume_ratio': 1.1,
         'price_momentum': 0.03, 'volatility': 0.14, 'market_regime': 1, 'sentiment_score': 0.4,
         'timestamp': datetime.now()}
    ]
    
    filtered = ml_filter.filter_signals(test_signals)
    ranked = ml_filter.rank_signals(test_signals)
    
    print(f"Original signals: {len(test_signals)}")
    print(f"Filtered signals: {len(filtered)}")
    print(f"Top ranked signal quality: {ranked[0].get('ml_quality', 0):.3f}") 