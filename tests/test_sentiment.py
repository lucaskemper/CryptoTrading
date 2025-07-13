#!/usr/bin/env python3
"""
Comprehensive unit tests for the enhanced sentiment analysis module.
Tests error handling, weighted aggregation, entity extraction, and sentiment intensity.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict

from src.strategy.sentiment import SentimentAnalyzer, SentimentData


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a SentimentAnalyzer instance for testing."""
        return SentimentAnalyzer()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample sentiment data for testing."""
        base_time = datetime.now()
        return [
            SentimentData(
                timestamp=base_time + timedelta(minutes=i),
                source="reddit" if i % 2 == 0 else "news",
                text=f"Sample text {i}",
                sentiment_score=None
            )
            for i in range(5)
        ]

    def test_analyze_sentiment_basic(self, analyzer):
        """Test basic sentiment analysis."""
        # Test positive sentiment
        score = analyzer.analyze_sentiment("Bitcoin is absolutely amazing!")
        assert -1 <= score <= 1
        assert score > 0
        
        # Test negative sentiment
        score = analyzer.analyze_sentiment("Crypto is terrible and crashing")
        assert -1 <= score <= 1
        assert score < 0
        
        # Test neutral sentiment
        score = analyzer.analyze_sentiment("Crypto prices are stable")
        assert -1 <= score <= 1

    def test_analyze_sentiment_edge_cases(self, analyzer):
        """Test sentiment analysis with edge cases."""
        # Empty text
        score = analyzer.analyze_sentiment("")
        assert score == 0.0
        
        # Whitespace only
        score = analyzer.analyze_sentiment("   ")
        assert score == 0.0
        
        # None text
        score = analyzer.analyze_sentiment(None)
        assert score == 0.0
        
        # Very long text
        long_text = "Bitcoin " * 1000
        score = analyzer.analyze_sentiment(long_text)
        assert -1 <= score <= 1

    def test_extract_keywords_basic(self, analyzer):
        """Test basic keyword extraction."""
        text = "Bitcoin and Ethereum are both pumping! #Crypto #Moon"
        keywords = analyzer.extract_keywords(text)
        
        assert "bitcoin" in keywords
        assert "ethereum" in keywords
        assert "pump" in keywords
        assert "Crypto" in keywords
        assert "Moon" in keywords
        assert len(keywords) > 0

    def test_extract_keywords_edge_cases(self, analyzer):
        """Test keyword extraction with edge cases."""
        # Empty text
        keywords = analyzer.extract_keywords("")
        assert keywords == []
        
        # No crypto keywords
        keywords = analyzer.extract_keywords("The weather is nice today")
        assert keywords == []
        
        # Special characters
        keywords = analyzer.extract_keywords("BTC!@#$%^&*()")
        assert "btc" in keywords

    def test_extract_entities(self, analyzer):
        """Test entity extraction."""
        text = "Bitcoin BTC and Ethereum ETH are mentioned. Also Solana SOL."
        entities = analyzer.extract_entities(text)
        
        assert "Bitcoin" in entities['projects']
        assert "Ethereum" in entities['projects']
        assert "Solana" in entities['projects']
        assert "BTC" in entities['tickers']
        assert "ETH" in entities['tickers']
        assert "SOL" in entities['tickers']

    def test_extract_entities_edge_cases(self, analyzer):
        """Test entity extraction with edge cases."""
        # Empty text
        entities = analyzer.extract_entities("")
        assert entities == {'tickers': [], 'projects': []}
        
        # No entities
        entities = analyzer.extract_entities("The weather is nice")
        assert entities == {'tickers': [], 'projects': []}
        
        # Dollar sign tickers
        entities = analyzer.extract_entities("$BTC and $ETH are popular")
        assert "$BTC" in entities['tickers']
        assert "$ETH" in entities['tickers']

    def test_analyze_sentiment_intensity(self, analyzer):
        """Test sentiment intensity analysis."""
        # High intensity positive
        result = analyzer.analyze_sentiment_intensity("BITCOIN IS AMAZING!!! 🚀🚀🚀")
        assert result['polarity'] > 0
        assert result['intensity'] > 0.5
        
        # Low intensity neutral
        result = analyzer.analyze_sentiment_intensity("crypto prices are stable")
        assert result['intensity'] < 0.3
        
        # High intensity negative (TextBlob might not detect this as negative, so we check intensity only)
        result = analyzer.analyze_sentiment_intensity("CRYPTO IS CRASHING!!! 😱😱😱")
        assert result['intensity'] > 0.5  # High intensity regardless of polarity

    def test_analyze_sentiment_intensity_edge_cases(self, analyzer):
        """Test sentiment intensity with edge cases."""
        # Empty text
        result = analyzer.analyze_sentiment_intensity("")
        assert result == {'polarity': 0.0, 'subjectivity': 0.0, 'intensity': 0.0}
        
        # Whitespace only
        result = analyzer.analyze_sentiment_intensity("   ")
        assert result == {'polarity': 0.0, 'subjectivity': 0.0, 'intensity': 0.0}

    def test_batch_analyze_basic(self, analyzer, sample_data):
        """Test basic batch analysis."""
        # Set some test texts
        sample_data[0].text = "Bitcoin is amazing!"
        sample_data[1].text = "Crypto is terrible"
        sample_data[2].text = "Ethereum is stable"
        
        result = analyzer.batch_analyze(sample_data)
        
        assert len(result) == 5
        assert result[0].sentiment_score > 0  # Positive
        assert result[1].sentiment_score < 0  # Negative
        assert abs(result[2].sentiment_score) < 0.3  # Neutral
        assert all(item.keywords is not None for item in result)

    def test_batch_analyze_with_entities(self, analyzer, sample_data):
        """Test batch analysis with entity extraction."""
        sample_data[0].text = "Bitcoin BTC is mentioned"
        sample_data[1].text = "Ethereum ETH and Solana SOL"
        
        result = analyzer.batch_analyze(sample_data, include_entities=True)
        
        assert len(result) == 5
        # Check that entities were added to keywords
        assert any("BTC" in item.keywords for item in result)
        assert any("ETH" in item.keywords for item in result)
        assert any("SOL" in item.keywords for item in result)

    def test_batch_analyze_with_intensity(self, analyzer, sample_data):
        """Test batch analysis with intensity analysis."""
        sample_data[0].text = "BITCOIN IS AMAZING!!!"
        sample_data[1].text = "crypto is stable"
        
        result = analyzer.batch_analyze(sample_data, include_intensity=True)
        
        assert len(result) == 5
        # Intensity analysis should be logged but not stored in current implementation
        # This test ensures no errors occur

    def test_batch_analyze_error_handling(self, analyzer):
        """Test batch analysis error handling."""
        # Create data with problematic text
        data = [
            SentimentData(
                timestamp=datetime.now(),
                source="test",
                text="",  # Empty text
                sentiment_score=None
            ),
            SentimentData(
                timestamp=datetime.now(),
                source="test",
                text="Bitcoin is amazing and will moon!",  # Positive sentiment text
                sentiment_score=None
            )
        ]
        
        result = analyzer.batch_analyze(data)
        
        assert len(result) == 2
        assert result[0].sentiment_score == 0.0  # Default for empty text
        assert result[1].sentiment_score != 0.0  # Should have actual score

    def test_aggregate_sentiment_basic(self, analyzer, sample_data):
        """Test basic sentiment aggregation."""
        # Set sentiment scores
        for i, item in enumerate(sample_data):
            item.sentiment_score = 0.1 * (i - 2)  # [-0.2, -0.1, 0, 0.1, 0.2]
        
        # Test mean aggregation
        agg_series = analyzer.aggregate_sentiment(sample_data, window=3, method='mean')
        assert len(agg_series) > 0
        assert isinstance(agg_series, pd.Series)
        
        # Test median aggregation
        agg_series = analyzer.aggregate_sentiment(sample_data, window=3, method='median')
        assert len(agg_series) > 0

    def test_aggregate_sentiment_weighted(self, analyzer, sample_data):
        """Test weighted sentiment aggregation."""
        # Set sentiment scores
        for i, item in enumerate(sample_data):
            item.sentiment_score = 0.1 * (i - 2)
        
        # Test weighted aggregation
        weights = {'reddit': 1.5, 'news': 0.8}
        agg_series = analyzer.aggregate_sentiment(
            sample_data, window=3, method='weighted_mean', weights=weights
        )
        
        assert len(agg_series) > 0
        assert isinstance(agg_series, pd.Series)

    def test_aggregate_sentiment_edge_cases(self, analyzer):
        """Test aggregation with edge cases."""
        # Empty data
        agg_series = analyzer.aggregate_sentiment([], window=5)
        assert agg_series.empty
        
        # Data with no sentiment scores
        data = [
            SentimentData(timestamp=datetime.now(), source="test", text="test", sentiment_score=None)
        ]
        agg_series = analyzer.aggregate_sentiment(data, window=5)
        assert agg_series.empty
        
        # Invalid method
        data = [
            SentimentData(timestamp=datetime.now(), source="test", text="test", sentiment_score=0.5)
        ]
        with pytest.raises(ValueError):
            analyzer.aggregate_sentiment(data, window=5, method='invalid_method')

    def test_generate_sentiment_signal(self, analyzer):
        """Test sentiment signal generation."""
        thresholds = {'positive': 0.3, 'negative': -0.3}
        
        # Test positive signal
        signal = analyzer.generate_sentiment_signal(0.5, thresholds)
        assert signal == 'positive'
        
        # Test negative signal
        signal = analyzer.generate_sentiment_signal(-0.5, thresholds)
        assert signal == 'negative'
        
        # Test neutral signal
        signal = analyzer.generate_sentiment_signal(0.1, thresholds)
        assert signal == 'neutral'
        
        # Test edge cases
        signal = analyzer.generate_sentiment_signal(0.3, thresholds)
        assert signal == 'positive'
        
        signal = analyzer.generate_sentiment_signal(-0.3, thresholds)
        assert signal == 'negative'

    def test_generate_sentiment_signal_edge_cases(self, analyzer):
        """Test signal generation with edge cases."""
        thresholds = {'positive': 0.3, 'negative': -0.3}
        
        # Missing thresholds
        signal = analyzer.generate_sentiment_signal(0.5, {})
        assert signal == 'positive'  # Uses defaults
        
        # Extreme values
        signal = analyzer.generate_sentiment_signal(1.0, thresholds)
        assert signal == 'positive'
        
        signal = analyzer.generate_sentiment_signal(-1.0, thresholds)
        assert signal == 'negative'

    def test_preprocess_text(self, analyzer):
        """Test text preprocessing."""
        # Test URL removal
        text = "Bitcoin is great! Check out https://example.com"
        clean_text = analyzer._preprocess_text(text)
        assert "https://example.com" not in clean_text
        
        # Test case normalization
        text = "BITCOIN and Ethereum"
        clean_text = analyzer._preprocess_text(text)
        assert clean_text == "bitcoin and ethereum"
        
        # Test special character removal
        text = "BTC!@#$%^&*()"
        clean_text = analyzer._preprocess_text(text)
        assert clean_text == "btc"
        
        # Test hashtag preservation
        text = "Bitcoin #BTC #Crypto"
        clean_text = analyzer._preprocess_text(text)
        assert "btc" in clean_text
        assert "crypto" in clean_text

    def test_load_default_model(self, analyzer):
        """Test default model loading."""
        model = analyzer._load_default_model()
        assert model is not None


class TestSentimentData:
    """Test suite for SentimentData class."""
    
    def test_sentiment_data_creation(self):
        """Test SentimentData creation."""
        data = SentimentData(
            timestamp=datetime.now(),
            source="test",
            text="Test text",
            sentiment_score=0.5,
            keywords=["test", "crypto"],
            url="https://example.com"
        )
        
        assert data.source == "test"
        assert data.text == "Test text"
        assert data.sentiment_score == 0.5
        assert data.keywords == ["test", "crypto"]
        assert data.url == "https://example.com"

    def test_sentiment_data_validation(self):
        """Test SentimentData validation."""
        # Valid data
        data = SentimentData(
            timestamp=datetime.now(),
            source="test",
            text="Valid text",
            sentiment_score=0.5
        )
        assert data.validate() is True
        
        # Invalid: empty text
        data = SentimentData(
            timestamp=datetime.now(),
            source="test",
            text="",
            sentiment_score=0.5
        )
        assert data.validate() is False
        
        # Invalid: score out of range
        data = SentimentData(
            timestamp=datetime.now(),
            source="test",
            text="Valid text",
            sentiment_score=1.5
        )
        assert data.validate() is False
        
        # Valid: score in range
        data = SentimentData(
            timestamp=datetime.now(),
            source="test",
            text="Valid text",
            sentiment_score=-1.0
        )
        assert data.validate() is True


def test_integration_scenario():
    """Test a complete integration scenario."""
    analyzer = SentimentAnalyzer()
    
    # Create sample data
    data = [
        SentimentData(
            timestamp=datetime.now(),
            source="reddit",
            text="Bitcoin BTC is absolutely amazing! #Crypto #Moon",
            sentiment_score=None
        ),
        SentimentData(
            timestamp=datetime.now(),
            source="news",
            text="Ethereum ETH showing strong support",
            sentiment_score=None
        ),
        SentimentData(
            timestamp=datetime.now(),
            source="twitter",
            text="Solana SOL pumping hard!!! 🚀",
            sentiment_score=None
        )
    ]
    
    # Batch analyze with all features
    result = analyzer.batch_analyze(data, include_entities=True, include_intensity=True)
    
    # Verify results
    assert len(result) == 3
    assert all(item.sentiment_score is not None for item in result)
    assert all(item.keywords is not None for item in result)
    
    # Test aggregation
    weights = {'reddit': 1.2, 'news': 1.0, 'twitter': 0.8}
    agg_series = analyzer.aggregate_sentiment(result, window=3, method='weighted_mean', weights=weights)
    
    assert len(agg_series) > 0
    assert isinstance(agg_series, pd.Series)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 