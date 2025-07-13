import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.utils.logger import logger

@dataclass
class SentimentData:
    """
    Data structure for a single sentiment data point.
    """
    timestamp: datetime
    source: str
    text: str
    sentiment_score: Optional[float] = None
    keywords: Optional[List[str]] = None
    url: Optional[str] = None

    def validate(self) -> bool:
        """
        Validate sentiment data for non-empty text and valid score.
        """
        if not self.text or len(self.text.strip()) == 0:
            return False
        if self.sentiment_score is not None and not (-1 <= self.sentiment_score <= 1):
            return False
        return True

class SentimentAnalyzer:
    """
    Modular sentiment analysis and feature engineering for trading strategies.
    Supports batch processing, aggregation, and signal generation.
    """
    def __init__(self, model: Optional[Any] = None):
        self.model = model or self._load_default_model()
        self.crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol', 'crypto',
            'blockchain', 'defi', 'nft', 'altcoin', 'moon', 'pump', 'dump',
            'hodl', 'fomo', 'fud', 'bull', 'bear', 'mooning', 'crashing'
        ]
        
        # Entity extraction patterns
        self.ticker_patterns = [
            r'\b[A-Z]{2,5}\b',  # 2-5 letter tickers
            r'\$[A-Z]{2,5}\b',  # $BTC, $ETH style
        ]
        
        # Project name patterns
        self.project_patterns = [
            r'\bBitcoin\b', r'\bEthereum\b', r'\bSolana\b',
            r'\bCardano\b', r'\bPolygon\b', r'\bAvalanche\b',
            r'\bPolkadot\b', r'\bChainlink\b', r'\bUniswap\b'
        ]

    def _preprocess_text(self, text: str) -> str:
        """
        Clean and normalize text: remove URLs, emojis, special chars, lowercasing.
        """
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s#@]', '', text)  # Remove special chars except hashtags/mentions
        text = re.sub(r'[^\w\s]', '', text)  # Remove remaining special chars except spaces
        text = text.lower()
        return text.strip()

    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).
        Returns a normalized score between -1 and 1.
        """
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided for sentiment analysis")
            return 0.0
            
        try:
            # VADER works better with original text (preserves emojis, caps, etc.)
            # Only do minimal preprocessing for VADER
            if isinstance(self.model, SentimentIntensityAnalyzer):
                scores = self.model.polarity_scores(text)
                score = scores['compound']  # VADER compound score is already -1 to 1
            else:
                # Fallback to VADER if model is not properly initialized
                vader_analyzer = SentimentIntensityAnalyzer()
                scores = vader_analyzer.polarity_scores(text)
                score = scores['compound']
                
            # VADER compound score is already in [-1, 1] range, but validate anyway
            if not (-1 <= score <= 1):
                logger.warning(f"Sentiment score {score} out of expected range [-1, 1], clamping")
                score = max(-1.0, min(1.0, score))
                
            logger.debug(f"VADER sentiment score for text: {score}")
            return score
            
        except ImportError as e:
            logger.error(f"Required VADER sentiment model not available: {e}")
            return 0.0
        except AttributeError as e:
            logger.error(f"VADER model configuration error: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Unexpected error in VADER sentiment analysis: {e}")
            return 0.0

    def batch_analyze(self, data: List[SentimentData], include_entities: bool = False, 
                     include_intensity: bool = False) -> List[SentimentData]:
        """
        Analyze sentiment for a batch of SentimentData with optional entity extraction and intensity analysis.
        
        Args:
            data: List of SentimentData objects
            include_entities: Whether to extract entities (tickers, projects)
            include_intensity: Whether to analyze sentiment intensity
        """
        if not data:
            logger.warning("No data provided for batch analysis")
            return []
            
        processed_count = 0
        error_count = 0
        
        for item in data:
            try:
                # Basic sentiment analysis
                item.sentiment_score = self.analyze_sentiment(item.text)
                item.keywords = self.extract_keywords(item.text)
                
                # Optional entity extraction
                if include_entities:
                    entities = self.extract_entities(item.text)
                    # Store entities in keywords for now (could extend SentimentData later)
                    if entities['tickers']:
                        item.keywords.extend(entities['tickers'])
                    if entities['projects']:
                        item.keywords.extend(entities['projects'])
                
                # Optional intensity analysis
                if include_intensity:
                    intensity_data = self.analyze_sentiment_intensity(item.text)
                    # Store intensity data (could extend SentimentData later)
                    # For now, we'll log it
                    logger.debug(f"Intensity for {item.source}: {intensity_data}")
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing item from {item.source}: {e}")
                error_count += 1
                # Set default values for failed items
                item.sentiment_score = 0.0
                item.keywords = []
        
        logger.info(f"Batch analyzed {processed_count} sentiment items with {error_count} errors.")
        return data

    def aggregate_sentiment(self, data: List[SentimentData], window: int = 10, method: str = 'mean', 
                          weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        Aggregate sentiment scores over a rolling window with optional weighting.
        
        Args:
            data: List of SentimentData objects
            window: Rolling window size
            method: Aggregation method ('mean', 'median', 'weighted_mean')
            weights: Optional weights for sources (e.g., {'reddit': 0.8, 'news': 1.2})
        """
        if not data:
            logger.warning("No sentiment data provided for aggregation.")
            return pd.Series(dtype=float)
            
        # Create DataFrame with sentiment scores
        df_data = []
        for d in data:
            if d.sentiment_score is not None:
                df_data.append({
                    'timestamp': d.timestamp,
                    'score': d.sentiment_score,
                    'source': d.source
                })
        
        if not df_data:
            logger.warning("No valid sentiment scores found in data.")
            return pd.Series(dtype=float)
            
        df = pd.DataFrame(df_data)
        df = df.sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
        
        if method == 'mean':
            agg = df['score'].rolling(window=window, min_periods=1).mean()
        elif method == 'median':
            agg = df['score'].rolling(window=window, min_periods=1).median()
        elif method == 'weighted_mean':
            if weights is None:
                logger.warning("No weights provided for weighted_mean, falling back to mean")
                agg = df['score'].rolling(window=window, min_periods=1).mean()
            else:
                # Apply source weights
                df['weight'] = df['source'].map(lambda x: weights.get(x, 1.0))
                df['weighted_score'] = df['score'] * df['weight']
                
                # Calculate weighted rolling mean
                weighted_sum = df['weighted_score'].rolling(window=window, min_periods=1).sum()
                weight_sum = df['weight'].rolling(window=window, min_periods=1).sum()
                agg = weighted_sum / weight_sum
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
            
        logger.info(f"Aggregated sentiment over window={window} using {method}.")
        return agg

    def generate_sentiment_signal(self, score: float, thresholds: Dict[str, float]) -> str:
        """
        Generate a sentiment signal ('positive', 'negative', 'neutral') based on thresholds.
        Example thresholds: {'positive': 0.3, 'negative': -0.3}
        """
        if score >= thresholds.get('positive', 0.3):
            signal = 'positive'
        elif score <= thresholds.get('negative', -0.3):
            signal = 'negative'
        else:
            signal = 'neutral'
        logger.debug(f"Generated sentiment signal: {signal} for score: {score}")
        return signal

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract crypto-relevant keywords, hashtags, and mentions from text.
        """
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided for keyword extraction")
            return []
            
        try:
            text_lower = text.lower()
            found_keywords = [kw for kw in self.crypto_keywords if kw in text_lower]
            hashtags = re.findall(r'#(\w+)', text)
            mentions = re.findall(r'@(\w+)', text)
            found_keywords.extend(hashtags)
            found_keywords.extend(mentions)
            
            # Remove duplicates and filter out empty strings
            unique_keywords = list(set([kw for kw in found_keywords if kw.strip()]))
            
            logger.debug(f"Extracted {len(unique_keywords)} keywords from text")
            return unique_keywords
            
        except re.error as e:
            logger.error(f"Regex error in keyword extraction: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in keyword extraction: {e}")
            return []

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities (tickers, project names) from text.
        
        Returns:
            Dictionary with 'tickers' and 'projects' keys
        """
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided for entity extraction")
            return {'tickers': [], 'projects': []}
            
        try:
            entities = {'tickers': [], 'projects': []}
            
            # Extract tickers
            for pattern in self.ticker_patterns:
                matches = re.findall(pattern, text)
                entities['tickers'].extend(matches)
            
            # Extract project names
            for pattern in self.project_patterns:
                matches = re.findall(pattern, text)
                entities['projects'].extend(matches)
            
            # Remove duplicates and filter
            entities['tickers'] = list(set([t for t in entities['tickers'] if t.strip()]))
            entities['projects'] = list(set([p for p in entities['projects'] if p.strip()]))
            
            logger.debug(f"Extracted {len(entities['tickers'])} tickers and {len(entities['projects'])} projects")
            return entities
            
        except re.error as e:
            logger.error(f"Regex error in entity extraction: {e}")
            return {'tickers': [], 'projects': []}
        except Exception as e:
            logger.error(f"Unexpected error in entity extraction: {e}")
            return {'tickers': [], 'projects': []}

    def analyze_sentiment_intensity(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment intensity using VADER.
        
        Returns:
            Dictionary with 'polarity', 'subjectivity', and 'intensity' scores
        """
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided for intensity analysis")
            return {'polarity': 0.0, 'subjectivity': 0.0, 'intensity': 0.0}
            
        try:
            # Use VADER for sentiment analysis
            if isinstance(self.model, SentimentIntensityAnalyzer):
                scores = self.model.polarity_scores(text)
            else:
                vader_analyzer = SentimentIntensityAnalyzer()
                scores = vader_analyzer.polarity_scores(text)
            
            # VADER provides: neg, neu, pos, compound
            # We'll map these to our expected format
            polarity = scores['compound']  # Already -1 to 1
            
            # Calculate subjectivity as the sum of positive and negative scores
            # This gives us a measure of how strong the sentiment is
            subjectivity = scores['pos'] + scores['neg']
            
            # Calculate intensity based on VADER scores and text features
            intensity_indicators = 0
            intensity_indicators += text.count('!') * 0.1
            intensity_indicators += text.count('?') * 0.05
            intensity_indicators += sum(1 for c in text if c.isupper()) / len(text) * 0.2
            intensity_indicators += len(re.findall(r'\b[A-Z]{2,}\b', text)) * 0.1
            
            # Add VADER's intensity indicators
            intensity_indicators += abs(scores['compound']) * 0.5
            intensity_indicators += (scores['pos'] + scores['neg']) * 0.3
            
            # Normalize intensity to 0-1 range
            intensity = min(1.0, intensity_indicators)
            
            result = {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'intensity': intensity
            }
            
            logger.debug(f"VADER sentiment intensity analysis: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in VADER sentiment intensity analysis: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0, 'intensity': 0.0}

    def _load_default_model(self):
        """
        Load the default sentiment model (VADER).
        """
        try:
            return SentimentIntensityAnalyzer()
        except Exception as e:
            logger.error(f"Error loading default VADER sentiment model: {e}")
            return None

# Example usage (for documentation/testing):
if __name__ == "__main__":
    from datetime import datetime
    # Example data
    data = [
        SentimentData(timestamp=datetime.now(), source="reddit", text="Bitcoin is mooning! 🚀"),
        SentimentData(timestamp=datetime.now(), source="news", text="Crypto is crashing, this is terrible"),
        SentimentData(timestamp=datetime.now(), source="reddit", text="Ethereum and Solana are both doing well"),
        SentimentData(timestamp=datetime.now(), source="news", text="I'm neutral about crypto prices"),
    ]
    analyzer = SentimentAnalyzer()
    analyzed = analyzer.batch_analyze(data)
    agg = analyzer.aggregate_sentiment(analyzed, window=2)
    for d in analyzed:
        print(f"{d.source}: {d.text[:40]}... | Score: {d.sentiment_score:.3f} | Keywords: {d.keywords}")
    print("Aggregated sentiment:")
    print(agg)
    for d in analyzed:
        signal = analyzer.generate_sentiment_signal(d.sentiment_score, {'positive': 0.3, 'negative': -0.3})
        print(f"Signal for '{d.text[:30]}...': {signal}")
