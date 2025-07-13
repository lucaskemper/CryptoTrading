"""
Data Simulator for Backtesting

Generates realistic historical market data for backtesting crypto trading strategies.
Supports multiple assets, exchanges, and data frequencies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Iterator, Tuple, Any
import logging
from pathlib import Path
import yaml
import os

from src.utils.logger import logger


class DataSimulator:
    """
    Simulates historical market data for backtesting.
    
    Features:
    - Realistic price movements with volatility and trends
    - Multi-exchange data with spread simulation
    - Sentiment data generation
    - Configurable data frequency and quality
    """
    
    def __init__(self, config):
        """Initialize the data simulator."""
        self.config = config
        self.logger = logger
        
        # Data storage
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.sentiment_data: Dict[str, pd.DataFrame] = {}
        
        # Simulation parameters
        self.current_timestamp: Optional[datetime] = None
        self.data_iterator: Optional[Iterator] = None
        
        # Price simulation parameters
        self.volatility = 0.02  # 2% daily volatility
        self.trend = 0.0001  # Slight upward trend
        self.mean_reversion = 0.1  # Mean reversion strength
        self.correlation_matrix: Optional[np.ndarray] = None
        
        self.logger.info("DataSimulator initialized")
    
    def initialize(self):
        """Initialize the data simulation."""
        self.logger.info("Initializing data simulation...")
        
        # Generate market data for all symbols
        for symbol in self.config.symbols:
            self._generate_market_data(symbol)
        
        # Generate sentiment data if enabled
        if self.config.sentiment_enabled:
            self._generate_sentiment_data()
        
        # Initialize data iterator
        self._initialize_data_iterator()
        
        self.logger.info("Data simulation initialized successfully")
    
    def _generate_market_data(self, symbol: str):
        """Load real market data from CSV if available, else generate synthetic."""
        self.logger.info(f"Loading real market data for {symbol}")
        symbol_map = {
            'ETH/USDT': 'data/market_data_ETH/USDT_binance.csv',
            'BTC/USDT': 'data/market_data_BTC/USDT_binance.csv',
            'SOL/USDT': 'data/market_data_SOL/USDT_binance.csv',
        }
        file_path = symbol_map.get(symbol)
        if file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            # Filter by date range
            df = df[(df['timestamp'] >= self.config.start_date) & (df['timestamp'] <= self.config.end_date)]
            df = df.sort_values('timestamp')
            # Ensure required columns
            required_cols = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
            if not required_cols.issubset(df.columns):
                self.logger.warning(f"File {file_path} missing required columns, using synthetic data.")
                return self._generate_synthetic_market_data(symbol)
            # Keep timestamp as a column instead of setting it as index
            self.market_data[symbol] = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            self.logger.info(f"Loaded {len(df)} rows for {symbol} from {file_path}")
        else:
            self.logger.warning(f"No real data file found for {symbol}, using synthetic data.")
            self._generate_synthetic_market_data(symbol)
    
    def _generate_synthetic_market_data(self, symbol: str):
        """Generate synthetic market data when real data is not available."""
        # Generate timestamps
        timestamps = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq=self.config.data_frequency
        )
        
        # Generate price and volume series
        price_series = self._generate_price_series(symbol, timestamps)
        volume_series = self._generate_volume_series(timestamps)
        
        # Generate OHLCV data
        ohlcv_data = self._generate_ohlcv_data(price_series, volume_series, timestamps)
        
        # Store the data
        self.market_data[symbol] = ohlcv_data
    
    def _generate_price_series(self, symbol: str, timestamps: pd.DatetimeIndex) -> pd.Series:
        """Generate realistic price series with volatility and trends."""
        # Base price based on symbol
        base_prices = {
            'ETH/USDT': 2000,
            'SOL/USDT': 100,
            'BTC/USDT': 40000,
            'ADA/USDT': 0.5,
            'DOT/USDT': 7,
            'LINK/USDT': 15
        }
        
        base_price = base_prices.get(symbol, 100)
        
        # Generate returns with realistic characteristics
        n = len(timestamps)
        
        # Random walk with volatility clustering
        returns = np.random.normal(0, self.volatility / np.sqrt(252), n)
        
        # Add volatility clustering (GARCH-like effect)
        volatility = np.ones(n) * self.volatility
        for i in range(1, n):
            volatility[i] = 0.95 * volatility[i-1] + 0.05 * abs(returns[i-1])
            returns[i] = np.random.normal(0, volatility[i] / np.sqrt(252))
        
        # Add trend
        trend = np.linspace(0, self.trend * n, n)
        returns += trend / n
        
        # Add mean reversion
        for i in range(1, n):
            price_deviation = (np.exp(np.sum(returns[:i])) - 1)
            returns[i] -= self.mean_reversion * price_deviation / n
        
        # Convert to prices
        cumulative_returns = np.cumsum(returns)
        prices = base_price * np.exp(cumulative_returns)
        
        return pd.Series(prices, index=timestamps)
    
    def _generate_volume_series(self, timestamps: pd.DatetimeIndex) -> pd.Series:
        """Generate realistic volume data."""
        n = len(timestamps)
        
        # Base volume with daily patterns
        base_volume = 1000000  # 1M base volume
        
        # Add daily patterns (higher volume during trading hours)
        hour_of_day = timestamps.hour
        daily_pattern = 1 + 0.5 * np.sin(2 * np.pi * hour_of_day / 24)
        
        # Add weekly patterns (lower volume on weekends)
        day_of_week = timestamps.dayofweek
        weekly_pattern = 1 - 0.3 * (day_of_week >= 5)  # Weekend effect
        
        # Add random noise
        noise = np.random.lognormal(0, 0.3, n)
        
        # Combine all effects
        volume = base_volume * daily_pattern * weekly_pattern * noise
        
        return pd.Series(volume, index=timestamps)
    
    def _generate_ohlcv_data(self, price_series: pd.Series, volume_series: pd.Series, 
                            timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate OHLCV data from price series."""
        n = len(timestamps)
        
        # Generate high, low, open, close from price series
        close_prices = price_series.values
        
        # Generate realistic OHLC data
        data = []
        for i in range(n):
            close = close_prices[i]
            
            # Generate realistic high/low spread
            spread_pct = np.random.uniform(0.001, 0.01)  # 0.1% to 1% spread
            
            if i == 0:
                open_price = close
            else:
                open_price = data[i-1]['close']
            
            # Generate high and low
            price_range = close * spread_pct
            high = close + np.random.uniform(0, price_range)
            low = close - np.random.uniform(0, price_range)
            
            # Ensure high >= close >= low
            high = max(high, close)
            low = min(low, close)
            
            data.append({
                'timestamp': timestamps[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume_series.iloc[i]
            })
        
        return pd.DataFrame(data)
    
    def _generate_sentiment_data(self):
        """Generate sentiment data for backtesting."""
        self.logger.info("Generating sentiment data...")
        
        # Calculate sentiment frequency
        if self.config.sentiment_frequency == '1h':
            freq_seconds = 3600
        elif self.config.sentiment_frequency == '4h':
            freq_seconds = 14400
        elif self.config.sentiment_frequency == '1d':
            freq_seconds = 86400
        else:
            freq_seconds = 14400  # Default to 4h
        
        total_seconds = (self.config.end_date - self.config.start_date).total_seconds()
        num_points = int(total_seconds / freq_seconds)
        
        # Generate timestamps
        timestamps = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            periods=num_points
        )
        
        # Generate sentiment scores
        sentiment_scores = np.random.normal(0, 0.3, num_points)  # Mean 0, std 0.3
        
        # Add some structure to sentiment (correlation with price movements)
        for i in range(1, num_points):
            # Add some persistence
            sentiment_scores[i] = 0.7 * sentiment_scores[i-1] + 0.3 * sentiment_scores[i]
            
            # Add some correlation with market movements
            if i < len(self.market_data):
                # Get price change for first symbol
                symbol = list(self.market_data.keys())[0]
                price_change = self.market_data[symbol].iloc[i]['close'] - self.market_data[symbol].iloc[i-1]['close']
                sentiment_scores[i] += 0.1 * np.sign(price_change)
        
        # Create sentiment data
        sentiment_data = pd.DataFrame({
            'timestamp': timestamps,
            'sentiment_score': sentiment_scores,
            'confidence': np.random.uniform(0.5, 1.0, num_points),
            'source': 'simulated'
        })
        
        self.sentiment_data['overall'] = sentiment_data
        
        self.logger.info(f"Generated {len(sentiment_data)} sentiment data points")
    
    def _initialize_data_iterator(self):
        """Initialize the data iterator for simulation."""
        # Get all unique timestamps
        all_timestamps = set()
        for symbol, data in self.market_data.items():
            all_timestamps.update(data['timestamp'])
        
        # Add sentiment timestamps
        if self.sentiment_data:
            all_timestamps.update(self.sentiment_data['overall']['timestamp'])
        
        # Sort timestamps
        sorted_timestamps = sorted(all_timestamps)
        
        # Create iterator
        self.data_iterator = self._create_data_iterator(sorted_timestamps)
    
    def _create_data_iterator(self, timestamps: List[datetime]) -> Iterator[Tuple[datetime, Dict, Optional[Dict]]]:
        """Create an iterator that yields (timestamp, market_data, sentiment_data)."""
        for timestamp in timestamps:
            # Get market data for this timestamp
            market_data = {}
            for symbol, data in self.market_data.items():
                try:
                    # Find data for this timestamp
                    symbol_data = data[data['timestamp'] == timestamp]
                    if not symbol_data.empty:
                        market_data[symbol] = symbol_data.iloc[0].to_dict()
                except KeyError:
                    continue
            # Get sentiment data for this timestamp
            sentiment_data = None
            if self.sentiment_data:
                sentiment_data_df = self.sentiment_data['overall'][
                    self.sentiment_data['overall']['timestamp'] == timestamp
                ]
                if not sentiment_data_df.empty:
                    sentiment_data = sentiment_data_df.iloc[0].to_dict()
            yield timestamp, market_data, sentiment_data
    
    def get_data_iterator(self) -> Iterator[Tuple[datetime, Dict, Optional[Dict]]]:
        """Get the data iterator for simulation."""
        if self.data_iterator is None:
            self._initialize_data_iterator()
        
        return self.data_iterator
    
    def get_market_data(self, symbol: str, start_date: Optional[datetime] = None, 
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get market data for a specific symbol and time period."""
        if symbol not in self.market_data:
            raise ValueError(f"No data available for symbol {symbol}")
        
        data = self.market_data[symbol]
        
        if start_date:
            data = data[data.index >= start_date]
        
        if end_date:
            data = data[data.index <= end_date]
        
        return data.copy()
    
    def get_sentiment_data(self, start_date: Optional[datetime] = None, 
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get sentiment data for a specific time period."""
        if not self.sentiment_data:
            return pd.DataFrame()
        
        data = self.sentiment_data['overall']
        
        if start_date:
            data = data[data['timestamp'] >= start_date]
        
        if end_date:
            data = data[data['timestamp'] <= end_date]
        
        return data.copy()
    
    def add_correlation(self, symbol1: str, symbol2: str, correlation: float):
        """Add correlation between two symbols."""
        if self.correlation_matrix is None:
            # Initialize correlation matrix
            symbols = list(self.market_data.keys())
            n = len(symbols)
            self.correlation_matrix = np.eye(n)
            self.symbol_to_index = {symbol: i for i, symbol in enumerate(symbols)}
        
        if symbol1 in self.symbol_to_index and symbol2 in self.symbol_to_index:
            i, j = self.symbol_to_index[symbol1], self.symbol_to_index[symbol2]
            self.correlation_matrix[i, j] = correlation
            self.correlation_matrix[j, i] = correlation
    
    def save_data(self, output_dir: str):
        """Save generated data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save market data
        for symbol, data in self.market_data.items():
            symbol_clean = symbol.replace('/', '_')
            data.to_csv(output_path / f"{symbol_clean}_market_data.csv", index=False)
        
        # Save sentiment data
        if self.sentiment_data:
            for source, data in self.sentiment_data.items():
                data.to_csv(output_path / f"{source}_sentiment_data.csv", index=False)
        
        self.logger.info(f"Data saved to {output_path}")

    def get_initial_data(self) -> Dict[str, Dict]:
        """Get initial market data for all symbols."""
        initial_data = {}
        
        for symbol in self.config.symbols:
            if symbol in self.market_data:
                # Get the first few data points for initialization
                symbol_data = self.market_data[symbol]
                if not symbol_data.empty:
                    # Take the first 100 data points for initialization
                    initial_rows = symbol_data.head(100)
                    initial_data[symbol] = {
                        'close': initial_rows['close'].tolist(),
                        'timestamp': initial_rows['timestamp'].tolist()
                    }
        
        return initial_data


def create_data_simulator(config) -> DataSimulator:
    """Factory function to create a data simulator."""
    return DataSimulator(config) 