#!/usr/bin/env python3
"""
Multi-Asset Data Collection Script
Collects historical data for multiple cryptocurrencies from multiple exchanges
"""

import asyncio
import pandas as pd
import ccxt
import os
from datetime import datetime, timedelta
import logging
from typing import List, Dict
import time
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiAssetDataCollector:
    """Collect historical data for multiple cryptocurrency assets"""
    
    def __init__(self):
        self.exchanges = {
            'binance': ccxt.binance(),
            'kraken': ccxt.kraken(),
            'coinbase': ccxt.coinbase()
        }
        
        # Define assets to collect
        self.assets = [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT",
            "LINK/USDT", "MATIC/USDT", "AVAX/USDT", "UNI/USDT", "ATOM/USDT",
            "LTC/USDT", "XRP/USDT", "BCH/USDT", "ETC/USDT", "FIL/USDT",
            "NEAR/USDT", "ALGO/USDT", "VET/USDT", "ICP/USDT", "FTM/USDT"
        ]
        
        # Create data directories
        self._create_directories()
    
    def _create_directories(self):
        """Create directories for each asset"""
        for asset in self.assets:
            symbol = asset.split('/')[0]  # Extract symbol (e.g., BTC from BTC/USDT)
            dir_path = f"data/market_data_{symbol}"
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    async def collect_historical_data(self, days: int = 30):
        """Collect historical data for all assets"""
        logger.info(f"Starting data collection for {len(self.assets)} assets...")
        
        for asset in self.assets:
            try:
                logger.info(f"Collecting data for {asset}...")
                await self._collect_asset_data(asset, days)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error collecting data for {asset}: {e}")
                continue
    
    async def _collect_asset_data(self, asset: str, days: int):
        """Collect data for a specific asset"""
        symbol = asset.split('/')[0]
        
        # Try different exchanges
        for exchange_name, exchange in self.exchanges.items():
            try:
                logger.info(f"Trying {exchange_name} for {asset}...")
                
                # Get historical OHLCV data
                since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
                ohlcv = await exchange.fetch_ohlcv(asset, '1h', since=since, limit=1000)
                
                if ohlcv:
                    # Convert to DataFrame
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Save to CSV
                    filename = f"data/market_data_{symbol}/USDT_{exchange_name}.csv"
                    df.to_csv(filename, index=False)
                    
                    logger.info(f"Saved {len(df)} records for {asset} from {exchange_name}")
                    break  # Success, move to next asset
                    
            except Exception as e:
                logger.warning(f"Failed to get {asset} from {exchange_name}: {e}")
                continue
    
    async def collect_realtime_data(self, duration_minutes: int = 60):
        """Collect real-time data for all assets"""
        logger.info(f"Starting real-time data collection for {duration_minutes} minutes...")
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        while datetime.now() < end_time:
            for asset in self.assets:
                try:
                    await self._collect_realtime_asset_data(asset)
                except Exception as e:
                    logger.error(f"Error collecting real-time data for {asset}: {e}")
            
            await asyncio.sleep(60)  # Collect every minute
    
    async def _collect_realtime_asset_data(self, asset: str):
        """Collect real-time data for a specific asset"""
        symbol = asset.split('/')[0]
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                # Get current ticker
                ticker = await exchange.fetch_ticker(asset)
                
                # Create data point
                data_point = {
                    'timestamp': datetime.now(),
                    'price': ticker['last'],
                    'volume': ticker['baseVolume'],
                    'bid': ticker['bid'],
                    'ask': ticker['ask']
                }
                
                # Append to CSV
                filename = f"data/market_data_{symbol}/realtime_{exchange_name}.csv"
                
                df = pd.DataFrame([data_point])
                if os.path.exists(filename):
                    df.to_csv(filename, mode='a', header=False, index=False)
                else:
                    df.to_csv(filename, index=False)
                
                logger.info(f"Real-time data for {asset}: ${ticker['last']:.2f}")
                break
                
            except Exception as e:
                logger.warning(f"Failed to get real-time data for {asset} from {exchange_name}: {e}")
                continue
    
    def generate_sample_data(self):
        """Generate sample data for testing"""
        logger.info("Generating sample data for all assets...")
        
        for asset in self.assets:
            symbol = asset.split('/')[0]
            
            # Generate 1000 data points
            dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')
            
            # Generate realistic price data
            base_price = {
                'BTC': 50000, 'ETH': 3000, 'SOL': 100, 'ADA': 0.5, 'DOT': 7,
                'LINK': 15, 'MATIC': 0.8, 'AVAX': 25, 'UNI': 10, 'ATOM': 8,
                'LTC': 70, 'XRP': 0.5, 'BCH': 250, 'ETC': 30, 'FIL': 5,
                'NEAR': 3, 'ALGO': 0.2, 'VET': 0.03, 'ICP': 10, 'FTM': 0.3
            }
            
            price = base_price.get(symbol, 10)
            
            # Generate price series with some volatility
            np.random.seed(hash(symbol) % 1000)  # Consistent seed per symbol
            returns = np.random.normal(0, 0.02, 1000)  # 2% daily volatility
            prices = [price]
            
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 0.01))  # Ensure positive price
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'close': prices,
                'volume': np.random.uniform(1000, 10000, 1000)
            })
            
            # Save to CSV
            filename = f"data/market_data_{symbol}/USDT_sample.csv"
            df.to_csv(filename, index=False)
            
            logger.info(f"Generated sample data for {asset}: {len(df)} records")

async def main():
    """Main function"""
    collector = MultiAssetDataCollector()
    
    # Generate sample data for testing
    collector.generate_sample_data()
    
    # Uncomment to collect real data (requires API keys)
    # await collector.collect_historical_data(days=30)
    
    logger.info("Data collection completed!")

if __name__ == "__main__":
    asyncio.run(main()) 