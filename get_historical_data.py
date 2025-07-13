#!/usr/bin/env python3
"""
Script to collect historical data using the existing data_collector.py pipeline.
This will fetch historical OHLCV data from exchanges and save it to CSV files
for use in backtesting.
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_collector import data_collector
from datetime import datetime, timedelta
import pandas as pd
import os

def collect_historical_data(symbols=None, exchanges=None, limit=5000, timeframe='1h'):
    """Collect historical data for backtesting."""
    print("Starting historical data collection...")
    
    # Define symbols and exchanges
    if symbols is None:
        symbols = ['ETH/USDT', 'BTC/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT', 'LINK/USDT']
    if exchanges is None:
        exchanges = ['binance', 'kraken']
    
    # Create data directories if they don't exist
    for symbol in symbols:
        symbol_name = symbol.split('/')[0]  # ETH, BTC, SOL, etc.
        os.makedirs(f"data/market_data_{symbol_name}", exist_ok=True)
    
    total_records = 0
    
    for exchange in exchanges:
        print(f"\nCollecting data from {exchange}...")
        
        for symbol in symbols:
            symbol_name = symbol.split('/')[0]  # ETH, BTC, SOL, etc.
            csv_path = f"data/market_data_{symbol_name}/USDT_{exchange}.csv"
            
            print(f"  Fetching {symbol} data...")
            
            try:
                # Get historical OHLCV data with increased limit
                df = data_collector.get_historical_data(symbol, exchange, timeframe=timeframe, limit=limit)
                
                if not df.empty:
                    # Save to CSV
                    df.to_csv(csv_path, index=False)
                    print(f"    Saved {len(df)} records to {csv_path}")
                    total_records += len(df)
                else:
                    print(f"    No data received for {symbol} from {exchange}")
                    
            except Exception as e:
                print(f"    Error collecting {symbol} from {exchange}: {e}")
    
    print(f"\nData collection completed! Total records: {total_records}")
    print("Data saved to data/market_data_*/USDT_*.csv files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect historical crypto data')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols (e.g., BTC,ETH,SOL)')
    parser.add_argument('--exchanges', type=str, help='Comma-separated exchanges (e.g., binance,kraken)')
    parser.add_argument('--limit', type=int, default=5000, help='Number of records to fetch per symbol (default: 5000)')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe (1m, 5m, 15m, 1h, 4h, 1d)')
    
    args = parser.parse_args()
    
    # Parse symbols and exchanges
    symbols = None
    if args.symbols:
        symbols = [f"{s.strip()}/USDT" for s in args.symbols.split(',')]
    
    exchanges = None
    if args.exchanges:
        exchanges = [e.strip() for e in args.exchanges.split(',')]
    
    collect_historical_data(symbols, exchanges, args.limit, args.timeframe) 