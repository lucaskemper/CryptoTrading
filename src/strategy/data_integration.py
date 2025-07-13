"""
Data Integration Module

Connects statistical arbitrage strategy to data collector for real-time
and historical price data from multiple exchanges.
"""

import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import pandas as pd
from queue import Queue
import time

from data_collector import DataCollector, MarketData
from strategy.stat_arb import StatisticalArbitrage, create_stat_arb_strategy
from src.utils.logger import logger
from src.utils.config_loader import config


class StrategyDataIntegration:
    """
    Integrates statistical arbitrage strategy with data collector.
    
    Features:
    - Real-time price data streaming to strategy
    - Historical data loading for strategy initialization
    - Multi-exchange data aggregation
    - Automatic data validation and error handling
    - Configurable data update frequency
    """
    
    def __init__(self, strategy_config: Dict = None):
        """
        Initialize data integration.
        
        Args:
            strategy_config: Configuration for statistical arbitrage strategy
        """
        self.strategy = create_stat_arb_strategy(strategy_config)
        self.data_collector = DataCollector()
        self.running = False
        self.data_queue = Queue()
        self.update_thread = None
        
        # Configuration
        self.update_interval = config.get('data_collection.market_data_interval', 30)  # seconds
        self.symbols = config.get('data_collection.symbols', ['ETH/USDT', 'SOL/USDT', 'BTC/USDT'])
        self.exchanges = config.get('data_collection.exchanges', ['binance', 'kraken'])
        
        # Data validation
        self.min_data_points = 100  # Minimum data points for strategy initialization
        self.max_price_change = 0.5  # Maximum 50% price change for validation
        
        logger.info("Strategy data integration initialized")
    
    def start(self):
        """Start the data integration and strategy."""
        if self.running:
            logger.warning("Data integration already running")
            return
        
        self.running = True
        
        # Start data collector
        self.data_collector.start()
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        # Initialize strategy with historical data
        self._initialize_strategy_with_historical_data()
        
        logger.info("Strategy data integration started")
    
    def stop(self):
        """Stop the data integration."""
        self.running = False
        
        if self.data_collector:
            self.data_collector.stop()
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)
        
        logger.info("Strategy data integration stopped")
    
    def _initialize_strategy_with_historical_data(self):
        """Load historical data and initialize strategy pairs."""
        logger.info("Loading historical data for strategy initialization...")
        
        try:
            # Load historical data for each symbol
            for symbol in self.symbols:
                for exchange in self.exchanges:
                    try:
                        # Get historical data
                        historical_data = self.data_collector.get_historical_data(
                            symbol=symbol,
                            exchange=exchange,
                            timeframe='1h',
                            limit=1000
                        )
                        
                        if not historical_data.empty:
                            # Update strategy with historical data
                            self._update_strategy_with_dataframe(historical_data, symbol, exchange)
                            logger.info(f"Loaded {len(historical_data)} historical records for {symbol} from {exchange}")
                        
                    except Exception as e:
                        logger.error(f"Error loading historical data for {symbol} from {exchange}: {e}")
            
            # Find cointegrated pairs
            assets = self._extract_assets_from_symbols()
            if assets:
                cointegrated_pairs = self.strategy.find_cointegrated_pairs(assets)
                logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs during initialization")
            
        except Exception as e:
            logger.error(f"Error initializing strategy with historical data: {e}")
    
    def _extract_assets_from_symbols(self) -> List[str]:
        """Extract asset symbols from trading pairs."""
        assets = set()
        for symbol in self.symbols:
            if '/' in symbol:
                base_asset = symbol.split('/')[0]
                assets.add(base_asset)
        return list(assets)
    
    def _update_strategy_with_dataframe(self, df: pd.DataFrame, symbol: str, exchange: str):
        """Update strategy with data from DataFrame."""
        if df.empty:
            return
        
        # Extract asset from symbol (e.g., 'ETH/USDT' -> 'ETH')
        asset = symbol.split('/')[0] if '/' in symbol else symbol
        
        # Update strategy with each data point
        for _, row in df.iterrows():
            timestamp = pd.to_datetime(row['timestamp'])
            price = row['close']  # Use close price
            
            # Validate price
            if self._validate_price(price, asset):
                self.strategy.update_price_data(asset, price, timestamp)
    
    def _validate_price(self, price: float, asset: str) -> bool:
        """Validate price data for reasonableness."""
        if price <= 0:
            return False
        
        # Check for extreme price changes
        if asset in self.strategy.price_data and not self.strategy.price_data[asset].empty:
            last_price = self.strategy.price_data[asset].iloc[-1]
            if last_price > 0:
                price_change = abs(price - last_price) / last_price
                if price_change > self.max_price_change:
                    logger.warning(f"Large price change detected for {asset}: {price_change:.2%}")
                    return False
        
        return True
    
    def _update_loop(self):
        """Main update loop for real-time data."""
        logger.info("Starting real-time data update loop")
        
        while self.running:
            try:
                # Collect current market data
                self._collect_and_update_market_data()
                
                # Generate signals
                signals = self.strategy.generate_signals()
                if signals:
                    logger.info(f"Generated {len(signals)} signals")
                    # Here you could route signals to execution
                    # self.strategy.route_signal_to_execution(signals)
                
                # Update positions (if any)
                if self.strategy.positions:
                    self._update_position_pnls()
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _collect_and_update_market_data(self):
        """Collect current market data and update strategy."""
        for symbol in self.symbols:
            for exchange in self.exchanges:
                try:
                    # Get latest market data
                    market_data = self.data_collector.get_latest_market_data(symbol, exchange)
                    
                    if market_data and market_data.validate():
                        # Extract asset and update strategy
                        asset = symbol.split('/')[0] if '/' in symbol else symbol
                        
                        if self._validate_price(market_data.price, asset):
                            self.strategy.update_price_data(asset, market_data.price, market_data.timestamp)
                            
                            # Log significant price movements
                            if asset in self.strategy.price_data and len(self.strategy.price_data[asset]) > 1:
                                last_price = self.strategy.price_data[asset].iloc[-2]
                                price_change = (market_data.price - last_price) / last_price
                                if abs(price_change) > 0.05:  # 5% change
                                    logger.info(f"Significant price change for {asset}: {price_change:.2%}")
                
                except Exception as e:
                    logger.error(f"Error collecting market data for {symbol} from {exchange}: {e}")
    
    def _update_position_pnls(self):
        """Update PnL for open positions."""
        for pair_key, position in self.strategy.positions.items():
            try:
                # Get current prices
                current_prices = {}
                for asset in [position.asset1, position.asset2]:
                    if asset in self.strategy.price_data and not self.strategy.price_data[asset].empty:
                        current_prices[asset] = self.strategy.price_data[asset].iloc[-1]
                
                if len(current_prices) == 2:
                    # Calculate current PnL
                    pnl1 = (current_prices[position.asset1] - position.entry_price1) * position.size1
                    pnl2 = (current_prices[position.asset2] - position.entry_price2) * position.size2
                    total_pnl = pnl1 + pnl2
                    
                    # Update position PnL
                    position.current_pnl = total_pnl
                    
                    # Log significant PnL changes
                    if abs(total_pnl) > 100:  # Log significant PnL
                        logger.info(f"Position {pair_key} PnL: {total_pnl:.2f}")
            
            except Exception as e:
                logger.error(f"Error updating PnL for position {pair_key}: {e}")
    
    def get_strategy_status(self) -> Dict:
        """Get current strategy status."""
        return {
            'running': self.running,
            'pairs_count': len(self.strategy.pairs_data),
            'positions_count': len(self.strategy.positions),
            'signals_count': len(self.strategy.signals),
            'performance': self.strategy.get_performance_summary(),
            'last_update': self.strategy.last_update
        }
    
    def get_open_positions(self) -> Dict:
        """Get current open positions."""
        return self.strategy.get_open_positions()
    
    def get_recent_signals(self, hours: int = 24) -> List:
        """Get recent trading signals."""
        return self.strategy.get_recent_signals(hours)
    
    def force_signal_generation(self) -> List:
        """Force signal generation (for testing)."""
        return self.strategy.generate_signals()


# Convenience function for creating integration
def create_strategy_integration(strategy_config: Dict = None) -> StrategyDataIntegration:
    """
    Create a new strategy data integration instance.
    
    Args:
        strategy_config: Optional configuration for the strategy
        
    Returns:
        StrategyDataIntegration instance
    """
    return StrategyDataIntegration(strategy_config)


# Example usage and testing
if __name__ == "__main__":
    # Create integration
    integration = create_strategy_integration()
    
    try:
        # Start integration
        integration.start()
        
        # Run for a few minutes
        print("Running strategy integration for 5 minutes...")
        time.sleep(300)  # 5 minutes
        
        # Print status
        status = integration.get_strategy_status()
        print(f"Strategy Status: {status}")
        
    except KeyboardInterrupt:
        print("Stopping integration...")
    finally:
        integration.stop()
        print("Integration stopped.") 