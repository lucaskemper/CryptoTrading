"""
Crypto Trading Bot - Main Application

This is the main entry point for the crypto trading bot system.
It orchestrates all modules including data collection, strategy execution,
risk management, and order management.

Features:
- Real-time data collection from multiple exchanges
- Statistical arbitrage and sentiment analysis
- Risk management and position tracking
- Order execution and portfolio management
- Graceful shutdown and error recovery
"""

import asyncio
import signal
import sys
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import traceback

# Import all modules
from src.data_collector import DataCollector, ExchangeDataCollector, SentimentDataCollector
from src.strategy.stat_arb import StatisticalArbitrage
from src.strategy.sentiment import SentimentAnalyzer
from src.strategy.signal_generator import SignalGenerator
from src.execution.order_manager import OrderManager
from src.execution.risk_manager import RiskManager
from src.execution.position_manager import PositionManager, PositionSide, PositionType
from src.utils.config_loader import config
from src.utils.logger import logger
from src.strategy.enhanced_signal_generator import EnhancedSignalGenerator

# Import monitoring and web server
from src.utils.monitoring import monitor
from src.utils.web_server import start_web_server, stop_web_server


class TradingBot:
    """
    Main trading bot class that orchestrates all components.
    """
    
    def __init__(self):
        """Initialize the trading bot with all components."""
        self.logger = logger
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Initialize all components
        self.data_collector = None
        self.exchange_collector = None
        self.sentiment_collector = None
        self.stat_arb = None
        self.sentiment_analyzer = None
        self.signal_generator = None
        self.enhanced_signal_generator = None  # <-- NEW
        self.order_manager = None
        self.risk_manager = None
        self.position_manager = None
        
        # Load configuration from config.yaml
        self._load_configuration()
        
        # Performance tracking
        self.start_time = None
        self.total_signals = 0
        self.executed_trades = 0
        self.total_pnl = 0.0
        
        self.logger.info("TradingBot initialized")
    
    def _load_configuration(self):
        """Load all configuration settings from config.yaml."""
        # Trading mode configuration
        self.trading_enabled = config.get('TRADING_ENABLED', 'false').lower() == 'true'
        self.simulation_mode = config.get('SIMULATION_MODE', 'true').lower() == 'true'
        
        # Data collection configuration
        data_config = config.get('data_collection', {})
        self.market_data_interval = data_config.get('market_data_interval', 30)
        self.sentiment_data_interval = data_config.get('sentiment_data_interval', 300)
        self.symbols = data_config.get('symbols', ['ETH/USDT', 'SOL/USDT', 'BTC/USDT'])
        self.exchanges = data_config.get('exchanges', ['binance', 'kraken'])
        
        # Risk management configuration
        risk_config = config.get('risk', {})
        self.max_position_size = risk_config.get('max_position_size', 0.1)
        self.max_order_size = risk_config.get('max_order_size', 0.05)
        self.risk_per_trade = risk_config.get('risk_per_trade', 0.02)
        self.stop_loss_percentage = risk_config.get('stop_loss_percentage', 0.05)
        self.take_profit_percentage = risk_config.get('take_profit_percentage', 0.1)
        self.max_total_exposure = risk_config.get('max_total_exposure', 0.8)
        self.max_single_asset_exposure = risk_config.get('max_single_asset_exposure', 0.3)
        self.max_open_positions = risk_config.get('max_open_positions', 10)
        self.max_positions_per_asset = risk_config.get('max_positions_per_asset', 3)
        self.max_daily_drawdown = risk_config.get('max_daily_drawdown', 0.05)
        self.max_total_drawdown = risk_config.get('max_total_drawdown', 0.15)
        self.max_consecutive_losses = risk_config.get('max_consecutive_losses', 5)
        self.volatility_threshold = risk_config.get('volatility_threshold', 0.1)
        
        # Strategy configuration
        strategy_config = config.get('strategy', {})
        
        # Statistical arbitrage configuration
        stat_arb_config = strategy_config.get('statistical_arbitrage', {})
        self.stat_arb_enabled = stat_arb_config.get('enabled', True)
        self.z_score_threshold = stat_arb_config.get('z_score_threshold', 2.0)
        self.cointegration_lookback = stat_arb_config.get('cointegration_lookback', 100)
        self.correlation_threshold = stat_arb_config.get('correlation_threshold', 0.7)
        self.cointegration_pvalue_threshold = stat_arb_config.get('cointegration_pvalue_threshold', 0.05)
        self.min_spread_std = stat_arb_config.get('min_spread_std', 0.001)
        self.position_size_limit = stat_arb_config.get('position_size_limit', 0.1)
        self.stop_loss_pct = stat_arb_config.get('stop_loss_pct', 0.05)
        self.take_profit_pct = stat_arb_config.get('take_profit_pct', 0.1)
        self.max_positions = stat_arb_config.get('max_positions', 5)
        self.rebalance_frequency = stat_arb_config.get('rebalance_frequency', 300)
        self.dynamic_threshold_window = stat_arb_config.get('dynamic_threshold_window', 100)
        self.spread_model = stat_arb_config.get('spread_model', 'ols')
        self.slippage = stat_arb_config.get('slippage', 0.001)
        
        # Sentiment analysis configuration
        sentiment_config = strategy_config.get('sentiment_analysis', {})
        self.sentiment_enabled = sentiment_config.get('enabled', True)
        self.sentiment_model = sentiment_config.get('model', 'gpt-3.5-turbo')
        self.sentiment_confidence_threshold = sentiment_config.get('confidence_threshold', 0.7)
        
        # Signal generator configuration
        signal_config = strategy_config.get('signal_generator', {})
        self.signal_generator_enabled = signal_config.get('enabled', True)
        self.combination_method = signal_config.get('combination_method', 'consensus')
        self.stat_weight = signal_config.get('stat_weight', 0.6)
        self.sentiment_weight = signal_config.get('sentiment_weight', 0.4)
        self.min_confidence = signal_config.get('min_confidence', 0.3)
        self.enable_risk_checks = signal_config.get('enable_risk_checks', True)
        self.require_risk_approval = signal_config.get('require_risk_approval', False)
        self.max_signals_per_batch = signal_config.get('max_signals_per_batch', 10)
        self.signal_timeout_seconds = signal_config.get('signal_timeout_seconds', 300)
        # Add config flag for enhanced ML
        self.use_enhanced_signal_generator = signal_config.get('use_enhanced_ml', False)
        
        # Sentiment thresholds
        sentiment_thresholds = signal_config.get('sentiment_thresholds', {})
        self.positive_sentiment_threshold = sentiment_thresholds.get('positive', 0.2)
        self.negative_sentiment_threshold = sentiment_thresholds.get('negative', -0.2)
        self.neutral_sentiment_threshold = sentiment_thresholds.get('neutral', 0.0)
        
        # Execution configuration
        execution_config = config.get('execution', {})
        order_manager_config = execution_config.get('order_manager', {})
        self.max_retries = order_manager_config.get('max_retries', 3)
        self.retry_delay = order_manager_config.get('retry_delay', 1.0)
        self.max_slippage = order_manager_config.get('max_slippage', 0.02)
        self.order_timeout = order_manager_config.get('order_timeout', 300)
        self.batch_size_limit = order_manager_config.get('batch_size_limit', 10)
        self.enable_partial_fills = order_manager_config.get('enable_partial_fills', True)
        self.auto_cancel_expired = order_manager_config.get('auto_cancel_expired', True)
        
        # Exchange settings
        self.exchange_settings = order_manager_config.get('exchange_settings', {})
        
        # Database configuration
        db_config = config.get('database', {})
        self.sqlite_path = db_config.get('sqlite_path', 'data/trading_bot.db')
        self.csv_dir = db_config.get('csv_dir', 'data/')
        self.postgres_url = db_config.get('postgres_url', '')
        
        # Logging configuration
        logging_config = config.get('logging', {})
        self.log_level = logging_config.get('level', 'INFO')
        self.log_rotation = logging_config.get('file_rotation', 'daily')
        
        # Sentiment sources configuration
        sentiment_sources = config.get('sentiment', {})
        self.reddit_subreddits = sentiment_sources.get('reddit_subreddits', ['CryptoCurrency', 'ethtrader', 'solana'])
        self.news_keywords = sentiment_sources.get('news_keywords', ['bitcoin', 'ethereum', 'solana', 'crypto', 'blockchain'])
        
        # Portfolio configuration
        self.initial_balance = config.get('INITIAL_BALANCE', 10000.0)
        
        self.logger.info("Configuration loaded successfully")
    
    async def initialize(self):
        """Initialize all trading bot components."""
        try:
            self.logger.info("Initializing trading bot components...")
            
            # Initialize data collection components
            self.logger.info("Initializing data collectors...")
            self.exchange_collector = ExchangeDataCollector()
            self.sentiment_collector = SentimentDataCollector()
            self.data_collector = DataCollector()
            
            # Initialize strategy components with configuration
            self.logger.info("Initializing strategy components...")
            
            # Initialize statistical arbitrage with config
            if self.stat_arb_enabled:
                stat_arb_config = {
                    'z_score_threshold': self.z_score_threshold,
                    'lookback_period': self.cointegration_lookback,
                    'correlation_threshold': self.correlation_threshold,
                    'cointegration_pvalue_threshold': self.cointegration_pvalue_threshold,
                    'min_spread_std': self.min_spread_std,
                    'position_size_limit': self.position_size_limit,
                    'stop_loss_pct': self.stop_loss_pct,
                    'take_profit_pct': self.take_profit_pct,
                    'max_positions': self.max_positions,
                    'rebalance_frequency': self.rebalance_frequency,
                    'dynamic_threshold_window': self.dynamic_threshold_window,
                    'spread_model': self.spread_model,
                    'slippage': self.slippage
                }
                self.stat_arb = StatisticalArbitrage(config=stat_arb_config)
            else:
                self.stat_arb = None
            
            # Initialize sentiment analyzer with config
            if self.sentiment_enabled:
                self.sentiment_analyzer = SentimentAnalyzer(
                    model=self.sentiment_model
                )
            else:
                self.sentiment_analyzer = None
            
            # Initialize signal generator
            if self.use_enhanced_signal_generator:
                self.logger.info("Using Enhanced ML Signal Generator.")
                self.enhanced_signal_generator = EnhancedSignalGenerator(initial_capital=self.initial_balance)
                self.signal_generator = self.enhanced_signal_generator
            elif self.signal_generator_enabled:
                self.signal_generator = SignalGenerator(
                    stat_arb=self.stat_arb,
                    sentiment_analyzer=self.sentiment_analyzer,
                    config={
                        'combination_method': self.combination_method,
                        'stat_weight': self.stat_weight,
                        'sentiment_weight': self.sentiment_weight,
                        'min_confidence': self.min_confidence,
                        'enable_risk_checks': self.enable_risk_checks,
                        'require_risk_approval': self.require_risk_approval,
                        'max_signals_per_batch': self.max_signals_per_batch,
                        'signal_timeout_seconds': self.signal_timeout_seconds,
                        'sentiment_thresholds': {
                            'positive': self.positive_sentiment_threshold,
                            'negative': self.negative_sentiment_threshold,
                            'neutral': self.neutral_sentiment_threshold
                        }
                    }
                )
            else:
                self.signal_generator = None
            
            # Initialize execution components with configuration
            self.logger.info("Initializing execution components...")
            
            # Initialize risk manager with config
            self.risk_manager = RiskManager()
            
            # Initialize position manager
            self.position_manager = PositionManager()
            
            # Initialize order manager
            self.order_manager = OrderManager()
            
            # Connect components
            self.logger.info("Connecting components...")
            self._connect_components()
            
            # Initialize portfolio
            self.logger.info("Initializing portfolio...")
            self.risk_manager.set_initial_portfolio_value(self.initial_balance)
            
            self.logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            traceback.print_exc()
            return False
    
    def _connect_components(self):
        """Connect all components together."""
        # Connect position manager
        self.position_manager.set_order_manager(self.order_manager)
        self.position_manager.set_risk_manager(self.risk_manager)
        
        # Connect risk manager
        self.risk_manager.set_position_manager(self.position_manager)
        self.risk_manager.set_order_manager(self.order_manager)
        
        # Connect order manager
        self.order_manager.set_risk_manager(self.risk_manager)
        self.order_manager.set_position_manager(self.position_manager)
        
        self.logger.info("✅ All components connected")
    
    async def start(self):
        """Start the trading bot."""
        try:
            self.logger.info("🚀 Starting crypto trading bot...")
            
            # Initialize components
            if not await self.initialize():
                self.logger.error("❌ Failed to initialize trading bot")
                return False
            
            # Start web server for monitoring
            self.logger.info("🌐 Starting web server for monitoring...")
            await start_web_server(host='0.0.0.0', port=8080, trading_bot=self)
            
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.running = True
            self.start_time = datetime.now()
            
            # Start data collection
            self.logger.info("📡 Starting data collection...")
            await self._start_data_collection()
            
            # Main trading loop
            self.logger.info("🔄 Starting main trading loop...")
            await self._main_trading_loop()
            
        except Exception as e:
            self.logger.error(f"❌ Trading bot failed: {e}")
            traceback.print_exc()
            return False
    
    async def _start_data_collection(self):
        """Start data collection processes."""
        try:
            # Start market data collection
            self.logger.info("📊 Starting market data collection...")
            
            # Collect initial market data using configured symbols and exchanges
            for exchange in self.exchanges:
                for symbol in self.symbols:
                    try:
                        ohlcv_data = self.exchange_collector.get_ohlcv(symbol, exchange, '1h', limit=100)
                        if not ohlcv_data.empty:
                            # Update price data in statistical arbitrage
                            for _, row in ohlcv_data.iterrows():
                                asset = symbol.split('/')[0]
                                if self.stat_arb:
                                    self.stat_arb.update_price_data(asset, row['close'], row['timestamp'])
                            
                            self.logger.info(f"✅ Updated {symbol} data from {exchange}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to get {symbol} data from {exchange}: {e}")
            
            # Start sentiment data collection
            self.logger.info("🧠 Starting sentiment data collection...")
            
            if self.sentiment_collector:
                try:
                    # Collect Reddit sentiment using configured subreddits
                    sentiment_data = self.sentiment_collector.get_reddit_sentiment(
                        subreddits=self.reddit_subreddits,
                        limit=20
                    )
                    if sentiment_data:
                        self.logger.info(f"✅ Collected {len(sentiment_data)} sentiment items")
                except Exception as e:
                    self.logger.warning(f"⚠️ Sentiment collection failed: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ Data collection failed: {e}")
    
    async def _main_trading_loop(self):
        """Main trading loop."""
        self.logger.info("🔄 Entering main trading loop...")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Update market data
                await self._update_market_data()
                
                # Update sentiment data
                await self._update_sentiment_data()
                
                # Generate signals
                signals = await self._generate_signals()
                
                # Process signals
                if signals:
                    await self._process_signals(signals)
                
                # Update portfolio and risk metrics
                await self._update_portfolio_metrics()
                
                # Check for stop conditions
                if not self._check_trading_conditions():
                    self.logger.warning("⚠️ Trading conditions not met, pausing...")
                
                # Log performance metrics
                self._log_performance_metrics()
                
                # Wait for next update using configured interval
                await asyncio.sleep(self.market_data_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Error in main trading loop: {e}")
                traceback.print_exc()
                await asyncio.sleep(10)  # Wait before retrying
    
    async def _update_market_data(self):
        """Update market data from exchanges."""
        try:
            for exchange in self.exchanges:
                for symbol in self.symbols:
                    try:
                        # Get latest ticker
                        ticker = self.exchange_collector.get_ticker(symbol, exchange)
                        if ticker and self.stat_arb:
                            asset = symbol.split('/')[0]
                            self.stat_arb.update_price_data(asset, ticker.price, ticker.timestamp)
                    except Exception as e:
                        self.logger.debug(f"Failed to update {symbol} from {exchange}: {e}")
                        
        except Exception as e:
            self.logger.error(f"❌ Market data update failed: {e}")
    
    async def _update_sentiment_data(self):
        """Update sentiment data."""
        try:
            if not self.sentiment_collector:
                return
                
            # Collect new sentiment data
            sentiment_data = []
            
            # Reddit sentiment using configured subreddits
            try:
                reddit_data = self.sentiment_collector.get_reddit_sentiment(
                    subreddits=self.reddit_subreddits,
                    limit=10
                )
                sentiment_data.extend(reddit_data)
            except Exception as e:
                self.logger.debug(f"Reddit sentiment update failed: {e}")
            
            # News sentiment using configured keywords
            try:
                news_data = self.sentiment_collector.get_news_sentiment(
                    keywords=self.news_keywords
                )
                sentiment_data.extend(news_data)
            except Exception as e:
                self.logger.debug(f"News sentiment update failed: {e}")
            
            if sentiment_data:
                self.logger.info(f"📊 Updated {len(sentiment_data)} sentiment items")
                
        except Exception as e:
            self.logger.error(f"❌ Sentiment data update failed: {e}")
    
    async def _generate_signals(self):
        """Generate trading signals."""
        try:
            if not self.signal_generator:
                return []
            # Get current market prices
            current_prices = {}
            for symbol in self.symbols:
                try:
                    ticker = self.exchange_collector.get_ticker(symbol, 'binance')
                    if ticker:
                        asset = symbol.split('/')[0]
                        current_prices[asset] = ticker.price
                except Exception as e:
                    self.logger.debug(f"Failed to get price for {symbol}: {e}")
            # Generate signals
            if self.use_enhanced_signal_generator and self.enhanced_signal_generator:
                # Use enhanced generator (returns EnhancedSignal dataclasses)
                signals = self.enhanced_signal_generator.generate_signals(
                    market_data=current_prices,
                    sentiment_data=[]  # TODO: populate with real sentiment data
                )
                # Convert EnhancedSignal dataclasses to dicts for downstream compatibility
                signals = [s.__dict__ for s in signals]
            else:
                signals = self.signal_generator.generate_signals(
                    market_data=current_prices,
                    sentiment_data=[]  # TODO: populate with real sentiment data
                )
            if signals:
                self.logger.info(f"🎯 Generated {len(signals)} signals")
                self.total_signals += len(signals)
            return signals
        except Exception as e:
            self.logger.error(f"❌ Signal generation failed: {e}")
            return []
    
    async def _process_signals(self, signals):
        """Process generated signals."""
        try:
            for signal in signals:
                # Check risk management
                is_allowed, reason = self.risk_manager.check_order_risk({
                    'symbol': signal.symbol,
                    'side': signal.side,
                    'quantity': signal.quantity,
                    'price': signal.price
                })
                
                if not is_allowed:
                    self.logger.warning(f"⚠️ Signal rejected by risk manager: {reason}")
                    continue
                
                # Execute order (or simulate in simulation mode)
                if self.simulation_mode:
                    self.logger.info(f"🎮 SIMULATION: Would execute {signal.symbol} {signal.side} {signal.quantity}")
                    self.executed_trades += 1
                else:
                    # Real order execution
                    try:
                        order = await self.order_manager.submit_order(signal)
                        self.logger.info(f"📋 Order submitted: {order.id}")
                        self.executed_trades += 1
                    except Exception as e:
                        self.logger.error(f"❌ Order execution failed: {e}")
                
        except Exception as e:
            self.logger.error(f"❌ Signal processing failed: {e}")
    
    async def _update_portfolio_metrics(self):
        """Update portfolio and risk metrics."""
        try:
            # Update portfolio metrics
            portfolio_metrics = self.position_manager.get_portfolio_metrics()
            
            # Update risk metrics
            self.risk_manager.update_portfolio_metrics(
                portfolio_metrics.total_market_value,
                portfolio_metrics.daily_pnl
            )
            
            # Check for risk events
            stop_losses = self.risk_manager.check_stop_losses()
            take_profits = self.risk_manager.check_take_profits()
            
            if stop_losses or take_profits:
                self.logger.warning(f"⚠️ Risk events detected: {len(stop_losses)} stop losses, {len(take_profits)} take profits")
                
        except Exception as e:
            self.logger.error(f"❌ Portfolio metrics update failed: {e}")
    
    def _check_trading_conditions(self):
        """Check if trading should continue."""
        try:
            # Check if trading is allowed by risk manager
            if not self.risk_manager.is_trading_allowed():
                return False
            
            # Check position limits
            open_positions = self.position_manager.get_open_positions()
            if len(open_positions) >= self.max_open_positions:
                return False
            
            # Check portfolio value
            portfolio_metrics = self.position_manager.get_portfolio_metrics()
            if portfolio_metrics.total_market_value <= 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Trading conditions check failed: {e}")
            return False
    
    def _log_performance_metrics(self):
        """Log current performance metrics."""
        try:
            if self.start_time:
                runtime = datetime.now() - self.start_time
                
                portfolio_metrics = self.position_manager.get_portfolio_metrics()
                
                self.logger.info(f"📊 Performance Update:")
                self.logger.info(f"  Runtime: {runtime}")
                self.logger.info(f"  Total signals: {self.total_signals}")
                self.logger.info(f"  Executed trades: {self.executed_trades}")
                self.logger.info(f"  Portfolio value: ${portfolio_metrics.total_market_value:.2f}")
                self.logger.info(f"  Total PnL: ${portfolio_metrics.total_pnl:.2f}")
                self.logger.info(f"  Open positions: {portfolio_metrics.open_position_count}")
                
        except Exception as e:
            self.logger.error(f"❌ Performance logging failed: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"🛑 Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()
        self.running = False
    
    async def shutdown(self):
        """Gracefully shutdown the trading bot."""
        try:
            self.logger.info("🛑 Shutting down trading bot...")
            
            self.running = False
            self.shutdown_event.set()
            
            # Stop web server
            self.logger.info("🌐 Stopping web server...")
            await stop_web_server()
            
            # Shutdown components
            if self.order_manager:
                await self.order_manager.shutdown()
            
            if self.position_manager:
                await self.position_manager.shutdown()
            
            # Log final performance
            if self.start_time:
                runtime = datetime.now() - self.start_time
                self.logger.info(f"📊 Final Performance:")
                self.logger.info(f"  Total runtime: {runtime}")
                self.logger.info(f"  Total signals: {self.total_signals}")
                self.logger.info(f"  Executed trades: {self.executed_trades}")
            
            self.logger.info("✅ Trading bot shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Shutdown failed: {e}")
            traceback.print_exc()


async def main():
    """Main entry point for the trading bot."""
    bot = TradingBot()
    
    try:
        # Start the trading bot
        success = await bot.start()
        
        if not success:
            logger.error("❌ Trading bot failed to start")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return 1
    finally:
        # Ensure graceful shutdown
        await bot.shutdown()


if __name__ == "__main__":
    # Set up logging based on configuration
    log_level = getattr(logging, config.get('logging.level', 'INFO').upper())
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/trading_bot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Run the trading bot
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
