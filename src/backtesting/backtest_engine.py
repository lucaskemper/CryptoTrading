"""
Backtesting Engine

Main engine for running comprehensive backtests of crypto trading strategies.
Handles data simulation, strategy execution, and performance analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.strategy.stat_arb import StatisticalArbitrage
from src.strategy.sentiment import SentimentAnalyzer
from src.strategy.signal_generator import SignalGenerator
from src.execution.risk_manager import RiskManager
from src.execution.position_manager import PositionManager
from src.utils.logger import logger
from src.utils.config_loader import config

from .data_simulator import DataSimulator
from .performance_analyzer import PerformanceAnalyzer
from .strategy_runner import StrategyRunner


@dataclass
class BacktestConfig:
    """Configuration for backtesting runs."""
    
    # Time period
    start_date: datetime
    end_date: datetime
    
    # Assets and data
    symbols: List[str] = field(default_factory=lambda: ['ETH/USDT', 'SOL/USDT', 'BTC/USDT'])
    exchanges: List[str] = field(default_factory=lambda: ['binance', 'kraken'])
    
    # Strategy configuration
    strategy_config: Dict[str, Any] = field(default_factory=dict)
    
    # Risk management
    initial_capital: float = 100000.0
    max_position_size: float = 0.1
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.1
    
    # Execution settings
    slippage: float = 0.001  # 0.1% slippage
    commission: float = 0.001  # 0.1% commission
    min_trade_size: float = 10.0
    
    # Data settings
    data_frequency: str = '1h'  # 1m, 5m, 15m, 1h, 4h, 1d
    sentiment_enabled: bool = True
    sentiment_frequency: str = '4h'
    
    # Performance tracking
    track_metrics: bool = True
    generate_plots: bool = True
    save_results: bool = True
    results_dir: str = 'data/backtest_results'


@dataclass
class BacktestResult:
    """Results from a backtesting run."""
    
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Portfolio metrics
    final_portfolio_value: float
    peak_portfolio_value: float
    volatility: float
    
    # Strategy metrics
    strategy_signals: int
    executed_trades: int
    rejected_trades: int
    
    # Timestamps
    start_time: datetime
    end_time: datetime
    duration: timedelta
    
    # Data
    equity_curve: pd.Series
    trade_history: List[Dict]
    position_history: List[Dict]
    signal_history: List[Dict]
    
    # Configuration
    config: BacktestConfig


class BacktestEngine:
    """
    Main backtesting engine for crypto trading strategies.
    
    Features:
    - Historical data simulation with realistic market conditions
    - Multi-strategy backtesting (statistical arbitrage, sentiment)
    - Comprehensive risk management simulation
    - Detailed performance analysis and reporting
    - Visualization and result export
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize the backtesting engine.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.logger = logger
        
        # Initialize components
        self.data_simulator = DataSimulator(config)
        self.performance_analyzer = PerformanceAnalyzer()
        self.strategy_runner = StrategyRunner(config)
        
        # Results storage
        self.results: Optional[BacktestResult] = None
        self.equity_curve: List[float] = []
        self.trade_history: List[Dict] = []
        self.position_history: List[Dict] = []
        self.signal_history: List[Dict] = []
        
        # State tracking
        self.current_portfolio_value: float = config.initial_capital
        self.peak_portfolio_value: float = config.initial_capital
        self.current_positions: Dict[str, Dict] = {}
        
        self.logger.info(f"BacktestEngine initialized for period {config.start_date} to {config.end_date}")
    
    def run_backtest(self) -> BacktestResult:
        """
        Run a complete backtest.
        
        Returns:
            BacktestResult with comprehensive performance metrics
        """
        self.logger.info("Starting backtest run...")
        
        try:
            # Initialize data simulation
            self.logger.info("Initializing data simulation...")
            self.data_simulator.initialize()
            
            # Initialize strategy components
            self.logger.info("Initializing strategy components...")
            self._initialize_strategies()
            
            # Initialize cointegrated pairs for statistical arbitrage
            self._initialize_cointegrated_pairs()
            
            # Run the backtest
            self.logger.info("Running backtest simulation...")
            self._run_simulation()
            
            # Analyze results
            self.logger.info("Analyzing backtest results...")
            self.results = self._analyze_results()
            
            # Generate reports and visualizations
            if self.config.generate_plots:
                self._generate_plots()
            
            # Save results
            if self.config.save_results:
                self._save_results()
            
            self.logger.info("Backtest completed successfully")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise
    
    def _initialize_strategies(self):
        """Initialize all strategy components."""
        # Initialize statistical arbitrage
        stat_arb_config = self.config.strategy_config.get('statistical_arbitrage', {})
        self.stat_arb = StatisticalArbitrage(stat_arb_config)
        
        # Initialize sentiment analyzer
        if self.config.sentiment_enabled:
            sentiment_config = self.config.strategy_config.get('sentiment_analysis', {})
            self.sentiment_analyzer = SentimentAnalyzer(sentiment_config)
        else:
            self.sentiment_analyzer = None
        
        # Initialize signal generator
        signal_config = self.config.strategy_config.get('signal_generator', {})
        self.signal_generator = SignalGenerator(signal_config)
        
        # Initialize risk manager
        risk_config = self.config.strategy_config.get('risk_management', {})
        # Add initial portfolio value to risk config
        risk_config['initial_portfolio_value'] = self.config.initial_capital
        self.risk_manager = RiskManager(risk_config)
        
        # Initialize position manager
        self.position_manager = PositionManager()

        # To use ML-enhanced signal generator, set 'use_enhanced_ml': True in signal_generator config
        # Example:
        # self.config.strategy_config = {
        #     'signal_generator': {'use_enhanced_ml': True},
        #     ...
        # }
    
    def _initialize_cointegrated_pairs(self):
        """Initialize cointegrated pairs after data is loaded."""
        self.logger.info("Initializing cointegrated pairs for statistical arbitrage...")
        
        # Load initial price data from data simulator
        self.logger.info("Loading initial price data for pair analysis...")
        initial_data = self.data_simulator.get_initial_data()
        
        # Update statistical arbitrage with initial data
        for symbol, data in initial_data.items():
            if 'close' in data and 'timestamp' in data:
                # Iterate through the data points to load them individually
                for timestamp, price in zip(data['timestamp'], data['close']):
                    self.stat_arb.update_price_data(symbol, price, timestamp)
        
        # Use the full symbol names that match the data keys
        assets = self.config.symbols  # Use full symbols like 'ETH/USDT', 'BTC/USDT', etc.
        cointegrated_pairs = self.stat_arb.find_cointegrated_pairs(assets)
        self.logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs: {cointegrated_pairs}")
    
    def _run_simulation(self):
        """Run the main simulation loop."""
        start_time = datetime.now()
        
        # Get data iterator
        data_iterator = self.data_simulator.get_data_iterator()
        
        for timestamp, market_data, sentiment_data in data_iterator:
            try:
                # Update strategies with new data
                self._update_strategies(timestamp, market_data, sentiment_data)
                
                # Generate signals
                signals = self._generate_signals(market_data, sentiment_data)
                
                # Apply risk management
                validated_signals = self._apply_risk_management(signals)
                
                # Execute trades
                executed_trades = self._execute_trades(validated_signals, market_data)
                
                # Update portfolio
                self._update_portfolio(executed_trades, market_data)
                
                # Record metrics
                self._record_metrics(timestamp, signals, executed_trades)
                
            except Exception as e:
                self.logger.error(f"Error in simulation step at {timestamp}: {e}")
                continue
        
        end_time = datetime.now()
        self.logger.info(f"Simulation completed in {end_time - start_time}")
    
    def _update_strategies(self, timestamp: datetime, market_data: Dict, sentiment_data: Optional[Dict]):
        """Update all strategies with new market data."""
        # Update statistical arbitrage
        for symbol, data in market_data.items():
            if 'close' in data:
                self.stat_arb.update_price_data(symbol, data['close'], timestamp)
        
        # Update sentiment analyzer
        if self.sentiment_analyzer and sentiment_data:
            self.sentiment_analyzer.update_sentiment_data(sentiment_data)
    
    def _generate_signals(self, market_data: Dict, sentiment_data: Optional[Dict]) -> List[Dict]:
        """Generate trading signals from all strategies."""
        signals = []
        
        # Get statistical arbitrage signals
        stat_signals = self.stat_arb.generate_signals()
        # Convert StatSignal objects to dictionary format
        for signal in stat_signals:
            # Convert StatSignal to dictionary format expected by backtest engine
            signal_dict = {
                'timestamp': signal.timestamp,
                'symbol': signal.asset1,  # Use asset1 as primary symbol
                'side': signal.action1,   # Use action1 as primary side
                'quantity': signal.size1,
                'price': None,  # Market order
                'confidence': signal.confidence,
                'strategy': 'statistical_arbitrage',
                'signal_type': signal.signal_type,
                'z_score': signal.z_score,
                'spread_value': signal.spread_value,
                'pair': signal.pair,
                'asset1': signal.asset1,
                'asset2': signal.asset2,
                'action1': signal.action1,
                'action2': signal.action2,
                'size1': signal.size1,
                'size2': signal.size2
            }
            signals.append(signal_dict)
        
        # Get sentiment signals (if enabled)
        if self.sentiment_analyzer and sentiment_data:
            sentiment_signals = self.sentiment_analyzer.generate_signals(sentiment_data)
            signals.extend(sentiment_signals)
        
        return signals
    
    def _apply_risk_management(self, signals: List[Dict]) -> List[Dict]:
        """Apply risk management rules to signals."""
        # Ensure risk manager uses the latest portfolio value
        self.risk_manager.update_portfolio_value(self.current_portfolio_value)
        validated_signals = []
        
        for signal in signals:
            # Check risk limits
            if self.risk_manager.validate_signal(signal, self.current_positions):
                validated_signals.append(signal)
            else:
                self.logger.debug(f"Signal rejected by risk manager: {signal}")
        
        return validated_signals
    
    def _execute_trades(self, signals: List[Dict], market_data: Dict) -> List[Dict]:
        """Execute trades based on signals."""
        executed_trades = []
        
        for signal in signals:
            try:
                # Calculate position size
                position_size = self._calculate_position_size(signal, market_data)
                
                if position_size > 0:
                    # Simulate trade execution with slippage and commission
                    execution_price = self._calculate_execution_price(signal, market_data)
                    
                    trade = {
                        'timestamp': signal.get('timestamp'),
                        'symbol': signal.get('symbol'),
                        'side': signal.get('side'),
                        'quantity': position_size,
                        'price': execution_price,
                        'commission': position_size * execution_price * self.config.commission,
                        'slippage': position_size * execution_price * self.config.slippage,
                        'strategy': signal.get('strategy', 'unknown')
                    }
                    
                    executed_trades.append(trade)
                    
                    # Update positions
                    self._update_positions(trade)
                    
                    # Update statistical arbitrage strategy positions
                    if hasattr(self, 'stat_arb') and self.stat_arb:
                        # Convert trade to Signal format for the strategy
                        strategy_signal = self._convert_trade_to_signal(trade, signal)
                        self.stat_arb.update_positions([strategy_signal])
                    
            except Exception as e:
                self.logger.error(f"Error executing trade: {e}")
        
        return executed_trades
    
    def _convert_trade_to_signal(self, trade: Dict, original_signal: Dict) -> 'Signal':
        """
        Convert a trade dictionary to a Signal object for the statistical arbitrage strategy.
        
        Args:
            trade: Trade dictionary with execution details
            original_signal: Original signal dictionary
            
        Returns:
            Signal object compatible with StatisticalArbitrage.update_positions()
        """
        from src.strategy.stat_arb import Signal
        
        # Parse asset1 and asset2 from the original signal's pair field
        pair = original_signal.get('pair')
        if pair and '_' in pair:
            asset1, asset2 = pair.split('_', 1)
        else:
            # Fallback to symbol and USDT
            asset1 = trade['symbol']
            asset2 = 'USDT'
        
        # Determine signal type based on side
        side = trade['side']
        if side == 'buy':
            signal_type = 'entry_long'
            action1, action2 = 'buy', 'sell'
        else:
            signal_type = 'exit'
            action1, action2 = 'sell', 'buy'
        
        signal = Signal(
            timestamp=trade['timestamp'],
            pair=pair or f"{asset1}_{asset2}",
            signal_type=signal_type,
            asset1=asset1,
            asset2=asset2,
            action1=action1,
            action2=action2,
            size1=trade['quantity'],
            size2=trade['quantity'],
            z_score=original_signal.get('z_score', 0.0),
            spread_value=original_signal.get('spread_value', 0.0),
            confidence=original_signal.get('confidence', 0.5)
        )
        return signal
    
    def _calculate_position_size(self, signal: Dict, market_data: Dict) -> float:
        """Calculate position size based on risk management rules."""
        symbol = signal.get('symbol')
        if symbol not in market_data:
            return 0.0
        
        current_price = market_data[symbol].get('close', 0)
        if current_price <= 0:
            return 0.0
        
        # Calculate position size based on risk per trade - align with risk manager limits
        # Use the smaller of max_position_size or risk_per_trade to ensure consistency
        max_position_pct = min(self.config.max_position_size, 0.10)  # Cap at 10% to match risk manager
        position_value = self.current_portfolio_value * max_position_pct
        
        return position_value / current_price
    
    def _calculate_execution_price(self, signal: Dict, market_data: Dict) -> float:
        """Calculate execution price with slippage simulation."""
        symbol = signal.get('symbol')
        side = signal.get('side', 'buy')
        
        if symbol not in market_data:
            return 0.0
        
        base_price = market_data[symbol].get('close', 0)
        
        # Apply slippage based on order side
        if side == 'buy':
            execution_price = base_price * (1 + self.config.slippage)
        else:
            execution_price = base_price * (1 - self.config.slippage)
        
        return execution_price
    
    def _update_positions(self, trade: Dict):
        """Update current positions and realize P&L on sells."""
        symbol = trade['symbol']
        side = trade['side']
        quantity = trade['quantity']
        price = trade['price']
        realized_pnl = 0.0

        if symbol not in self.current_positions:
            self.current_positions[symbol] = {'quantity': 0, 'avg_price': 0}

        position = self.current_positions[symbol]

        if side == 'buy':
            # Add to position
            total_quantity = position['quantity'] + quantity
            total_value = (position['quantity'] * position['avg_price'] + 
                          quantity * price)
            position['quantity'] = total_quantity
            position['avg_price'] = total_value / total_quantity if total_quantity > 0 else 0
        else:
            # Sell: realize P&L on closed/reduced position
            sell_quantity = min(quantity, position['quantity'])
            realized_pnl = (price - position['avg_price']) * sell_quantity
            position['quantity'] -= sell_quantity
            if position['quantity'] <= 0:
                del self.current_positions[symbol]

        # Store realized P&L in trade
        trade['realized_pnl'] = realized_pnl
        # Update portfolio value only on realized P&L
        if realized_pnl != 0.0:
            self.current_portfolio_value += realized_pnl

    def _update_portfolio(self, executed_trades: List[Dict], market_data: Dict):
        """Update portfolio value based on realized P&L only."""
        # No need to recalculate P&L here; it's handled in _update_positions
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.current_portfolio_value)
        # Update risk manager with current portfolio value
        if hasattr(self.risk_manager, 'update_portfolio_value'):
            self.risk_manager.update_portfolio_value(self.current_portfolio_value)
        # Record equity curve
        self.equity_curve.append(self.current_portfolio_value)
    
    def _record_metrics(self, timestamp: datetime, signals: List[Dict], executed_trades: List[Dict]):
        """Record metrics for analysis."""
        # Record signals
        for signal in signals:
            self.signal_history.append({
                'timestamp': timestamp,
                'signal': signal
            })
        
        # Record trades
        for trade in executed_trades:
            self.trade_history.append({
                'timestamp': timestamp,
                'trade': trade
            })
        
        # Record positions
        self.position_history.append({
            'timestamp': timestamp,
            'positions': self.current_positions.copy(),
            'portfolio_value': self.current_portfolio_value
        })
    
    def _analyze_results(self) -> BacktestResult:
        """Analyze backtest results and calculate performance metrics."""
        # Calculate basic metrics
        total_return = (self.current_portfolio_value - self.config.initial_capital) / self.config.initial_capital
        
        # Calculate annualized return
        duration_days = (self.config.end_date - self.config.start_date).days
        annualized_return = ((1 + total_return) ** (365 / duration_days)) - 1 if duration_days > 0 else 0
        
        # Calculate Sharpe ratio
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Calculate max drawdown
        equity_series = pd.Series(self.equity_curve)
        # --- FIX: Ensure DatetimeIndex ---
        if not isinstance(equity_series.index, pd.DatetimeIndex):
            # Generate a date range matching the backtest period and equity curve length
            equity_series.index = pd.date_range(
                start=self.config.start_date,
                periods=len(equity_series),
                freq=self.config.data_frequency
            )
        # --- END FIX ---
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate trade statistics
        winning_trades = len([t for t in self.trade_history if t['trade'].get('pnl', 0) > 0])
        losing_trades = len([t for t in self.trade_history if t['trade'].get('pnl', 0) < 0])
        total_trades = len(self.trade_history)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        total_profit = sum([t['trade'].get('pnl', 0) for t in self.trade_history if t['trade'].get('pnl', 0) > 0])
        total_loss = abs(sum([t['trade'].get('pnl', 0) for t in self.trade_history if t['trade'].get('pnl', 0) < 0]))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=total_profit / winning_trades if winning_trades > 0 else 0,
            avg_loss=total_loss / losing_trades if losing_trades > 0 else 0,
            largest_win=max([t['trade'].get('pnl', 0) for t in self.trade_history], default=0),
            largest_loss=min([t['trade'].get('pnl', 0) for t in self.trade_history], default=0),
            final_portfolio_value=self.current_portfolio_value,
            peak_portfolio_value=self.peak_portfolio_value,
            volatility=returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
            strategy_signals=len(self.signal_history),
            executed_trades=len(self.trade_history),
            rejected_trades=len(self.signal_history) - len(self.trade_history),
            start_time=self.config.start_date,
            end_time=self.config.end_date,
            duration=self.config.end_date - self.config.start_date,
            equity_curve=equity_series,
            trade_history=self.trade_history,
            position_history=self.position_history,
            signal_history=self.signal_history,
            config=self.config
        )
    
    def _generate_plots(self):
        """Generate performance visualization plots."""
        if not self.results:
            return
        
        # Create results directory
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Equity Curve
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(self.results.equity_curve)
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        
        # Plot 2: Drawdown
        plt.subplot(2, 2, 2)
        equity_series = self.results.equity_curve
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        if len(drawdown) > 0:
            plt.fill_between(range(len(drawdown)), drawdown.values, 0, alpha=0.3, color='red')
            plt.title('Portfolio Drawdown')
            plt.xlabel('Time')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
        
        # Plot 3: Trade Distribution
        plt.subplot(2, 2, 3)
        trade_pnls = [t['trade'].get('pnl', 0) for t in self.trade_history]
        plt.hist(trade_pnls, bins=20, alpha=0.7)
        plt.title('Trade P&L Distribution')
        plt.xlabel('P&L ($)')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Plot 4: Monthly Returns
        plt.subplot(2, 2, 4)
        returns = self.results.equity_curve.pct_change().dropna()
        if len(returns) > 0:
            monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
            if len(monthly_returns) > 0:
                plt.bar(range(len(monthly_returns)), monthly_returns)
                plt.title('Monthly Returns')
                plt.xlabel('Month')
                plt.ylabel('Return (%)')
                plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'backtest_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self):
        """Save backtest results to files."""
        if not self.results:
            return
        
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results summary
        summary = {
            'total_return': self.results.total_return,
            'annualized_return': self.results.annualized_return,
            'sharpe_ratio': self.results.sharpe_ratio,
            'max_drawdown': self.results.max_drawdown,
            'win_rate': self.results.win_rate,
            'profit_factor': self.results.profit_factor,
            'total_trades': self.results.total_trades,
            'winning_trades': self.results.winning_trades,
            'losing_trades': self.results.losing_trades,
            'final_portfolio_value': self.results.final_portfolio_value,
            'peak_portfolio_value': self.results.peak_portfolio_value,
            'volatility': self.results.volatility
        }
        
        with open(results_dir / 'backtest_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save trade history
        trade_df = pd.DataFrame(self.trade_history)
        trade_df.to_csv(results_dir / 'trade_history.csv', index=False)
        
        # Save equity curve
        equity_df = pd.DataFrame({
            'timestamp': pd.date_range(self.config.start_date, self.config.end_date, periods=len(self.equity_curve)),
            'portfolio_value': self.equity_curve
        })
        equity_df.to_csv(results_dir / 'equity_curve.csv', index=False)
        
        self.logger.info(f"Results saved to {results_dir}")


def create_backtest_engine(config: BacktestConfig) -> BacktestEngine:
    """Factory function to create a backtest engine."""
    return BacktestEngine(config) 