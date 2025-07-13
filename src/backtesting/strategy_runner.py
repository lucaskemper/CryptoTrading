"""
Strategy Runner for Backtesting

Executes trading strategies during backtesting simulation.
Handles strategy initialization, signal generation, and trade execution.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

from src.strategy.stat_arb import StatisticalArbitrage
from src.strategy.enhanced_stat_arb import EnhancedStatisticalArbitrage, create_enhanced_stat_arb_strategy
from src.strategy.sentiment import SentimentAnalyzer
from src.strategy.signal_generator import SignalGenerator
from src.strategy.enhanced_signal_generator import EnhancedSignalGenerator
from src.execution.risk_manager import RiskManager
from src.execution.position_manager import PositionManager
from src.utils.logger import logger


class StrategyRunner:
    """
    Executes trading strategies during backtesting.
    
    Features:
    - Multi-strategy execution (statistical arbitrage, sentiment)
    - Signal generation and validation
    - Risk management integration
    - Position tracking and management
    """
    
    def __init__(self, config):
        """Initialize the strategy runner."""
        self.config = config
        self.logger = logger
        
        # Strategy components
        self.stat_arb: Optional[StatisticalArbitrage] = None
        self.enhanced_stat_arb: Optional[EnhancedStatisticalArbitrage] = None
        self.sentiment_analyzer: Optional[SentimentAnalyzer] = None
        self.signal_generator: Optional[SignalGenerator] = None
        self.enhanced_signal_generator: Optional[EnhancedSignalGenerator] = None  # NEW
        self.risk_manager: Optional[RiskManager] = None
        self.position_manager: Optional[PositionManager] = None
        
        # State tracking
        self.current_positions: Dict[str, Dict] = {}
        self.signal_history: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # Configuration
        self.initial_capital = config.initial_capital
        self.max_position_size = config.max_position_size
        self.stop_loss_pct = config.stop_loss_pct
        self.take_profit_pct = config.take_profit_pct
        self.slippage = config.slippage
        self.commission = config.commission
        
        self.logger.info("StrategyRunner initialized")
    
    def initialize_strategies(self):
        """Initialize all strategy components."""
        self.logger.info("Initializing strategy components...")
        
        # Initialize enhanced statistical arbitrage (preferred) or regular statistical arbitrage
        stat_arb_config = self.config.strategy_config.get('statistical_arbitrage', {})
        if self.config.strategy_config.get('use_enhanced_stat_arb', True):
            self.logger.info("Using Enhanced Statistical Arbitrage Strategy")
            self.enhanced_stat_arb = create_enhanced_stat_arb_strategy(stat_arb_config)
            self.stat_arb = self.enhanced_stat_arb  # For compatibility
        else:
            self.logger.info("Using Standard Statistical Arbitrage Strategy")
            self.stat_arb = StatisticalArbitrage(stat_arb_config)
        
        # Initialize sentiment analyzer
        if self.config.sentiment_enabled:
            sentiment_config = self.config.strategy_config.get('sentiment_analysis', {})
            self.sentiment_analyzer = SentimentAnalyzer(sentiment_config)
        
        # Initialize signal generator
        signal_config = self.config.strategy_config.get('signal_generator', {})
        if signal_config.get('use_enhanced_ml', False):
            self.logger.info("Using Enhanced ML Signal Generator for backtesting.")
            self.enhanced_signal_generator = EnhancedSignalGenerator(initial_capital=self.initial_capital)
            self.signal_generator = self.enhanced_signal_generator
        else:
            self.signal_generator = SignalGenerator(signal_config)
        
        # Initialize risk manager
        risk_config = self.config.strategy_config.get('risk_management', {})
        self.risk_manager = RiskManager(risk_config)
        
        # Initialize position manager
        position_config = self.config.strategy_config.get('position_management', {})
        if not position_config:
            position_config = {
                'enable_position_tracking': True,
                'enable_pnl_calculation': True,
                'enable_risk_checks': True
            }
        self.position_manager = PositionManager(position_config)
        
        self.logger.info("Strategy components initialized successfully")
    
    def update_market_data(self, market_data: Dict[str, Dict], timestamp: datetime):
        """Update strategies with new market data."""
        # Update statistical arbitrage
        if self.stat_arb:
            for symbol, data in market_data.items():
                if 'close' in data:
                    self.stat_arb.update_price_data(symbol, data['close'], timestamp)
        
        # Update position manager
        if self.position_manager:
            self.position_manager.update_market_data(market_data, timestamp)
    
    def update_sentiment_data(self, sentiment_data: Dict, timestamp: datetime):
        """Update sentiment analyzer with new sentiment data."""
        if self.sentiment_analyzer and sentiment_data:
            self.sentiment_analyzer.update_sentiment_data(sentiment_data)
    
    def generate_signals(self, market_data: Dict[str, Dict], 
                        sentiment_data: Optional[Dict] = None) -> List[Dict]:
        """Generate trading signals from all strategies."""
        signals = []
        
        # Generate statistical arbitrage signals
        if self.stat_arb:
            stat_signals = self.stat_arb.generate_signals()
            signals.extend(stat_signals)
        
        # Generate sentiment signals
        if self.sentiment_analyzer and sentiment_data:
            sentiment_signals = self.sentiment_analyzer.generate_signals(sentiment_data)
            signals.extend(sentiment_signals)
        
        # Use enhanced ML signal generator if enabled
        if self.enhanced_signal_generator:
            enhanced_signals = self.enhanced_signal_generator.generate_signals(market_data, sentiment_data)
            # Convert EnhancedSignal dataclasses to dicts
            enhanced_signals = [s.__dict__ for s in enhanced_signals]
            return enhanced_signals
        
        # Combine signals using signal generator
        if signals and self.signal_generator:
            combined_signals = self.signal_generator.combine_signals(signals)
            return combined_signals
        
        return signals
    
    def validate_signals(self, signals: List[Dict]) -> List[Dict]:
        """Validate signals using risk management rules."""
        if not self.risk_manager:
            return signals
        
        validated_signals = []
        
        for signal in signals:
            if self.risk_manager.validate_signal(signal, self.current_positions):
                validated_signals.append(signal)
            else:
                self.logger.debug(f"Signal rejected by risk manager: {signal}")
        
        return validated_signals
    
    def execute_trades(self, signals: List[Dict], market_data: Dict[str, Dict], 
                      timestamp: datetime) -> List[Dict]:
        """Execute trades based on validated signals."""
        executed_trades = []
        
        for signal in signals:
            try:
                # Calculate position size
                position_size = self._calculate_position_size(signal, market_data)
                
                if position_size > 0:
                    # Calculate execution price with slippage
                    execution_price = self._calculate_execution_price(signal, market_data)
                    
                    # Create trade record
                    trade = {
                        'timestamp': timestamp,
                        'symbol': signal.get('symbol'),
                        'side': signal.get('side'),
                        'quantity': position_size,
                        'price': execution_price,
                        'commission': position_size * execution_price * self.commission,
                        'slippage': position_size * execution_price * self.slippage,
                        'strategy': signal.get('strategy', 'unknown'),
                        'signal_confidence': signal.get('confidence', 0.0)
                    }
                    
                    # Calculate trade P&L
                    trade['pnl'] = self._calculate_trade_pnl(trade)
                    
                    executed_trades.append(trade)
                    
                    # Update positions
                    self._update_positions(trade)
                    
                    # Update statistical arbitrage strategy positions
                    if self.stat_arb:
                        # Convert trade to Signal format for the strategy
                        strategy_signal = self._convert_trade_to_signal(trade, signal)
                        self.stat_arb.update_positions([strategy_signal])
                    
                    # Record trade
                    self.trade_history.append(trade)
                    
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
    
    def _calculate_position_size(self, signal: Dict, market_data: Dict[str, Dict]) -> float:
        """Calculate position size based on risk management rules."""
        symbol = signal.get('symbol')
        if symbol not in market_data:
            return 0.0
        
        current_price = market_data[symbol].get('close', 0)
        if current_price <= 0:
            return 0.0
        
        # Calculate position size based on risk per trade - align with risk manager limits
        # Use the smaller of max_position_size or risk_per_trade to ensure consistency
        max_position_pct = min(self.max_position_size, 0.10)  # Cap at 10% to match risk manager
        position_value = self.initial_capital * max_position_pct
        
        # Adjust for signal confidence
        confidence = signal.get('confidence', 0.5)
        position_value *= confidence
        
        return position_value / current_price
    
    def _calculate_execution_price(self, signal: Dict, market_data: Dict[str, Dict]) -> float:
        """Calculate execution price with slippage simulation."""
        symbol = signal.get('symbol')
        side = signal.get('side', 'buy')
        
        if symbol not in market_data:
            return 0.0
        
        base_price = market_data[symbol].get('close', 0)
        
        # Apply slippage based on order side
        if side == 'buy':
            execution_price = base_price * (1 + self.slippage)
        else:
            execution_price = base_price * (1 - self.slippage)
        
        return execution_price
    
    def _calculate_trade_pnl(self, trade: Dict) -> float:
        """Calculate realized P&L for a single trade."""
        # Use realized P&L if available (from position management)
        if 'realized_pnl' in trade:
            return trade['realized_pnl']
        # Fallback: no P&L for individual trades (should be calculated in position management)
        return 0.0

    def _update_positions(self, trade: Dict):
        """Update current positions and realize P&L on sells."""
        symbol = trade['symbol']
        side = trade['side']
        quantity = trade['quantity']
        price = trade['price']
        realized_pnl = 0.0

        if symbol not in self.current_positions:
            self.current_positions[symbol] = {
                'quantity': 0, 
                'avg_price': 0,
                'total_cost': 0
            }

        position = self.current_positions[symbol]

        if side == 'buy':
            # Add to position
            total_quantity = position['quantity'] + quantity
            total_cost = position['total_cost'] + (quantity * price)
            
            position['quantity'] = total_quantity
            position['total_cost'] = total_cost
            position['avg_price'] = total_cost / total_quantity if total_quantity > 0 else 0
        else:
            # Sell: realize P&L on closed/reduced position
            sell_quantity = min(quantity, position['quantity'])
            realized_pnl = (price - position['avg_price']) * sell_quantity
            position['quantity'] -= sell_quantity
            if position['quantity'] <= 0:
                del self.current_positions[symbol]

        # Store realized P&L in trade
        trade['realized_pnl'] = realized_pnl
    
    def check_stop_loss_take_profit(self, market_data: Dict[str, Dict], timestamp: datetime) -> List[Dict]:
        """Check for stop-loss and take-profit triggers."""
        exit_signals = []
        
        for symbol, position in self.current_positions.items():
            if symbol not in market_data:
                continue
            
            current_price = market_data[symbol].get('close', 0)
            if current_price <= 0:
                continue
            
            avg_price = position['avg_price']
            quantity = position['quantity']
            
            if quantity <= 0:
                continue
            
            # Calculate P&L
            unrealized_pnl = (current_price - avg_price) * quantity
            
            # Check stop-loss
            if unrealized_pnl < -(avg_price * quantity * self.stop_loss_pct):
                exit_signals.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'quantity': quantity,
                    'reason': 'stop_loss',
                    'strategy': 'risk_management',
                    'confidence': 1.0
                })
            
            # Check take-profit
            elif unrealized_pnl > (avg_price * quantity * self.take_profit_pct):
                exit_signals.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'quantity': quantity,
                    'reason': 'take_profit',
                    'strategy': 'risk_management',
                    'confidence': 1.0
                })
        
        return exit_signals
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate current performance metrics."""
        if not self.trade_history:
            return {}
        
        # Calculate basic metrics
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in self.trade_history if t.get('pnl', 0) < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = sum([t.get('pnl', 0) for t in self.trade_history])
        
        # Calculate average win/loss
        winning_pnls = [t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) > 0]
        losing_pnls = [t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) < 0]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        
        # Calculate profit factor
        total_profit = sum(winning_pnls)
        total_loss = abs(sum(losing_pnls))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'current_positions': len(self.current_positions)
        }
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get current position summary."""
        position_summary = {
            'total_positions': len(self.current_positions),
            'positions': {}
        }
        
        for symbol, position in self.current_positions.items():
            position_summary['positions'][symbol] = {
                'quantity': position['quantity'],
                'avg_price': position['avg_price'],
                'total_cost': position['total_cost']
            }
        
        return position_summary
    
    def reset(self):
        """Reset strategy runner state."""
        self.current_positions = {}
        self.signal_history = []
        self.trade_history = []
        self.performance_metrics = {}
        
        # Reset strategy components
        if self.stat_arb:
            self.stat_arb = StatisticalArbitrage(self.config.strategy_config.get('statistical_arbitrage', {}))
        
        if self.sentiment_analyzer:
            self.sentiment_analyzer = SentimentAnalyzer(self.config.strategy_config.get('sentiment_analysis', {}))
        
        if self.signal_generator:
            self.signal_generator = SignalGenerator(self.config.strategy_config.get('signal_generator', {}))
        
        if self.enhanced_signal_generator: # NEW
            self.enhanced_signal_generator = EnhancedSignalGenerator(self.config.strategy_config.get('signal_generator', {})) # NEW
        
        if self.risk_manager:
            self.risk_manager = RiskManager(self.config.strategy_config.get('risk_management', {}))
        
        if self.position_manager:
            self.position_manager = PositionManager(self.config.strategy_config.get('position_management', {}))
        
        self.logger.info("Strategy runner reset")


def create_strategy_runner(config) -> StrategyRunner:
    """Factory function to create a strategy runner."""
    return StrategyRunner(config) 