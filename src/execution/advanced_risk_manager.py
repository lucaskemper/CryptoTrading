#!/usr/bin/env python3
"""
Advanced Risk Management System
Dynamic position sizing, portfolio optimization, and real-time risk monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Position:
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    pnl: float = 0.0
    unrealized_pnl: float = 0.0

class AdvancedRiskManager:
    """Advanced risk management with dynamic position sizing and portfolio optimization"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.risk_level = RiskLevel.LOW
        self.max_drawdown = 0.0
        self.daily_pnl = 0.0
        self.volatility_window = 30  # days
        self.price_history = {}
        self.correlation_matrix = None
        
        # Risk parameters
        self.max_position_size = 0.05  # 5% of portfolio
        self.max_portfolio_risk = 0.02  # 2% max portfolio risk
        self.max_daily_loss = 0.03  # 3% max daily loss
        self.max_total_drawdown = 0.10  # 10% max total drawdown
        self.volatility_threshold = 0.2  # 20% volatility threshold
        
    def update_market_data(self, symbol: str, price: float, volume: float = None):
        """Update market data for risk calculations"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            'price': price,
            'volume': volume,
            'timestamp': datetime.now()
        })
        
        # Keep only last 1000 data points
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-1000:]
    
    def calculate_volatility(self, symbol: str, window: int = 30) -> float:
        """Calculate rolling volatility for a symbol"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < window:
            return 0.0
        
        prices = [p['price'] for p in self.price_history[symbol][-window:]]
        returns = np.diff(np.log(prices))
        return np.std(returns) * np.sqrt(252)  # Annualized volatility
    
    def calculate_correlation_matrix(self, symbols: List[str]) -> np.ndarray:
        """Calculate correlation matrix for portfolio assets"""
        if len(symbols) < 2:
            return np.array([[1.0]])
        
        # Get price data for all symbols
        price_data = {}
        min_length = float('inf')
        
        for symbol in symbols:
            if symbol in self.price_history:
                prices = [p['price'] for p in self.price_history[symbol]]
                price_data[symbol] = prices
                min_length = min(min_length, len(prices))
        
        if min_length < 30:
            return np.eye(len(symbols))
        
        # Align price data
        aligned_prices = {}
        for symbol in symbols:
            if symbol in price_data:
                aligned_prices[symbol] = price_data[symbol][-min_length:]
        
        # Calculate returns and correlation
        returns_data = {}
        for symbol, prices in aligned_prices.items():
            returns = np.diff(np.log(prices))
            returns_data[symbol] = returns
        
        # Create correlation matrix
        symbols_list = list(returns_data.keys())
        n = len(symbols_list)
        corr_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i+1, n):
                symbol1, symbol2 = symbols_list[i], symbols_list[j]
                correlation = np.corrcoef(returns_data[symbol1], returns_data[symbol2])[0, 1]
                corr_matrix[i, j] = correlation
                corr_matrix[j, i] = correlation
        
        return corr_matrix
    
    def calculate_portfolio_risk(self) -> float:
        """Calculate current portfolio risk using VaR"""
        if not self.positions:
            return 0.0
        
        # Calculate position weights and correlations
        total_value = sum(abs(pos.quantity * pos.current_price) for pos in self.positions.values())
        if total_value == 0:
            return 0.0
        
        weights = []
        symbols = []
        for symbol, pos in self.positions.items():
            weight = abs(pos.quantity * pos.current_price) / total_value
            weights.append(weight)
            symbols.append(symbol)
        
        # Get correlation matrix
        corr_matrix = self.calculate_correlation_matrix(symbols)
        
        # Calculate individual volatilities
        volatilities = []
        for symbol in symbols:
            vol = self.calculate_volatility(symbol)
            volatilities.append(vol)
        
        # Calculate portfolio volatility
        portfolio_vol = 0.0
        for i in range(len(weights)):
            for j in range(len(weights)):
                portfolio_vol += weights[i] * weights[j] * volatilities[i] * volatilities[j] * corr_matrix[i, j]
        
        portfolio_vol = np.sqrt(portfolio_vol)
        
        # Calculate VaR (95% confidence)
        var_95 = portfolio_vol * 1.645 * np.sqrt(1/252)  # Daily VaR
        
        return var_95
    
    def calculate_optimal_position_size(self, symbol: str, signal_strength: float) -> float:
        """Calculate optimal position size based on risk and signal strength"""
        # Base position size
        base_size = self.max_position_size * signal_strength
        
        # Adjust for volatility
        volatility = self.calculate_volatility(symbol)
        vol_adjustment = max(0.1, 1.0 - volatility / self.volatility_threshold)
        
        # Adjust for portfolio concentration
        portfolio_concentration = len(self.positions) / 10  # Normalize by max positions
        concentration_adjustment = max(0.5, 1.0 - portfolio_concentration)
        
        # Adjust for current risk level
        risk_adjustment = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.8,
            RiskLevel.HIGH: 0.5,
            RiskLevel.CRITICAL: 0.2
        }[self.risk_level]
        
        optimal_size = base_size * vol_adjustment * concentration_adjustment * risk_adjustment
        
        # Ensure minimum and maximum limits
        min_size = 0.001  # 0.1% minimum
        max_size = self.max_position_size * 2  # 10% maximum
        
        return np.clip(optimal_size, min_size, max_size)
    
    def update_risk_level(self):
        """Update risk level based on current market conditions"""
        # Calculate current drawdown
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        current_drawdown = -total_pnl / self.initial_capital
        
        # Calculate daily loss
        if self.daily_pnl < 0:
            daily_loss_pct = abs(self.daily_pnl) / self.initial_capital
        else:
            daily_loss_pct = 0.0
        
        # Calculate portfolio risk
        portfolio_risk = self.calculate_portfolio_risk()
        
        # Determine risk level
        if (current_drawdown > self.max_total_drawdown or 
            daily_loss_pct > self.max_daily_loss or 
            portfolio_risk > self.max_portfolio_risk):
            self.risk_level = RiskLevel.CRITICAL
        elif (current_drawdown > self.max_total_drawdown * 0.7 or 
              daily_loss_pct > self.max_daily_loss * 0.7 or 
              portfolio_risk > self.max_portfolio_risk * 0.7):
            self.risk_level = RiskLevel.HIGH
        elif (current_drawdown > self.max_total_drawdown * 0.3 or 
              daily_loss_pct > self.max_daily_loss * 0.3 or 
              portfolio_risk > self.max_portfolio_risk * 0.3):
            self.risk_level = RiskLevel.MEDIUM
        else:
            self.risk_level = RiskLevel.LOW
        
        logger.info(f"Risk level updated to: {self.risk_level.value}")
    
    def check_signal_risk(self, signal: Dict) -> Tuple[bool, str, float]:
        """Check if signal meets risk requirements"""
        symbol = signal.get('symbol', '')
        signal_strength = signal.get('confidence', 0.5)
        
        # Update risk level
        self.update_risk_level()
        
        # Calculate optimal position size
        optimal_size = self.calculate_optimal_position_size(symbol, signal_strength)
        
        # Check portfolio risk limits
        portfolio_risk = self.calculate_portfolio_risk()
        if portfolio_risk > self.max_portfolio_risk:
            return False, f"Portfolio risk {portfolio_risk:.3f} exceeds limit {self.max_portfolio_risk}", 0.0
        
        # Check daily loss limits
        if self.daily_pnl < -self.max_daily_loss * self.initial_capital:
            return False, f"Daily loss limit exceeded", 0.0
        
        # Check drawdown limits
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        current_drawdown = -total_pnl / self.initial_capital
        if current_drawdown > self.max_total_drawdown:
            return False, f"Maximum drawdown exceeded: {current_drawdown:.3f}", 0.0
        
        return True, "Signal approved", optimal_size
    
    def add_position(self, symbol: str, side: str, quantity: float, price: float):
        """Add a new position"""
        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            current_price=price,
            entry_time=datetime.now()
        )
        
        self.positions[symbol] = position
        logger.info(f"Added position: {symbol} {side} {quantity} @ {price}")
    
    def update_position(self, symbol: str, current_price: float):
        """Update position with current price"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = current_price
            
            # Calculate PnL
            if position.side == 'long':
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:  # short
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
    
    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close a position and return realized PnL"""
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        
        # Calculate realized PnL
        if position.side == 'long':
            realized_pnl = (exit_price - position.entry_price) * position.quantity
        else:  # short
            realized_pnl = (position.entry_price - exit_price) * position.quantity
        
        # Update daily PnL
        self.daily_pnl += realized_pnl
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Closed position {symbol}: PnL = {realized_pnl:.2f}")
        return realized_pnl
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        portfolio_risk = self.calculate_portfolio_risk()
        
        return {
            'risk_level': self.risk_level.value,
            'current_capital': self.current_capital + total_pnl,
            'total_pnl': total_pnl,
            'daily_pnl': self.daily_pnl,
            'portfolio_risk': portfolio_risk,
            'max_drawdown': self.max_drawdown,
            'open_positions': len(self.positions),
            'position_details': [
                {
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl
                }
                for pos in self.positions.values()
            ]
        }
    
    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of new day)"""
        self.daily_pnl = 0.0
        logger.info("Daily metrics reset")

# Example usage
if __name__ == "__main__":
    # Initialize risk manager
    risk_manager = AdvancedRiskManager(initial_capital=100000.0)
    
    # Simulate market data
    risk_manager.update_market_data("BTC/USDT", 50000.0)
    risk_manager.update_market_data("ETH/USDT", 3000.0)
    
    # Test signal risk check
    signal = {
        'symbol': 'BTC/USDT',
        'confidence': 0.8,
        'side': 'long'
    }
    
    approved, reason, size = risk_manager.check_signal_risk(signal)
    print(f"Signal approved: {approved}")
    print(f"Reason: {reason}")
    print(f"Position size: {size:.3f}")
    
    # Generate risk report
    report = risk_manager.get_risk_report()
    print(f"Risk level: {report['risk_level']}")
    print(f"Portfolio risk: {report['portfolio_risk']:.3f}") 