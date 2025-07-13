#!/usr/bin/env python3
"""
Risk Manager Demo

This script demonstrates the comprehensive risk management system
for the crypto trading bot, showing all core responsibilities:

1. Pre-Trade Risk Checks
2. Real-Time & Post-Trade Monitoring  
3. Order and Position Lifecycle Management
4. Logging and Audit

Usage:
    python examples/risk_manager_demo.py
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from execution.risk_manager import RiskManager, RiskLevel, RiskEventType
from execution.order_manager import TradeSignal, OrderSide, OrderType


class RiskManagerDemo:
    """Demo class for showcasing risk manager functionality."""
    
    def __init__(self):
        self.risk_manager = RiskManager()
        self.portfolio_value = 10000.0
        self.daily_pnl = 0.0
        
    def demo_initialization(self):
        """Demonstrate risk manager initialization."""
        print("=" * 60)
        print("RISK MANAGER INITIALIZATION")
        print("=" * 60)
        
        print(f"Max Position Size: {self.risk_manager.max_position_size * 100}%")
        print(f"Max Order Size: {self.risk_manager.max_order_size * 100}%")
        print(f"Risk Per Trade: {self.risk_manager.risk_per_trade * 100}%")
        print(f"Max Daily Drawdown: {self.risk_manager.max_daily_drawdown * 100}%")
        print(f"Max Total Drawdown: {self.risk_manager.max_total_drawdown * 100}%")
        print(f"Max Open Positions: {self.risk_manager.max_open_positions}")
        print(f"Max Consecutive Losses: {self.risk_manager.max_consecutive_losses}")
        print()
    
    def demo_pre_trade_checks(self):
        """Demonstrate pre-trade risk checks."""
        print("=" * 60)
        print("PRE-TRADE RISK CHECKS")
        print("=" * 60)
        
        # Set initial portfolio value
        self.risk_manager.set_initial_portfolio_value(self.portfolio_value)
        
        # Test 1: Valid order
        print("Test 1: Valid Order")
        signal1 = TradeSignal(
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            quantity=0.5,  # Small quantity
            order_type=OrderType.MARKET,
            strategy_name="stat_arb"
        )
        
        is_allowed, reason = self.risk_manager.check_order_risk(signal1)
        print(f"  Order allowed: {is_allowed}")
        print(f"  Reason: {reason}")
        print()
        
        # Test 2: Order too large
        print("Test 2: Order Too Large")
        signal2 = TradeSignal(
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            quantity=10.0,  # Large quantity
            order_type=OrderType.MARKET,
            strategy_name="stat_arb"
        )
        
        is_allowed, reason = self.risk_manager.check_order_risk(signal2)
        print(f"  Order allowed: {is_allowed}")
        print(f"  Reason: {reason}")
        print()
        
        # Test 3: Too many positions
        print("Test 3: Too Many Positions")
        # Add maximum positions
        for i in range(10):
            self.risk_manager.update_position(f"ASSET{i}/USDT", "long", 1.0, 100.0)
        
        signal3 = TradeSignal(
            symbol="NEW/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.MARKET,
            strategy_name="stat_arb"
        )
        
        is_allowed, reason = self.risk_manager.check_order_risk(signal3)
        print(f"  Order allowed: {is_allowed}")
        print(f"  Reason: {reason}")
        print()
    
    def demo_position_management(self):
        """Demonstrate position lifecycle management."""
        print("=" * 60)
        print("POSITION LIFECYCLE MANAGEMENT")
        print("=" * 60)
        
        # Clear positions for demo
        self.risk_manager.positions.clear()
        
        # Create positions
        print("Creating positions...")
        self.risk_manager.update_position("ETH/USDT", "long", 2.0, 2000.0)
        self.risk_manager.update_position("SOL/USDT", "long", 10.0, 100.0)
        self.risk_manager.update_position("BTC/USDT", "short", 0.1, 50000.0)
        
        print(f"Total positions: {len(self.risk_manager.positions)}")
        for pos_id, position in self.risk_manager.positions.items():
            print(f"  {pos_id}: {position.quantity} @ ${position.entry_price:.2f}")
        print()
        
        # Update position with new trade
        print("Updating ETH position...")
        self.risk_manager.update_position("ETH/USDT", "long", 1.0, 2100.0)
        
        eth_position = self.risk_manager.positions["ETH/USDT_long"]
        print(f"  ETH Position: {eth_position.quantity} @ ${eth_position.entry_price:.2f}")
        print()
        
        # Close position
        print("Closing SOL position...")
        pnl = self.risk_manager.close_position("SOL/USDT", "long", 10.0, 110.0)
        print(f"  Realized PnL: ${pnl:.2f}")
        print(f"  Remaining positions: {len(self.risk_manager.positions)}")
        print()
    
    def demo_stop_loss_take_profit(self):
        """Demonstrate stop-loss and take-profit functionality."""
        print("=" * 60)
        print("STOP-LOSS & TAKE-PROFIT MANAGEMENT")
        print("=" * 60)
        
        # Clear positions
        self.risk_manager.positions.clear()
        
        # Create position with stop-loss and take-profit
        position = self.risk_manager.positions.get("ETH/USDT_long")
        if position:
            position.stop_loss = 1900.0
            position.take_profit = 2200.0
        
        # Mock current prices
        original_get_price = self.risk_manager._get_current_price
        self.risk_manager._get_current_price = lambda symbol: {
            "ETH/USDT": 1850.0,  # Below stop-loss
            "SOL/USDT": 120.0,   # Above take-profit
            "BTC/USDT": 50000.0  # Normal
        }.get(symbol, 100.0)
        
        try:
            # Check stop-losses
            print("Checking stop-losses...")
            positions_to_close = self.risk_manager.check_stop_losses()
            for pos_id, side, quantity in positions_to_close:
                print(f"  Stop-loss triggered: {pos_id} - {side} {quantity}")
            
            # Check take-profits
            print("Checking take-profits...")
            positions_to_close = self.risk_manager.check_take_profits()
            for pos_id, side, quantity in positions_to_close:
                print(f"  Take-profit triggered: {pos_id} - {side} {quantity}")
            print()
        finally:
            # Restore original method
            self.risk_manager._get_current_price = original_get_price
    
    def demo_portfolio_monitoring(self):
        """Demonstrate portfolio monitoring and drawdown controls."""
        print("=" * 60)
        print("PORTFOLIO MONITORING & DRAWDOWN CONTROLS")
        print("=" * 60)
        
        # Update portfolio metrics
        print("Updating portfolio metrics...")
        self.risk_manager.update_portfolio_metrics(self.portfolio_value, self.daily_pnl)
        
        metrics = self.risk_manager.portfolio_metrics
        if metrics:
            print(f"Portfolio Value: ${metrics.total_value:,.2f}")
            print(f"Total PnL: ${metrics.total_pnl:,.2f} ({metrics.total_pnl_percentage:.2f}%)")
            print(f"Daily PnL: ${metrics.daily_pnl:,.2f} ({metrics.daily_pnl_percentage:.2f}%)")
            print(f"Max Drawdown: {metrics.max_drawdown_percentage:.2f}%")
            print(f"VaR (95%): ${metrics.var_95:,.2f}")
            print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
            print()
        
        # Simulate drawdown
        print("Simulating drawdown...")
        self.risk_manager.update_portfolio_metrics(8500.0, -1500.0)
        
        metrics = self.risk_manager.portfolio_metrics
        if metrics:
            print(f"Portfolio Value: ${metrics.total_value:,.2f}")
            print(f"Max Drawdown: {metrics.max_drawdown_percentage:.2f}%")
            print(f"Risk Level: {self.risk_manager.get_risk_level().value}")
            print()
    
    def demo_circuit_breakers(self):
        """Demonstrate circuit breaker functionality."""
        print("=" * 60)
        print("CIRCUIT BREAKERS")
        print("=" * 60)
        
        # Test consecutive losses circuit breaker
        print("Testing consecutive losses circuit breaker...")
        self.risk_manager.consecutive_losses = 5  # Max allowed
        
        signal = TradeSignal(
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.MARKET,
            strategy_name="stat_arb"
        )
        
        is_allowed, reason = self.risk_manager.check_order_risk(signal)
        print(f"  Order allowed: {is_allowed}")
        print(f"  Reason: {reason}")
        print(f"  Trading paused: {self.risk_manager.trading_paused}")
        print(f"  Pause reason: {self.risk_manager.pause_reason}")
        print()
        
        # Resume trading
        print("Resuming trading...")
        self.risk_manager.resume_trading()
        print(f"  Trading paused: {self.risk_manager.trading_paused}")
        print()
    
    def demo_risk_reporting(self):
        """Demonstrate risk reporting and audit functionality."""
        print("=" * 60)
        print("RISK REPORTING & AUDIT")
        print("=" * 60)
        
        # Generate risk report
        report = self.risk_manager.get_risk_report()
        
        print("Trading Status:")
        status = report["trading_status"]
        print(f"  Paused: {status['paused']}")
        print(f"  Consecutive Losses: {status['consecutive_losses']}")
        print()
        
        print("Risk Limits:")
        limits = report["risk_limits"]
        for key, value in limits.items():
            if isinstance(value, float):
                print(f"  {key}: {value*100:.1f}%")
            else:
                print(f"  {key}: {value}")
        print()
        
        print("Recent Risk Events:")
        events = report["recent_risk_events"]
        for event in events[-5:]:  # Last 5 events
            print(f"  {event['timestamp']}: {event['event_type']} - {event['message']}")
        print()
    
    def demo_risk_levels(self):
        """Demonstrate risk level determination."""
        print("=" * 60)
        print("RISK LEVEL DETERMINATION")
        print("=" * 60)
        
        # Test different scenarios
        scenarios = [
            (10500.0, 500.0, "Low Risk"),
            (9700.0, -300.0, "Medium Risk"),
            (9400.0, -600.0, "High Risk"),
            (8900.0, -1100.0, "Critical Risk")
        ]
        
        for portfolio_value, daily_pnl, description in scenarios:
            self.risk_manager.update_portfolio_metrics(portfolio_value, daily_pnl)
            risk_level = self.risk_manager.get_risk_level()
            print(f"{description}: {risk_level.value.upper()}")
        print()
    
    def run_demo(self):
        """Run the complete risk manager demo."""
        print("CRYPTO TRADING BOT - RISK MANAGER DEMO")
        print("=" * 60)
        print()
        
        try:
            self.demo_initialization()
            self.demo_pre_trade_checks()
            self.demo_position_management()
            self.demo_stop_loss_take_profit()
            self.demo_portfolio_monitoring()
            self.demo_circuit_breakers()
            self.demo_risk_reporting()
            self.demo_risk_levels()
            
            print("=" * 60)
            print("DEMO COMPLETED SUCCESSFULLY")
            print("=" * 60)
            
        except Exception as e:
            print(f"Demo error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the demo."""
    demo = RiskManagerDemo()
    demo.run_demo()


if __name__ == "__main__":
    main() 