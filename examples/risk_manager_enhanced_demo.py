#!/usr/bin/env python3
"""
Enhanced Risk Manager Demo

This demo showcases the enhanced risk management system with:
- Portfolio integration with real-time values
- Sector classification and correlation analysis
- Database logging for audit and compliance
- Advanced risk scenarios and edge cases
- Comprehensive reporting and monitoring

Usage:
    PYTHONPATH=src python examples/risk_manager_enhanced_demo.py
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from execution.risk_manager import (
    RiskManager, RiskLevel, RiskEventType, Position, 
    PortfolioMetrics, TradeSignal, OrderSide, OrderType
)
from utils.logger import logger


class MockPortfolioManager:
    """Mock portfolio manager for demo."""
    
    def __init__(self, initial_value: float = 10000.0):
        self.total_value = initial_value
        self.cash = initial_value
        self.positions = {}
    
    def get_total_value(self) -> float:
        """Get total portfolio value including cash and positions."""
        total = self.cash
        for symbol, position in self.positions.items():
            total += position['quantity'] * position['current_price']
        return total
    
    def update_position(self, symbol: str, quantity: float, price: float):
        """Update position in portfolio."""
        if symbol in self.positions:
            # Update existing position
            pos = self.positions[symbol]
            total_quantity = pos['quantity'] + quantity
            total_value = (pos['quantity'] * pos['current_price']) + (quantity * price)
            pos['current_price'] = total_value / total_quantity
            pos['quantity'] = total_quantity
        else:
            # New position
            self.positions[symbol] = {
                'quantity': quantity,
                'current_price': price
            }
        
        # Update cash
        self.cash -= quantity * price


class MockMarketDataManager:
    """Mock market data manager for demo."""
    
    def __init__(self):
        self.prices = {
            "ETH/USDT": 2000.0,
            "BTC/USDT": 50000.0,
            "SOL/USDT": 100.0,
            "MATIC/USDT": 0.8,
            "UNI/USDT": 5.0,
            "LINK/USDT": 15.0,
            "USDT/USDT": 1.0
        }
        self.price_history = {}
    
    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for symbol."""
        base_price = self.prices.get(symbol, 100.0)
        
        # Add some realistic price movement
        import random
        movement = random.uniform(-0.02, 0.02)  # ±2% movement
        return base_price * (1 + movement)
    
    def update_price(self, symbol: str, new_price: float):
        """Update price for symbol."""
        self.prices[symbol] = new_price


def print_separator(title: str):
    """Print a formatted separator."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_risk_report(risk_manager: RiskManager):
    """Print formatted risk report."""
    report = risk_manager.get_risk_report()
    
    print("\n📊 RISK REPORT")
    print("-" * 40)
    
    # Trading status
    status = report["trading_status"]
    print(f"Trading Paused: {status['paused']}")
    if status['pause_reason']:
        print(f"Pause Reason: {status['pause_reason']}")
    print(f"Consecutive Losses: {status['consecutive_losses']}")
    
    # Portfolio metrics
    if report["portfolio_metrics"]:
        metrics = report["portfolio_metrics"]
        print(f"\nPortfolio Value: ${metrics['total_value']:,.2f}")
        print(f"Total PnL: ${metrics['total_pnl']:,.2f} ({metrics['total_pnl_percentage']:.2f}%)")
        print(f"Daily PnL: ${metrics['daily_pnl']:,.2f} ({metrics['daily_pnl_percentage']:.2f}%)")
        print(f"Max Drawdown: {metrics['max_drawdown_percentage']:.2f}%")
        print(f"VaR (95%): ${metrics['var_95']:,.2f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    
    # Positions
    positions = report["positions"]
    if positions:
        print(f"\nOpen Positions ({len(positions)}):")
        for pos_id, pos in positions.items():
            print(f"  {pos_id}: {pos['quantity']} @ ${pos['entry_price']:.4f} "
                  f"(Current: ${pos['current_price']:.4f}, PnL: ${pos['unrealized_pnl']:.2f})")
    
    # Recent risk events
    events = report["recent_risk_events"]
    if events:
        print(f"\nRecent Risk Events ({len(events)}):")
        for event in events[-3:]:  # Last 3 events
            print(f"  [{event['risk_level'].upper()}] {event['event_type']}: {event['message']}")


def demo_portfolio_integration():
    """Demo portfolio integration features."""
    print_separator("PORTFOLIO INTEGRATION DEMO")
    
    # Initialize managers
    portfolio_manager = MockPortfolioManager(10000.0)
    market_data_manager = MockMarketDataManager()
    
    # Initialize risk manager with dependencies
    risk_manager = RiskManager(
        portfolio_manager=portfolio_manager,
        market_data_manager=market_data_manager,
        db_path="data/demo_risk_events.db"
    )
    
    print("✅ Risk manager initialized with portfolio and market data integration")
    
    # Set initial portfolio value
    risk_manager.set_initial_portfolio_value(10000.0)
    print(f"✅ Initial portfolio value set to: ${10000:,.2f}")
    
    # Test portfolio value retrieval
    portfolio_value = risk_manager._get_portfolio_value()
    print(f"✅ Current portfolio value: ${portfolio_value:,.2f}")
    
    # Test current price retrieval
    eth_price = risk_manager._get_current_price("ETH/USDT")
    print(f"✅ Current ETH price: ${eth_price:,.2f}")
    
    return risk_manager, portfolio_manager, market_data_manager


def demo_sector_analysis():
    """Demo sector classification and analysis."""
    print_separator("SECTOR ANALYSIS DEMO")
    
    risk_manager, portfolio_manager, market_data_manager = demo_portfolio_integration()
    
    # Add positions in different sectors
    positions = [
        ("ETH/USDT", "buy", 2.0, 2000.0),    # Layer1
        ("SOL/USDT", "buy", 10.0, 100.0),     # Layer1
        ("MATIC/USDT", "buy", 100.0, 0.8),    # Layer2
        ("UNI/USDT", "buy", 20.0, 5.0),       # DeFi
        ("LINK/USDT", "buy", 5.0, 15.0),      # Oracle
    ]
    
    print("📈 Adding positions across different sectors:")
    for symbol, side, quantity, price in positions:
        risk_manager.update_position(symbol, side, quantity, price)
        portfolio_manager.update_position(symbol, quantity, price)
        print(f"  {symbol}: {quantity} @ ${price:.4f}")
    
    # Calculate sector exposure
    portfolio_value = risk_manager._get_portfolio_value()
    sector_exposure = risk_manager._calculate_sector_exposure(portfolio_value)
    
    print("\n🏭 Sector Exposure Analysis:")
    for sector, exposure in sector_exposure.items():
        print(f"  {sector}: {exposure*100:.2f}%")
    
    return risk_manager, portfolio_manager, market_data_manager


def demo_correlation_analysis():
    """Demo correlation analysis and limits."""
    print_separator("CORRELATION ANALYSIS DEMO")
    
    risk_manager, portfolio_manager, market_data_manager = demo_sector_analysis()
    
    # Add price history for correlation analysis
    print("📊 Building price history for correlation analysis...")
    
    # Simulate correlated price movements
    for i in range(50):
        # ETH and BTC move together (high correlation)
        eth_price = 2000.0 + i * 2
        btc_price = 50000.0 + i * 100
        
        # SOL moves independently
        sol_price = 100.0 + (i % 10) * 0.5
        
        risk_manager._update_price_history("ETH/USDT", eth_price)
        risk_manager._update_price_history("BTC/USDT", btc_price)
        risk_manager._update_price_history("SOL/USDT", sol_price)
    
    # Calculate correlation matrix
    correlation_matrix = risk_manager._calculate_correlation_matrix()
    
    print("\n🔗 Correlation Matrix:")
    for asset1 in ["ETH/USDT", "BTC/USDT", "SOL/USDT"]:
        if asset1 in correlation_matrix:
            print(f"\n{asset1}:")
            for asset2 in ["ETH/USDT", "BTC/USDT", "SOL/USDT"]:
                if asset2 in correlation_matrix[asset1]:
                    corr = correlation_matrix[asset1][asset2]
                    print(f"  vs {asset2}: {corr:.3f}")
    
    # Test correlation limits
    print("\n🚫 Testing correlation limits...")
    
    # Try to add highly correlated position
    signal = TradeSignal(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.1,  # Large enough to trigger correlation limit
        order_type=OrderType.MARKET,
        strategy_name="correlation_test"
    )
    
    is_allowed, reason = risk_manager.check_order_risk(signal)
    print(f"Order allowed: {is_allowed}")
    print(f"Reason: {reason}")
    
    return risk_manager, portfolio_manager, market_data_manager


def demo_database_logging():
    """Demo database logging and audit features."""
    print_separator("DATABASE LOGGING DEMO")
    
    risk_manager, portfolio_manager, market_data_manager = demo_correlation_analysis()
    
    # Log various risk events
    print("📝 Logging risk events to database...")
    
    risk_events = [
        (RiskEventType.ORDER_BLOCKED, "ETH/USDT", "Order size exceeds limits", RiskLevel.HIGH),
        (RiskEventType.EXPOSURE_LIMIT_BREACH, "BTC/USDT", "Asset exposure limit exceeded", RiskLevel.HIGH),
        (RiskEventType.DRAWDOWN_BREACH, "PORTFOLIO", "Daily drawdown limit exceeded", RiskLevel.CRITICAL),
        (RiskEventType.CIRCUIT_BREAKER_TRIGGERED, "SYSTEM", "High volatility detected", RiskLevel.CRITICAL),
        (RiskEventType.STOP_LOSS_TRIGGERED, "SOL/USDT", "Stop-loss triggered at $95.00", RiskLevel.MEDIUM),
    ]
    
    for event_type, symbol, message, risk_level in risk_events:
        risk_manager._log_risk_event(event_type, symbol, message, risk_level)
        print(f"  Logged: [{risk_level.value.upper()}] {event_type.value}")
    
    # Update portfolio metrics
    print("\n📊 Logging portfolio metrics...")
    risk_manager.update_portfolio_metrics(10500.0, 500.0)
    
    # Retrieve historical events
    print("\n📋 Retrieving historical risk events...")
    historical_events = risk_manager.get_historical_risk_events(days=1)
    print(f"Found {len(historical_events)} historical events")
    
    for event in historical_events[:3]:  # Show first 3
        print(f"  [{event['risk_level'].upper()}] {event['event_type']}: {event['message']}")
    
    return risk_manager, portfolio_manager, market_data_manager


def demo_advanced_scenarios():
    """Demo advanced risk scenarios and edge cases."""
    print_separator("ADVANCED SCENARIOS DEMO")
    
    risk_manager, portfolio_manager, market_data_manager = demo_database_logging()
    
    # Scenario 1: Rapid drawdown
    print("\n📉 Scenario 1: Rapid Drawdown")
    risk_manager.update_portfolio_metrics(9500.0, -500.0)  # -5% daily
    risk_manager.update_portfolio_metrics(9000.0, -500.0)  # -5% daily
    risk_manager.update_portfolio_metrics(8500.0, -500.0)  # -5% daily
    
    signal = TradeSignal(
        symbol="ETH/USDT",
        side=OrderSide.BUY,
        quantity=0.1,
        order_type=OrderType.MARKET,
        strategy_name="drawdown_test"
    )
    
    is_allowed, reason = risk_manager.check_order_risk(signal)
    print(f"Order allowed after drawdown: {is_allowed}")
    print(f"Reason: {reason}")
    
    # Scenario 2: Consecutive losses
    print("\n💸 Scenario 2: Consecutive Losses")
    risk_manager.consecutive_losses = 4
    
    is_allowed, reason = risk_manager.check_order_risk(signal)
    print(f"Order allowed with 4 consecutive losses: {is_allowed}")
    
    risk_manager.consecutive_losses = 5
    is_allowed, reason = risk_manager.check_order_risk(signal)
    print(f"Order allowed with 5 consecutive losses: {is_allowed}")
    print(f"Reason: {reason}")
    
    # Scenario 3: Partial position close
    print("\n✂️ Scenario 3: Partial Position Close")
    
    # Create position
    risk_manager.update_position("ETH/USDT", "buy", 2.0, 2000.0)
    
    # Partial close
    pnl = risk_manager.close_position("ETH/USDT", "buy", 1.0, 2200.0)
    print(f"Partial close PnL: ${pnl:.2f}")
    
    # Check remaining position
    positions = risk_manager.get_positions()
    if "ETH/USDT_buy" in positions:
        pos = positions["ETH/USDT_buy"]
        print(f"Remaining position: {pos.quantity} ETH")
    
    # Scenario 4: Stop-loss and take-profit triggers
    print("\n🎯 Scenario 4: Stop-Loss and Take-Profit")
    
    # Create position with stop-loss and take-profit
    risk_manager.update_position("SOL/USDT", "buy", 10.0, 100.0)
    position = risk_manager.positions["SOL/USDT_buy"]
    position.stop_loss = 95.0
    position.take_profit = 110.0
    
    # Check stop-loss trigger
    with patch.object(risk_manager, '_get_current_price', return_value=94.0):
        stop_loss_positions = risk_manager.check_stop_losses()
        if stop_loss_positions:
            print(f"Stop-loss triggered: {stop_loss_positions[0]}")
    
    # Check take-profit trigger
    with patch.object(risk_manager, '_get_current_price', return_value=115.0):
        take_profit_positions = risk_manager.check_take_profits()
        if take_profit_positions:
            print(f"Take-profit triggered: {take_profit_positions[0]}")
    
    return risk_manager, portfolio_manager, market_data_manager


def demo_stress_test():
    """Demo stress testing with multiple positions and rapid updates."""
    print_separator("STRESS TEST DEMO")
    
    risk_manager, portfolio_manager, market_data_manager = demo_advanced_scenarios()
    
    print("🔥 Running stress test with multiple positions and rapid updates...")
    
    # Add multiple positions rapidly
    positions = [
        ("ETH/USDT", "buy", 1.0, 2000.0),
        ("BTC/USDT", "buy", 0.01, 50000.0),
        ("SOL/USDT", "sell", 5.0, 100.0),
        ("MATIC/USDT", "buy", 100.0, 0.8),
        ("UNI/USDT", "sell", 10.0, 5.0),
    ]
    
    for symbol, side, quantity, price in positions:
        risk_manager.update_position(symbol, side, quantity, price)
        portfolio_manager.update_position(symbol, quantity, price)
    
    print(f"✅ Added {len(positions)} positions")
    
    # Rapid portfolio updates
    print("📈 Simulating rapid portfolio updates...")
    
    for i in range(10):
        portfolio_value = 10000.0 + i * 100
        daily_pnl = 100.0 if i % 2 == 0 else -50.0
        risk_manager.update_portfolio_metrics(portfolio_value, daily_pnl)
        
        # Update some prices
        if i % 3 == 0:
            market_data_manager.update_price("ETH/USDT", 2000.0 + i * 10)
            market_data_manager.update_price("BTC/USDT", 50000.0 + i * 100)
    
    print("✅ Completed rapid updates")
    
    # Generate comprehensive report
    print("\n📊 Generating comprehensive risk report...")
    print_risk_report(risk_manager)
    
    return risk_manager, portfolio_manager, market_data_manager


def demo_risk_levels():
    """Demo different risk levels and their handling."""
    print_separator("RISK LEVELS DEMO")
    
    risk_manager, portfolio_manager, market_data_manager = demo_stress_test()
    
    print("🎚️ Testing different risk levels...")
    
    # Test different risk levels
    risk_levels = [
        (RiskLevel.LOW, "Low risk event"),
        (RiskLevel.MEDIUM, "Medium risk event"),
        (RiskLevel.HIGH, "High risk event"),
        (RiskLevel.CRITICAL, "Critical risk event"),
    ]
    
    for risk_level, message in risk_levels:
        risk_manager._log_risk_event(
            RiskEventType.ORDER_BLOCKED,
            "TEST/USDT",
            message,
            risk_level
        )
        print(f"  Logged {risk_level.value.upper()} risk event")
    
    # Get current risk level
    current_risk_level = risk_manager.get_risk_level()
    print(f"\n🎯 Current portfolio risk level: {current_risk_level.value.upper()}")
    
    return risk_manager, portfolio_manager, market_data_manager


def main():
    """Run the enhanced risk manager demo."""
    print("🚀 Enhanced Risk Manager Demo")
    print("=" * 60)
    print("This demo showcases the enhanced risk management system with:")
    print("• Portfolio integration with real-time values")
    print("• Sector classification and correlation analysis")
    print("• Database logging for audit and compliance")
    print("• Advanced risk scenarios and edge cases")
    print("• Comprehensive reporting and monitoring")
    print("=" * 60)
    
    try:
        # Run all demos
        risk_manager, portfolio_manager, market_data_manager = demo_risk_levels()
        
        print_separator("DEMO COMPLETED")
        print("✅ All enhanced risk manager features demonstrated successfully!")
        print("\n📋 Summary of features tested:")
        print("  • Portfolio integration with real-time value tracking")
        print("  • Sector classification (Layer1, Layer2, DeFi, Oracle, etc.)")
        print("  • Correlation analysis and limits")
        print("  • Database logging for risk events and metrics")
        print("  • Advanced scenarios (drawdown, consecutive losses, partial closes)")
        print("  • Stop-loss and take-profit triggers")
        print("  • Stress testing with multiple positions")
        print("  • Risk level assessment and reporting")
        print("  • Circuit breakers and trading pauses")
        
        print("\n📊 Final Risk Report:")
        print_risk_report(risk_manager)
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Add missing import for patch
    from unittest.mock import patch
    
    main() 