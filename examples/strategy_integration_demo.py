#!/usr/bin/env python3
"""
Strategy Integration Demo

This script demonstrates how the OrderManager integrates with strategy components
to create a complete trading system.

Usage:
    python examples/strategy_integration_demo.py
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.execution.order_manager import (
    OrderManager, 
    TradeSignal, 
    OrderSide, 
    OrderType
)
from src.utils.logger import logger
from src.utils.config_loader import config


class MockStrategy:
    """Mock strategy that generates trade signals."""
    
    def __init__(self, name: str = "mock_strategy"):
        self.name = name
        self.last_signal_time = None
        self.signal_count = 0
    
    def generate_signals(self, market_data: Dict[str, float]) -> List[TradeSignal]:
        """Generate trade signals based on market data."""
        signals = []
        
        # Simple strategy: buy when price drops, sell when it rises
        for symbol, price in market_data.items():
            # Simulate some trading logic
            if self.signal_count % 3 == 0:  # Every 3rd signal
                if price < 50000:  # Buy signal for BTC
                    signal = TradeSignal(
                        symbol=f"{symbol}/USDT",
                        side=OrderSide.BUY,
                        quantity=0.001,
                        order_type=OrderType.LIMIT,
                        price=price * 0.99,  # 1% below current price
                        strategy_name=self.name
                    )
                    signals.append(signal)
                elif price > 55000:  # Sell signal for BTC
                    signal = TradeSignal(
                        symbol=f"{symbol}/USDT",
                        side=OrderSide.SELL,
                        quantity=0.001,
                        order_type=OrderType.LIMIT,
                        price=price * 1.01,  # 1% above current price
                        strategy_name=self.name
                    )
                    signals.append(signal)
        
        self.signal_count += 1
        self.last_signal_time = datetime.now()
        
        return signals


class MockRiskManager:
    """Mock risk manager for demonstration."""
    
    def __init__(self):
        self.max_order_value = 1000.0
        self.daily_order_limit = 10
        self.daily_orders = 0
        self.last_reset = datetime.now()
    
    def check_order_risk(self, signal: TradeSignal) -> tuple[bool, str]:
        """Check if order meets risk requirements."""
        # Reset daily counter if needed
        if datetime.now() - self.last_reset > timedelta(days=1):
            self.daily_orders = 0
            self.last_reset = datetime.now()
        
        # Check daily limit
        if self.daily_orders >= self.daily_order_limit:
            logger.warning("Daily order limit reached")
            return False, "Daily order limit reached"
        
        # Check order value
        order_value = signal.quantity * (signal.price or 50000)
        if order_value > self.max_order_value:
            logger.warning(f"Order value ${order_value:.2f} exceeds limit ${self.max_order_value}")
            return False, f"Order value ${order_value:.2f} exceeds limit ${self.max_order_value}"
        
        self.daily_orders += 1
        return True, "Order approved"


class MockPositionManager:
    """Mock position manager for demonstration."""
    
    def __init__(self):
        self.positions: Dict[str, float] = {}
        self.position_history: List[Dict[str, Any]] = []
    
    async def update_position(self, order):
        """Update position after order fill."""
        symbol = order.symbol.split('/')[0]  # Extract base currency
        
        if symbol not in self.positions:
            self.positions[symbol] = 0.0
        
        # Update position
        if order.side.value == 'buy':
            self.positions[symbol] += order.filled_quantity
        else:
            self.positions[symbol] -= order.filled_quantity
        
        # Record position update
        self.position_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': order.side.value,
            'quantity': order.filled_quantity,
            'price': order.average_price,
            'position': self.positions[symbol]
        })
        
        logger.info(f"Position updated: {symbol} = {self.positions[symbol]:.6f}")
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        return self.positions.copy()
    
    def get_position_history(self) -> List[Dict[str, Any]]:
        """Get position history."""
        return self.position_history.copy()


class TradingSystem:
    """Complete trading system integrating strategy, risk management, and execution."""
    
    def __init__(self, exchange_name: str = "binance"):
        self.order_manager = OrderManager(exchange_name)
        self.strategy = MockStrategy("integration_demo")
        self.risk_manager = MockRiskManager()
        self.position_manager = MockPositionManager()
        
        # Connect components
        self.order_manager.set_risk_manager(self.risk_manager)
        self.order_manager.set_position_manager(self.position_manager)
        
        self.running = False
        self.trade_count = 0
        
        logger.info("Trading system initialized")
    
    async def start(self):
        """Start the trading system."""
        self.running = True
        logger.info("Trading system started")
        
        try:
            while self.running:
                await self.trading_cycle()
                await asyncio.sleep(30)  # Run every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await self.stop()
    
    async def trading_cycle(self):
        """Execute one trading cycle."""
        try:
            # 1. Get market data (simulated)
            market_data = self.get_market_data()
            
            # 2. Generate signals from strategy
            signals = self.strategy.generate_signals(market_data)
            
            if not signals:
                logger.debug("No signals generated this cycle")
                return
            
            # 3. Submit orders for each signal
            for signal in signals:
                try:
                    order = await self.order_manager.submit_order(signal)
                    self.trade_count += 1
                    logger.info(f"Order submitted: {signal.symbol} {signal.side.value} {signal.quantity}")
                    
                    # Track order status
                    await self.order_manager.track_order(order.id)
                    
                except Exception as e:
                    logger.error(f"Failed to submit order: {e}")
            
            # 4. Update positions and performance
            await self.update_performance()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def get_market_data(self) -> Dict[str, float]:
        """Get current market data (simulated)."""
        # Simulate market data
        import random
        base_prices = {
            'BTC': 50000 + random.uniform(-1000, 1000),
            'ETH': 3000 + random.uniform(-100, 100),
            'SOL': 100 + random.uniform(-10, 10)
        }
        return base_prices
    
    async def update_performance(self):
        """Update and log performance metrics."""
        # Get order manager metrics
        order_metrics = self.order_manager.get_performance_metrics()
        
        # Get position information
        positions = self.position_manager.get_positions()
        
        # Log performance summary
        logger.info(f"Performance Update:")
        logger.info(f"  Total trades: {self.trade_count}")
        logger.info(f"  Success rate: {order_metrics['success_rate']:.2%}")
        logger.info(f"  Active orders: {order_metrics['active_orders']}")
        logger.info(f"  Current positions: {positions}")
    
    async def stop(self):
        """Stop the trading system."""
        self.running = False
        logger.info("Stopping trading system...")
        
        # Cancel all active orders
        active_orders = self.order_manager.get_active_orders()
        for order in active_orders:
            try:
                await self.order_manager.cancel_order(order.id)
            except Exception as e:
                logger.error(f"Failed to cancel order {order.id}: {e}")
        
        # Shutdown order manager
        await self.order_manager.shutdown()
        
        # Log final performance
        await self.update_performance()
        logger.info("Trading system stopped")


async def demo_trading_system():
    """Demonstrate the complete trading system."""
    logger.info("=== Trading System Integration Demo ===")
    
    # Create trading system
    trading_system = TradingSystem("binance")
    
    try:
        # Run for a limited time (2 minutes)
        logger.info("Starting trading system for 2 minutes...")
        
        # Run trading cycles manually for demo
        for i in range(4):  # 4 cycles = 2 minutes
            logger.info(f"\n--- Trading Cycle {i+1} ---")
            await trading_system.trading_cycle()
            await asyncio.sleep(30)  # Wait 30 seconds between cycles
        
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
    
    finally:
        await trading_system.stop()


async def demo_manual_trading():
    """Demonstrate manual trading operations."""
    logger.info("\n=== Manual Trading Demo ===")
    
    order_manager = OrderManager("binance")
    risk_manager = MockRiskManager()
    position_manager = MockPositionManager()
    
    order_manager.set_risk_manager(risk_manager)
    order_manager.set_position_manager(position_manager)
    
    try:
        # Create a manual trade signal
        signal = TradeSignal(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.001,
            order_type=OrderType.LIMIT,
            price=45000.0,
            strategy_name="manual_demo"
        )
        
        logger.info("Submitting manual order...")
        order = await order_manager.submit_order(signal)
        logger.info(f"Order submitted: {order.id}")
        
        # Track order
        await order_manager.track_order(order.id)
        logger.info(f"Order status: {order.status.value}")
        
        # Show active orders
        active_orders = order_manager.get_active_orders()
        logger.info(f"Active orders: {len(active_orders)}")
        
        # Cancel order
        if active_orders:
            await order_manager.cancel_order(order.id)
            logger.info("Order cancelled")
        
    except Exception as e:
        logger.error(f"Manual trading error: {e}")
    
    finally:
        await order_manager.shutdown()


async def main():
    """Run the integration demo."""
    logger.info("Starting Strategy Integration Demo")
    logger.info("=" * 50)
    
    # Run demos
    await demo_manual_trading()
    await demo_trading_system()
    
    logger.info("\nIntegration demo completed!")


if __name__ == "__main__":
    asyncio.run(main()) 