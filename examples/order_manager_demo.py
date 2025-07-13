#!/usr/bin/env python3
"""
Order Manager Demo

This script demonstrates how to use the OrderManager for crypto trading.
It shows order submission, tracking, cancellation, and batch operations.

Usage:
    python examples/order_manager_demo.py

Note: This is a demo script. Set up your API keys in config/secrets.env before running.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.execution.order_manager import (
    OrderManager, 
    TradeSignal, 
    OrderSide, 
    OrderType
)
from src.utils.logger import logger


class MockRiskManager:
    """Mock risk manager for demonstration."""
    
    def check_order_risk(self, signal):
        """Simple risk check - allow orders under $1000."""
        max_order_value = 1000.0
        order_value = signal.quantity * (signal.price or 50000)  # Use BTC price as default
        
        if order_value > max_order_value:
            logger.warning(f"Order rejected: Value ${order_value:.2f} exceeds limit ${max_order_value}")
            return False
        return True


class MockPositionManager:
    """Mock position manager for demonstration."""
    
    def __init__(self):
        self.positions = {}
    
    async def update_position(self, order):
        """Update position after order fill."""
        symbol = order.symbol
        if symbol not in self.positions:
            self.positions[symbol] = 0.0
        
        if order.side.value == 'buy':
            self.positions[symbol] += order.filled_quantity
        else:
            self.positions[symbol] -= order.filled_quantity
        
        logger.info(f"Position updated: {symbol} = {self.positions[symbol]:.6f}")


async def demo_basic_operations():
    """Demonstrate basic order manager operations."""
    logger.info("=== Order Manager Demo - Basic Operations ===")
    
    # Initialize order manager
    order_manager = OrderManager("binance")
    
    # Set up mock managers
    risk_manager = MockRiskManager()
    position_manager = MockPositionManager()
    order_manager.set_risk_manager(risk_manager)
    order_manager.set_position_manager(position_manager)
    
    try:
        # Example 1: Submit a limit buy order
        logger.info("\n1. Submitting limit buy order...")
        buy_signal = TradeSignal(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.001,  # Small amount for demo
            order_type=OrderType.LIMIT,
            price=45000.0,  # Below current market price
            strategy_name="demo_strategy"
        )
        
        buy_order = await order_manager.submit_order(buy_signal)
        logger.info(f"Buy order submitted: {buy_order.id}")
        logger.info(f"Order status: {buy_order.status.value}")
        
        # Example 2: Submit a market sell order
        logger.info("\n2. Submitting market sell order...")
        sell_signal = TradeSignal(
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            quantity=0.01,
            order_type=OrderType.MARKET,
            strategy_name="demo_strategy"
        )
        
        sell_order = await order_manager.submit_order(sell_signal)
        logger.info(f"Sell order submitted: {sell_order.id}")
        
        # Example 3: Track order status
        logger.info("\n3. Tracking order status...")
        await order_manager.track_order(buy_order.id)
        logger.info(f"Updated order status: {buy_order.status.value}")
        
        # Example 4: Get active orders
        logger.info("\n4. Active orders:")
        active_orders = order_manager.get_active_orders()
        for order in active_orders:
            logger.info(f"  - {order.symbol}: {order.side.value} {order.quantity} @ {order.price}")
        
        # Example 5: Cancel an order
        logger.info("\n5. Cancelling order...")
        if active_orders:
            order_to_cancel = active_orders[0]
            success = await order_manager.cancel_order(order_to_cancel.id)
            logger.info(f"Order cancellation: {'Success' if success else 'Failed'}")
        
        # Example 6: Performance metrics
        logger.info("\n6. Performance metrics:")
        metrics = order_manager.get_performance_metrics()
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
            
    except Exception as e:
        logger.error(f"Demo error: {e}")
    
    finally:
        await order_manager.shutdown()


async def demo_batch_operations():
    """Demonstrate batch order operations."""
    logger.info("\n=== Order Manager Demo - Batch Operations ===")
    
    order_manager = OrderManager("binance")
    risk_manager = MockRiskManager()
    order_manager.set_risk_manager(risk_manager)
    
    try:
        # Create multiple signals for portfolio rebalancing
        signals = [
            TradeSignal(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=0.001,
                order_type=OrderType.LIMIT,
                price=45000.0,
                strategy_name="portfolio_rebalance"
            ),
            TradeSignal(
                symbol="ETH/USDT",
                side=OrderSide.SELL,
                quantity=0.01,
                order_type=OrderType.LIMIT,
                price=3200.0,
                strategy_name="portfolio_rebalance"
            ),
            TradeSignal(
                symbol="SOL/USDT",
                side=OrderSide.BUY,
                quantity=1.0,
                order_type=OrderType.MARKET,
                strategy_name="portfolio_rebalance"
            )
        ]
        
        logger.info(f"Submitting {len(signals)} orders in batch...")
        orders = await order_manager.batch_submit_orders(signals)
        
        logger.info(f"Batch submission complete: {len(orders)} orders submitted")
        for i, order in enumerate(orders, 1):
            logger.info(f"  Order {i}: {order.symbol} {order.side.value} {order.quantity}")
        
        # Clean up - cancel all orders
        logger.info("Cancelling all orders...")
        for order in orders:
            if order.id in order_manager.active_orders:
                await order_manager.cancel_order(order.id)
        
    except Exception as e:
        logger.error(f"Batch demo error: {e}")
    
    finally:
        await order_manager.shutdown()


async def demo_error_handling():
    """Demonstrate error handling scenarios."""
    logger.info("\n=== Order Manager Demo - Error Handling ===")
    
    order_manager = OrderManager("binance")
    risk_manager = MockRiskManager()
    order_manager.set_risk_manager(risk_manager)
    
    try:
        # Example 1: Invalid signal (no price for limit order)
        logger.info("\n1. Testing invalid signal...")
        invalid_signal = TradeSignal(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.001,
            order_type=OrderType.LIMIT,  # Missing price
            strategy_name="error_demo"
        )
        
        try:
            await order_manager.submit_order(invalid_signal)
        except ValueError as e:
            logger.info(f"Caught expected error: {e}")
        
        # Example 2: Order that exceeds risk limits
        logger.info("\n2. Testing risk limit violation...")
        large_signal = TradeSignal(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,  # Large amount
            order_type=OrderType.LIMIT,
            price=50000.0,
            strategy_name="error_demo"
        )
        
        try:
            await order_manager.submit_order(large_signal)
        except ValueError as e:
            logger.info(f"Caught expected error: {e}")
        
        # Example 3: Tracking non-existent order
        logger.info("\n3. Testing tracking non-existent order...")
        try:
            await order_manager.track_order("non_existent_id")
        except ValueError as e:
            logger.info(f"Caught expected error: {e}")
        
    except Exception as e:
        logger.error(f"Error handling demo error: {e}")
    
    finally:
        await order_manager.shutdown()


async def main():
    """Run all demos."""
    logger.info("Starting Order Manager Demo")
    logger.info("=" * 50)
    
    # Run demos
    await demo_basic_operations()
    await demo_batch_operations()
    await demo_error_handling()
    
    logger.info("\nDemo completed successfully!")
    logger.info("Check the logs for detailed information about each operation.")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 