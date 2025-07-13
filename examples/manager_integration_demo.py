#!/usr/bin/env python3
"""
Manager Integration Demo

This demo shows how to properly integrate the Position Manager, Order Manager, 
and Risk Manager for a complete, automated trading system with real-time risk 
management and position tracking.

Key Integration Points:
1. Pre-trade risk checks before order submission
2. Position updates on order completion
3. Real-time risk monitoring and forced exits
4. Circuit breakers and emergency shutdowns
5. Comprehensive audit trail and logging
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from src.execution.position_manager import PositionManager, PositionSide, PositionType
from src.execution.order_manager import OrderManager, TradeSignal, OrderSide, OrderType
from src.execution.risk_manager import RiskManager
from src.utils.logger import logger
from src.utils.config_loader import config


class TradingSystem:
    """Complete trading system with integrated managers."""
    
    def __init__(self):
        """Initialize the trading system with all managers."""
        logger.info("Initializing trading system...")
        
        # Initialize managers
        self.position_manager = PositionManager()
        self.order_manager = OrderManager("binance")
        self.risk_manager = RiskManager()
        
        # Wire managers together
        self._connect_managers()
        
        # System state
        self.is_running = False
        self.risk_monitoring_task = None
        
        logger.info("Trading system initialized successfully")
    
    def _connect_managers(self):
        """Connect all managers with proper references."""
        logger.info("Connecting managers...")
        
        # Order Manager connections
        self.order_manager.set_risk_manager(self.risk_manager)
        self.order_manager.set_position_manager(self.position_manager)
        
        # Position Manager connections
        self.position_manager.set_order_manager(self.order_manager)
        self.position_manager.set_risk_manager(self.risk_manager)
        
        # Risk Manager connections
        self.risk_manager.set_position_manager(self.position_manager)
        self.risk_manager.set_order_manager(self.order_manager)
        
        logger.info("All managers connected successfully")
    
    async def start(self):
        """Start the trading system."""
        logger.info("Starting trading system...")
        self.is_running = True
        
        # Start risk monitoring loop
        self.risk_monitoring_task = asyncio.create_task(self._risk_monitoring_loop())
        
        # Initialize exchange connection
        await self.order_manager.exchange_manager.load_markets()
        
        logger.info("Trading system started successfully")
    
    async def stop(self):
        """Stop the trading system."""
        logger.info("Stopping trading system...")
        self.is_running = False
        
        # Cancel risk monitoring
        if self.risk_monitoring_task:
            self.risk_monitoring_task.cancel()
            try:
                await self.risk_monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown managers
        await self.order_manager.shutdown()
        await self.position_manager.shutdown()
        
        logger.info("Trading system stopped")
    
    async def submit_trade_signal(self, signal: TradeSignal) -> bool:
        """
        Submit a trade signal with full risk management.
        
        Args:
            signal: Trade signal from strategy
            
        Returns:
            True if order was submitted successfully
        """
        try:
            logger.info(f"Processing trade signal: {signal.side.value} {signal.quantity} {signal.symbol}")
            
            # Step 1: Pre-trade risk check
            if self.risk_manager:
                is_allowed, reason = self.risk_manager.check_order_risk(signal)
                if not is_allowed:
                    logger.warning(f"Trade signal rejected by risk manager: {reason}")
                    return False
            
            # Step 2: Submit order
            order = await self.order_manager.submit_order(signal)
            logger.info(f"Order submitted successfully: {order.id}")
            
            # Step 3: Monitor order execution
            await self._monitor_order_execution(order.id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit trade signal: {e}")
            return False
    
    async def _monitor_order_execution(self, order_id: str, timeout: int = 300):
        """Monitor order execution with timeout."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                order = await self.order_manager.track_order(order_id)
                
                if order.status.value in ['filled', 'cancelled', 'rejected']:
                    logger.info(f"Order {order_id} completed with status: {order.status.value}")
                    return order
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring order {order_id}: {e}")
                await asyncio.sleep(5)
        
        logger.warning(f"Order {order_id} monitoring timed out")
        return None
    
    async def _risk_monitoring_loop(self):
        """Continuous risk monitoring loop."""
        logger.info("Starting risk monitoring loop...")
        
        while self.is_running:
            try:
                # Check for forced exits (stop-loss, take-profit, circuit breakers)
                await self.risk_manager.execute_forced_exits()
                
                # Update portfolio metrics
                if self.position_manager:
                    portfolio_metrics = self.position_manager.get_portfolio_metrics()
                    logger.info(f"Portfolio PnL: ${portfolio_metrics.total_pnl:.2f} "
                              f"({portfolio_metrics.total_pnl_percentage:.2f}%)")
                
                # Check risk level
                risk_level = self.risk_manager.get_risk_level()
                if risk_level.value in ['high', 'critical']:
                    logger.warning(f"High risk level detected: {risk_level.value}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                logger.info("Risk monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(30)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Portfolio metrics
            portfolio_metrics = self.position_manager.get_portfolio_metrics()
            
            # Risk metrics
            risk_events = self.risk_manager.get_risk_events(limit=10)
            risk_level = self.risk_manager.get_risk_level()
            
            # Order metrics
            order_metrics = self.order_manager.get_performance_metrics()
            
            return {
                'system_running': self.is_running,
                'portfolio': {
                    'total_pnl': portfolio_metrics.total_pnl,
                    'total_pnl_percentage': portfolio_metrics.total_pnl_percentage,
                    'open_positions': portfolio_metrics.open_position_count,
                    'total_positions': portfolio_metrics.position_count,
                    'max_drawdown': portfolio_metrics.max_drawdown_percentage
                },
                'risk': {
                    'risk_level': risk_level.value,
                    'recent_events': len(risk_events),
                    'trading_allowed': self.risk_manager.is_trading_allowed()
                },
                'orders': {
                    'active_orders': order_metrics['active_orders'],
                    'success_rate': order_metrics['success_rate'],
                    'total_orders': order_metrics['total_orders']
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}


async def demo_basic_integration():
    """Demo basic manager integration."""
    logger.info("=== Basic Manager Integration Demo ===")
    
    # Initialize trading system
    trading_system = TradingSystem()
    
    try:
        # Start system
        await trading_system.start()
        
        # Create test signals
        signals = [
            TradeSignal(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=0.001,
                order_type=OrderType.LIMIT,
                price=50000.0,
                strategy_name="demo_strategy"
            ),
            TradeSignal(
                symbol="ETH/USDT",
                side=OrderSide.BUY,
                quantity=0.01,
                order_type=OrderType.MARKET,
                strategy_name="demo_strategy"
            )
        ]
        
        # Submit signals (these will be rejected in demo mode)
        for i, signal in enumerate(signals):
            logger.info(f"Submitting signal {i+1}...")
            success = await trading_system.submit_trade_signal(signal)
            logger.info(f"Signal {i+1} result: {'SUCCESS' if success else 'FAILED'}")
            
            await asyncio.sleep(2)
        
        # Show system status
        status = trading_system.get_system_status()
        logger.info(f"System Status: {status}")
        
        # Run risk monitoring for a bit
        await asyncio.sleep(10)
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
    
    finally:
        await trading_system.stop()


async def demo_risk_management():
    """Demo risk management features."""
    logger.info("=== Risk Management Demo ===")
    
    trading_system = TradingSystem()
    
    try:
        await trading_system.start()
        
        # Add some test positions to demonstrate risk monitoring
        position = trading_system.position_manager.add_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.001,
            entry_price=50000.0,
            order_id="demo_order",
            strategy_name="demo_strategy",
            stop_loss=48000.0,  # 4% stop-loss
            take_profit=52000.0  # 4% take-profit
        )
        
        logger.info(f"Added test position: {position.id}")
        
        # Update position price to trigger risk events
        trading_system.position_manager.update_position(
            position.id, 
            47000.0  # Below stop-loss
        )
        
        # Run risk monitoring
        await asyncio.sleep(5)
        
        # Check for forced exits
        await trading_system.risk_manager.execute_forced_exits()
        
        # Show risk events
        risk_events = trading_system.risk_manager.get_risk_events(limit=5)
        logger.info(f"Recent risk events: {len(risk_events)}")
        
    except Exception as e:
        logger.error(f"Risk management demo error: {e}")
    
    finally:
        await trading_system.stop()


async def demo_circuit_breaker():
    """Demo circuit breaker functionality."""
    logger.info("=== Circuit Breaker Demo ===")
    
    trading_system = TradingSystem()
    
    try:
        await trading_system.start()
        
        # Add multiple positions to simulate high exposure
        for i in range(5):
            position = trading_system.position_manager.add_position(
                symbol=f"BTC/USDT",
                side=PositionSide.LONG,
                quantity=0.001,
                entry_price=50000.0,
                order_id=f"demo_order_{i}",
                strategy_name="demo_strategy"
            )
            logger.info(f"Added position {i+1}: {position.id}")
        
        # Simulate extreme market conditions
        open_positions = trading_system.position_manager.get_open_positions()
        for position in open_positions:
            # Update prices to simulate large losses
            trading_system.position_manager.update_position(
                position.id,
                45000.0  # 10% loss
            )
        
        # Check portfolio metrics
        portfolio_metrics = trading_system.position_manager.get_portfolio_metrics()
        logger.info(f"Portfolio PnL: ${portfolio_metrics.total_pnl:.2f}")
        
        # Run risk monitoring
        await asyncio.sleep(5)
        
        # Execute forced exits
        await trading_system.risk_manager.execute_forced_exits()
        
    except Exception as e:
        logger.error(f"Circuit breaker demo error: {e}")
    
    finally:
        await trading_system.stop()


async def main():
    """Run all integration demos."""
    logger.info("Starting Manager Integration Demos")
    
    # Demo 1: Basic Integration
    await demo_basic_integration()
    await asyncio.sleep(2)
    
    # Demo 2: Risk Management
    await demo_risk_management()
    await asyncio.sleep(2)
    
    # Demo 3: Circuit Breaker
    await demo_circuit_breaker()
    
    logger.info("All demos completed")


if __name__ == "__main__":
    asyncio.run(main()) 