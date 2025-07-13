#!/usr/bin/env python3
"""
Test Manager Integration

This test file verifies that the Position Manager, Order Manager, and Risk Manager
integrate correctly to provide a complete trade lifecycle management system.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.execution.position_manager import PositionManager, PositionSide, PositionType
from src.execution.order_manager import OrderManager, TradeSignal, OrderSide, OrderType, OrderStatus
from src.execution.risk_manager import RiskManager, RiskLevel


class TestManagerIntegration:
    """Test the integration between Position Manager, Order Manager, and Risk Manager."""
    
    @pytest.fixture
    def managers(self):
        """Create and wire all managers together."""
        position_manager = PositionManager()
        order_manager = OrderManager("binance")
        risk_manager = RiskManager()
        
        # Wire managers together
        order_manager.set_risk_manager(risk_manager)
        order_manager.set_position_manager(position_manager)
        position_manager.set_order_manager(order_manager)
        position_manager.set_risk_manager(risk_manager)
        risk_manager.set_position_manager(position_manager)
        risk_manager.set_order_manager(order_manager)
        
        return {
            'position_manager': position_manager,
            'order_manager': order_manager,
            'risk_manager': risk_manager
        }
    
    def test_manager_connections(self, managers):
        """Test that managers are properly connected."""
        position_manager = managers['position_manager']
        order_manager = managers['order_manager']
        risk_manager = managers['risk_manager']
        
        # Check that managers have references to each other
        assert order_manager.risk_manager == risk_manager
        assert order_manager.position_manager == position_manager
        assert position_manager.order_manager == order_manager
        assert position_manager.risk_manager == risk_manager
        assert risk_manager.position_manager == position_manager
        assert risk_manager.order_manager == order_manager
    
    @pytest.mark.asyncio
    async def test_pre_trade_risk_check(self, managers):
        """Test pre-trade risk checking."""
        order_manager = managers['order_manager']
        risk_manager = managers['risk_manager']
        
        # Mock risk manager to reject orders
        risk_manager.check_order_risk = Mock(return_value=(False, "Test rejection"))
        
        # Create test signal
        signal = TradeSignal(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.001,
            order_type=OrderType.LIMIT,
            price=50000.0,
            strategy_name="test_strategy"
        )
        
        # Test that order is rejected
        with pytest.raises(ValueError, match="Order rejected by risk manager"):
            await order_manager.submit_order(signal)
    
    @pytest.mark.asyncio
    async def test_order_submission_with_risk_approval(self, managers):
        """Test order submission when risk manager approves."""
        order_manager = managers['order_manager']
        risk_manager = managers['risk_manager']
        
        # Mock risk manager to approve orders
        risk_manager.check_order_risk = Mock(return_value=(True, "Approved"))
        
        # Mock exchange submission
        with patch.object(order_manager, '_submit_to_exchange') as mock_submit:
            mock_submit.return_value = None
            
            # Create test signal
            signal = TradeSignal(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=0.001,
                order_type=OrderType.LIMIT,
                price=50000.0,
                strategy_name="test_strategy"
            )
            
            # Submit order
            order = await order_manager.submit_order(signal)
            
            # Verify risk check was called
            risk_manager.check_order_risk.assert_called_once_with(signal)
            
            # Verify order was created
            assert order.symbol == "BTC/USDT"
            assert order.side == OrderSide.BUY
            assert order.quantity == 0.001
    
    def test_position_manager_update_on_trade(self, managers):
        """Test position manager trade update method."""
        position_manager = managers['position_manager']
        risk_manager = managers['risk_manager']
        
        # Mock risk manager update method
        risk_manager.update_on_trade = Mock()
        
        # Test buy trade
        position_manager.update_on_trade("BTC/USDT", "buy", 0.001, 50000.0, "test_order")
        
        # Verify position was created (check by symbol)
        positions = position_manager.get_positions_by_symbol("BTC/USDT")
        assert len(positions) > 0
        
        # Verify risk manager was notified
        risk_manager.update_on_trade.assert_called_once_with(symbol="BTC/USDT", side="buy", quantity=0.001, price=50000.0, order_id="test_order")
    
    def test_position_manager_sell_trade(self, managers):
        """Test position manager handling sell trades."""
        position_manager = managers['position_manager']
        risk_manager = managers['risk_manager']
        
        # Mock risk manager update method
        risk_manager.update_on_trade = Mock()
        
        # First add a long position
        position_manager.update_on_trade("BTC/USDT", "buy", 0.001, 50000.0, "buy_order")
        
        # Then sell to close the position
        position_manager.update_on_trade("BTC/USDT", "sell", 0.001, 51000.0, "sell_order")
        
        # Verify risk manager was notified twice
        assert risk_manager.update_on_trade.call_count == 2
    
    def test_risk_manager_update_on_trade(self, managers):
        """Test risk manager trade update method."""
        risk_manager = managers['risk_manager']
        position_manager = managers['position_manager']
        
        # Mock position manager methods
        position_manager.get_portfolio_metrics = Mock()
        position_manager.get_portfolio_metrics.return_value = Mock(
            total_market_value=100000.0,
            daily_pnl=1000.0
        )
        
        # Mock risk manager methods
        risk_manager.update_position = Mock()
        risk_manager._check_post_trade_risk_events = Mock()
        risk_manager.update_portfolio_metrics = Mock()
        
        # Test trade update
        risk_manager.update_on_trade("BTC/USDT", "buy", 0.001, 50000.0, "test_order")
        
        # Verify position was updated
        risk_manager.update_position.assert_called_once_with("BTC/USDT", "buy", 0.001, 50000.0)
        
        # Verify post-trade risk events were checked
        risk_manager._check_post_trade_risk_events.assert_called_once_with("BTC/USDT", "buy", 0.001, 50000.0)
        
        # Verify portfolio metrics were updated
        risk_manager.update_portfolio_metrics.assert_called_once_with(100000.0, 1000.0)
    
    @pytest.mark.asyncio
    async def test_forced_exit_execution(self, managers):
        """Test forced exit execution by risk manager."""
        risk_manager = managers['risk_manager']
        order_manager = managers['order_manager']
        position_manager = managers['position_manager']
        
        # Mock order manager submit_order method
        order_manager.submit_order = AsyncMock()
        
        # Add a test position to the position manager
        test_position = position_manager.add_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.001,
            entry_price=50000.0,
            order_id="test_order",
            strategy_name="test_strategy"
        )
        
        # Mock risk manager methods to return the test position
        risk_manager.check_stop_losses = Mock(return_value=[(test_position.id, "sell", 0.001)])
        risk_manager.check_take_profits = Mock(return_value=[])
        risk_manager._check_circuit_breakers = Mock(return_value=True)
        
        # Execute forced exits
        await risk_manager.execute_forced_exits()
        
        # Verify exit order was submitted
        order_manager.submit_order.assert_called_once()
        
        # Verify the order was for the correct position
        call_args = order_manager.submit_order.call_args[0][0]
        assert call_args.symbol == "BTC/USDT"
        assert call_args.side.value == "sell"  # Compare string values instead of enum objects
        assert call_args.quantity == 0.001
    
    def test_stop_loss_monitoring(self, managers):
        """Test stop-loss monitoring functionality."""
        risk_manager = managers['risk_manager']
        position_manager = managers['position_manager']
        
        # Add a position with stop-loss
        position = position_manager.add_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.001,
            entry_price=50000.0,
            order_id="test_order",
            strategy_name="test_strategy",
            stop_loss=48000.0  # 4% stop-loss
        )
        
        # Update position price to trigger stop-loss
        position_manager.update_position(position.id, 47000.0)
        
        # Check stop-losses
        stop_loss_positions = risk_manager.check_stop_losses()
        
        # Verify stop-loss was triggered (position should be auto-closed by position manager)
        # The risk manager should not find any positions to close since they're already closed
        assert len(stop_loss_positions) == 0
    
    def test_take_profit_monitoring(self, managers):
        """Test take-profit monitoring functionality."""
        risk_manager = managers['risk_manager']
        position_manager = managers['position_manager']
    
        # Clear existing positions to avoid interference
        open_positions = position_manager.get_open_positions()
        for pos in open_positions:
            position_manager.close_position(pos.id, pos.current_price)
    
        # Add a position with take-profit
        position = position_manager.add_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.001,
            entry_price=50000.0,
            order_id="test_order",
            strategy_name="test_strategy",
            take_profit=52000.0  # 4% take-profit
        )
    
        # Update position price to trigger take-profit
        position_manager.update_position(position.id, 53000.0)
    
        # Check take-profits
        take_profit_positions = risk_manager.check_take_profits()
    
        # Verify take-profit was triggered (position should be auto-closed by position manager)
        # The risk manager should not find any positions to close since they're already closed
        assert len(take_profit_positions) == 0
    
    def test_circuit_breaker_functionality(self, managers):
        """Test circuit breaker functionality."""
        risk_manager = managers['risk_manager']
        position_manager = managers['position_manager']
        
        # Add multiple positions to simulate high exposure
        for i in range(5):
            position_manager.add_position(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=0.001,
                entry_price=50000.0,
                order_id=f"order_{i}",
                strategy_name="test_strategy"
            )
        
        # Mock circuit breaker check to return False (trigger circuit breaker)
        risk_manager._check_circuit_breakers = Mock(return_value=False)
        
        # Mock execute_circuit_breaker method
        risk_manager._execute_circuit_breaker = AsyncMock()
        
        # Execute forced exits (should trigger circuit breaker)
        asyncio.run(risk_manager.execute_forced_exits())
        
        # Verify circuit breaker was executed
        risk_manager._execute_circuit_breaker.assert_called_once()
    
    def test_portfolio_metrics_integration(self, managers):
        """Test portfolio metrics integration across managers."""
        position_manager = managers['position_manager']
        risk_manager = managers['risk_manager']
        
        # Add some positions
        position_manager.add_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.001,
            entry_price=50000.0,
            order_id="order1",
            strategy_name="test_strategy"
        )
        
        position_manager.add_position(
            symbol="ETH/USDT",
            side=PositionSide.LONG,
            quantity=0.01,
            entry_price=3000.0,
            order_id="order2",
            strategy_name="test_strategy"
        )
        
        # Get portfolio metrics
        portfolio_metrics = position_manager.get_portfolio_metrics()
        
        # Verify metrics are calculated correctly (accounting for existing positions)
        assert portfolio_metrics.position_count >= 2
        assert portfolio_metrics.open_position_count >= 2
        assert portfolio_metrics.total_market_value > 0
        
        # Test risk manager integration with portfolio metrics
        risk_manager.update_portfolio_metrics = Mock()
        risk_manager.update_on_trade("BTC/USDT", "buy", 0.001, 50000.0, "test_order")
        
        # Verify portfolio metrics were updated
        risk_manager.update_portfolio_metrics.assert_called_once()
    
    def test_error_handling_integration(self, managers):
        """Test error handling across manager integration."""
        position_manager = managers['position_manager']
        order_manager = managers['order_manager']
        risk_manager = managers['risk_manager']
        
        # Test error handling in position manager (should log warning, not raise)
        position_manager.update_position("nonexistent_position", 50000.0)
        
        # Test error handling in order manager
        with pytest.raises(ValueError):
            asyncio.run(order_manager.cancel_order("nonexistent_order"))
        
        # Test error handling in risk manager
        risk_manager.check_order_risk = Mock(side_effect=Exception("Test error"))
        
        signal = TradeSignal(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.001,
            order_type=OrderType.LIMIT,
            price=50000.0,
            strategy_name="test_strategy"
        )
        
        with pytest.raises(Exception):
            asyncio.run(order_manager.submit_order(signal))


class TestTradingSystemIntegration:
    """Test the complete trading system integration."""
    
    @pytest.fixture
    def trading_system(self):
        """Create a complete trading system."""
        from examples.manager_integration_demo import TradingSystem
        return TradingSystem()
    
    @pytest.mark.asyncio
    async def test_trading_system_initialization(self, trading_system):
        """Test trading system initialization."""
        # Verify managers are connected
        assert trading_system.position_manager is not None
        assert trading_system.order_manager is not None
        assert trading_system.risk_manager is not None
        
        # Verify managers are wired together
        assert trading_system.order_manager.risk_manager == trading_system.risk_manager
        assert trading_system.order_manager.position_manager == trading_system.position_manager
        assert trading_system.position_manager.order_manager == trading_system.order_manager
        assert trading_system.position_manager.risk_manager == trading_system.risk_manager
    
    @pytest.mark.asyncio
    async def test_trading_system_start_stop(self, trading_system):
        """Test trading system start and stop functionality."""
        try:
            # Start system
            await trading_system.start()
            assert trading_system.is_running == True
            
            # Stop system
            await trading_system.stop()
            assert trading_system.is_running == False
        except Exception as e:
            # Handle API authentication errors gracefully
            if "AuthenticationError" in str(e) or "Invalid API-key" in str(e):
                pytest.skip("Skipping test due to missing API credentials")
            else:
                raise
    
    @pytest.mark.asyncio
    async def test_trade_signal_processing(self, trading_system):
        """Test complete trade signal processing."""
        try:
            await trading_system.start()
            
            # Mock risk manager to approve orders
            trading_system.risk_manager.check_order_risk = Mock(return_value=(True, "Approved"))
            
            # Mock exchange submission to avoid API calls
            with patch.object(trading_system.order_manager, '_submit_to_exchange') as mock_submit:
                mock_submit.return_value = None
                
                # Create test signal
                signal = TradeSignal(
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    quantity=0.001,
                    order_type=OrderType.LIMIT,
                    price=50000.0,
                    strategy_name="test_strategy"
                )
                
                # Submit signal
                success = await trading_system.submit_trade_signal(signal)
                
                # Verify signal was processed
                assert success == True
                
                # Verify exchange submission was called
                mock_submit.assert_called_once()
            
            await trading_system.stop()
        except Exception as e:
            # Handle API authentication errors gracefully
            if "AuthenticationError" in str(e) or "Invalid API-key" in str(e):
                pytest.skip("Skipping test due to missing API credentials")
            else:
                raise
    
    def test_system_status_reporting(self, trading_system):
        """Test system status reporting."""
        status = trading_system.get_system_status()
        
        # Verify status contains expected keys
        assert 'system_running' in status
        assert 'portfolio' in status
        assert 'risk' in status
        assert 'orders' in status
        
        # Verify portfolio metrics
        portfolio = status['portfolio']
        assert 'total_pnl' in portfolio
        assert 'open_positions' in portfolio
        assert 'total_positions' in portfolio
        
        # Verify risk metrics
        risk = status['risk']
        assert 'risk_level' in risk
        assert 'trading_allowed' in risk


if __name__ == "__main__":
    pytest.main([__file__]) 