import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from src.execution.order_manager import (
    OrderManager, 
    ExchangeManager, 
    TradeSignal, 
    Order, 
    OrderStatus, 
    OrderType, 
    OrderSide
)


class TestOrderManager:
    """Test cases for OrderManager."""
    
    @pytest_asyncio.fixture
    async def order_manager(self):
        """Create a test order manager instance."""
        with patch('src.execution.order_manager.ExchangeManager') as mock_exchange_manager:
            mock_exchange_manager.return_value.async_exchange = AsyncMock()
            order_manager = OrderManager("binance")
            return order_manager
            # Note: Removed yield and await order_manager.shutdown() since it's not implemented
    
    @pytest.fixture
    def valid_signal(self):
        """Create a valid trade signal."""
        return TradeSignal(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.001,
            order_type=OrderType.LIMIT,
            price=50000.0,
            strategy_name="test_strategy"
        )
    
    @pytest.fixture
    def market_signal(self):
        """Create a market order signal."""
        return TradeSignal(
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            quantity=0.1,
            order_type=OrderType.MARKET,
            strategy_name="test_strategy"
        )
    
    @pytest.mark.asyncio
    async def test_order_manager_initialization(self, order_manager):
        """Test order manager initialization."""
        assert order_manager.active_orders == {}
        assert order_manager.order_history == []
        assert order_manager.max_retries == 3
        assert order_manager.retry_delay == 1.0
        assert order_manager.max_slippage == 0.02
        assert order_manager.order_timeout == 300
    
    @pytest.mark.asyncio
    async def test_validate_signal_valid(self, order_manager, valid_signal):
        """Test signal validation with valid signal."""
        assert order_manager._validate_signal(valid_signal) is True
    
    @pytest.mark.asyncio
    async def test_validate_signal_invalid_symbol(self, order_manager):
        """Test signal validation with invalid symbol."""
        invalid_signal = TradeSignal(
            symbol="",
            side=OrderSide.BUY,
            quantity=0.001,
            order_type=OrderType.LIMIT,
            price=50000.0
        )
        assert order_manager._validate_signal(invalid_signal) is False
    
    @pytest.mark.asyncio
    async def test_validate_signal_invalid_quantity(self, order_manager):
        """Test signal validation with invalid quantity."""
        invalid_signal = TradeSignal(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0,
            order_type=OrderType.LIMIT,
            price=50000.0
        )
        assert order_manager._validate_signal(invalid_signal) is False
    
    @pytest.mark.asyncio
    async def test_validate_signal_limit_without_price(self, order_manager):
        """Test signal validation for limit order without price."""
        invalid_signal = TradeSignal(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.001,
            order_type=OrderType.LIMIT
        )
        assert order_manager._validate_signal(invalid_signal) is False
    
    @pytest.mark.asyncio
    async def test_validate_signal_stop_without_stop_price(self, order_manager):
        """Test signal validation for stop order without stop price."""
        invalid_signal = TradeSignal(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.001,
            order_type=OrderType.STOP
        )
        assert order_manager._validate_signal(invalid_signal) is False
    
    @pytest.mark.asyncio
    async def test_submit_order_success(self, order_manager, valid_signal):
        """Test successful order submission."""
        # Mock exchange response
        mock_response = {
            'id': 'exchange_order_123',
            'status': 'open',
            'filled': 0.0,
            'remaining': 0.001,
            'average': None,
            'fee': {'cost': 0.0}
        }
        order_manager.exchange_manager.async_exchange.create_limit_order.return_value = mock_response
        
        # Submit order
        order = await order_manager.submit_order(valid_signal)
        
        # Verify order creation
        assert order.id in order_manager.active_orders
        assert order.symbol == valid_signal.symbol
        assert order.side == valid_signal.side
        assert order.order_type == valid_signal.order_type
        assert order.quantity == valid_signal.quantity
        assert order.price == valid_signal.price
        assert order.status == OrderStatus.PENDING
        assert order.exchange_order_id == 'exchange_order_123'
    
    @pytest.mark.asyncio
    async def test_submit_market_order(self, order_manager, market_signal):
        """Test market order submission."""
        mock_response = {
            'id': 'market_order_123',
            'status': 'closed',
            'filled': 0.1,
            'remaining': 0.0,
            'average': 3000.0,
            'fee': {'cost': 0.5}
        }
        order_manager.exchange_manager.async_exchange.create_market_order.return_value = mock_response
        
        order = await order_manager.submit_order(market_signal)
        
        assert order.order_type == OrderType.MARKET
        assert order.price is None
        order_manager.exchange_manager.async_exchange.create_market_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_submit_order_with_risk_manager_rejection(self, order_manager, valid_signal):
        """Test order submission rejected by risk manager."""
        # Mock risk manager rejection
        mock_risk_manager = Mock()
        mock_risk_manager.check_order_risk.return_value = (False, "Test rejection")
        order_manager.risk_manager = mock_risk_manager
    
        with pytest.raises(ValueError, match="Order rejected by risk manager"):
            await order_manager.submit_order(valid_signal)
    
    @pytest.mark.asyncio
    async def test_submit_order_exchange_error(self, order_manager, valid_signal):
        """Test order submission with exchange error."""
        order_manager.exchange_manager.async_exchange.create_limit_order.side_effect = Exception("Exchange error")
        
        with pytest.raises(Exception):
            await order_manager.submit_order(valid_signal)
        
        # Verify order was added to history as rejected
        assert len(order_manager.order_history) == 1
        rejected_order = order_manager.order_history[0]
        assert rejected_order.status == OrderStatus.REJECTED
        assert rejected_order.error_message == "Exchange error"
    
    @pytest.mark.asyncio
    async def test_track_order_success(self, order_manager, valid_signal):
        """Test successful order tracking."""
        # Create and submit order
        mock_response = {
            'id': 'exchange_order_123',
            'status': 'open',
            'filled': 0.0,
            'remaining': 0.001,
            'average': None,
            'fee': {'cost': 0.0}
        }
        order_manager.exchange_manager.async_exchange.create_limit_order.return_value = mock_response
        order = await order_manager.submit_order(valid_signal)
        
        # Mock exchange order status
        exchange_order = {
            'id': 'exchange_order_123',
            'status': 'closed',
            'filled': 0.001,
            'remaining': 0.0,
            'average': 50000.0,
            'fee': {'cost': 0.25}
        }
        order_manager.exchange_manager.async_exchange.fetch_order.return_value = exchange_order
        
        # Track order
        updated_order = await order_manager.track_order(order.id)
        
        assert updated_order.status == OrderStatus.FILLED
        assert updated_order.filled_quantity == 0.001
        assert updated_order.remaining_quantity == 0.0
        assert updated_order.average_price == 50000.0
        assert updated_order.commission == 0.25
    
    @pytest.mark.asyncio
    async def test_track_order_not_found(self, order_manager):
        """Test tracking non-existent order."""
        with pytest.raises(ValueError, match="Order.*not found in active orders"):
            await order_manager.track_order("non_existent_id")
    
    @pytest.mark.asyncio
    async def test_cancel_order_success(self, order_manager, valid_signal):
        """Test successful order cancellation."""
        # Create and submit order
        mock_response = {
            'id': 'exchange_order_123',
            'status': 'open',
            'filled': 0.0,
            'remaining': 0.001,
            'average': None,
            'fee': {'cost': 0.0}
        }
        order_manager.exchange_manager.async_exchange.create_limit_order.return_value = mock_response
        order = await order_manager.submit_order(valid_signal)
        
        # Cancel order
        result = await order_manager.cancel_order(order.id)
        
        assert result is True
        assert order.id not in order_manager.active_orders
        assert order.status == OrderStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, order_manager):
        """Test cancelling non-existent order."""
        with pytest.raises(ValueError, match="Order.*not found in active orders"):
            await order_manager.cancel_order("non_existent_id")
    
    @pytest.mark.asyncio
    async def test_replace_order(self, order_manager, valid_signal):
        """Test order replacement."""
        # Create and submit order
        mock_response = {
            'id': 'exchange_order_123',
            'status': 'open',
            'filled': 0.0,
            'remaining': 0.001,
            'average': None,
            'fee': {'cost': 0.0}
        }
        order_manager.exchange_manager.async_exchange.create_limit_order.return_value = mock_response
        order = await order_manager.submit_order(valid_signal)
        
        # Mock new order response
        new_mock_response = {
            'id': 'exchange_order_456',
            'status': 'open',
            'filled': 0.0,
            'remaining': 0.001,
            'average': None,
            'fee': {'cost': 0.0}
        }
        order_manager.exchange_manager.async_exchange.create_limit_order.return_value = new_mock_response
        
        # Replace order
        new_order = await order_manager.replace_order(order.id, 49000.0)
        
        assert new_order.price == 49000.0
        assert new_order.id != order.id
        assert order.id not in order_manager.active_orders
        assert new_order.id in order_manager.active_orders
    
    @pytest.mark.asyncio
    async def test_get_active_orders(self, order_manager):
        """Test getting active orders."""
        # Add mock orders
        order1 = Order(
            id="order1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0,
            status=OrderStatus.PENDING
        )
        order2 = Order(
            id="order2",
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=3000.0,
            status=OrderStatus.PENDING
        )
        
        order_manager.active_orders["order1"] = order1
        order_manager.active_orders["order2"] = order2
        
        active_orders = order_manager.get_active_orders()
        assert len(active_orders) == 2
        assert order1 in active_orders
        assert order2 in active_orders
    
    @pytest.mark.asyncio
    async def test_get_order_history(self, order_manager):
        """Test getting order history."""
        # Add mock orders to history
        order1 = Order(
            id="order1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0,
            status=OrderStatus.FILLED
        )
        order2 = Order(
            id="order2",
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=3000.0,
            status=OrderStatus.CANCELLED
        )
        
        order_manager.order_history = [order1, order2]
        
        history = order_manager.get_order_history()
        assert len(history) == 2
        assert order1 in history
        assert order2 in history
    
    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, order_manager):
        """Test performance metrics calculation."""
        # Add mock orders to history
        filled_order = Order(
            id="filled1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0,
            status=OrderStatus.FILLED
        )
        cancelled_order = Order(
            id="cancelled1",
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=3000.0,
            status=OrderStatus.CANCELLED
        )
        rejected_order = Order(
            id="rejected1",
            symbol="SOL/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=100.0,
            status=OrderStatus.REJECTED
        )
        
        order_manager.order_history = [filled_order, cancelled_order, rejected_order]
        
        metrics = order_manager.get_performance_metrics()
        
        assert metrics['total_orders'] == 3
        assert metrics['filled_orders'] == 1
        assert metrics['cancelled_orders'] == 1
        assert metrics['rejected_orders'] == 1
        assert metrics['success_rate'] == 1/3
        assert metrics['active_orders'] == 0
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_orders(self, order_manager):
        """Test cleanup of expired orders."""
        # Create an expired order
        expired_order = Order(
            id="expired1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0,
            status=OrderStatus.PENDING,
            timestamp=datetime.now() - timedelta(minutes=10)  # 10 minutes ago
        )
        
        order_manager.active_orders["expired1"] = expired_order
        order_manager.order_timeout = 300  # 5 minutes
        
        # Mock cancel order
        order_manager.cancel_order = AsyncMock(return_value=True)
        
        await order_manager.cleanup_expired_orders()
        
        order_manager.cancel_order.assert_called_once_with("expired1")
    
    @pytest.mark.asyncio
    async def test_batch_submit_orders(self, order_manager):
        """Test batch order submission."""
        signals = [
            TradeSignal(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=0.001,
                order_type=OrderType.LIMIT,
                price=50000.0,
                strategy_name="batch_test"
            ),
            TradeSignal(
                symbol="ETH/USDT",
                side=OrderSide.SELL,
                quantity=0.1,
                order_type=OrderType.LIMIT,
                price=3000.0,
                strategy_name="batch_test"
            )
        ]
        
        # Mock successful order submission
        mock_response = {
            'id': 'batch_order_123',
            'status': 'open',
            'filled': 0.0,
            'remaining': 0.001,
            'average': None,
            'fee': {'cost': 0.0}
        }
        order_manager.exchange_manager.async_exchange.create_limit_order.return_value = mock_response
        
        orders = await order_manager.batch_submit_orders(signals)
        
        assert len(orders) == 2
        assert all(order.status == OrderStatus.PENDING for order in orders)


class TestExchangeManager:
    """Test cases for ExchangeManager."""
    
    @pytest.fixture
    def exchange_manager(self):
        """Create a test exchange manager instance."""
        with patch('src.execution.order_manager.config') as mock_config:
            mock_config.get_exchange_config.return_value = {
                'apiKey': 'test_key',
                'secret': 'test_secret',
                'sandbox': True
            }
            with patch('src.execution.order_manager.ccxt') as mock_ccxt:
                mock_exchange_class = Mock()
                mock_ccxt.binance = mock_exchange_class
                mock_ccxt_async = Mock()
                mock_ccxt_async.binance = Mock()
                
                with patch('src.execution.order_manager.ccxt_async', mock_ccxt_async):
                    exchange_manager = ExchangeManager("binance")
                    return exchange_manager
    
    @pytest.mark.asyncio
    async def test_load_markets(self, exchange_manager):
        """Test loading markets."""
        exchange_manager.async_exchange.load_markets = AsyncMock()
        
        await exchange_manager.load_markets()
        
        exchange_manager.async_exchange.load_markets.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_ticker(self, exchange_manager):
        """Test getting ticker information."""
        mock_ticker = {
            'symbol': 'BTC/USDT',
            'last': 50000.0,
            'bid': 49999.0,
            'ask': 50001.0
        }
        exchange_manager.async_exchange.fetch_ticker = AsyncMock(return_value=mock_ticker)
        
        ticker = await exchange_manager.get_ticker("BTC/USDT")
        
        assert ticker == mock_ticker
        exchange_manager.async_exchange.fetch_ticker.assert_called_once_with("BTC/USDT")
    
    @pytest.mark.asyncio
    async def test_get_balance(self, exchange_manager):
        """Test getting account balance."""
        mock_balance = {
            'BTC': {'free': 1.0, 'used': 0.0, 'total': 1.0},
            'USDT': {'free': 50000.0, 'used': 0.0, 'total': 50000.0}
        }
        exchange_manager.async_exchange.fetch_balance = AsyncMock(return_value=mock_balance)
        
        balance = await exchange_manager.get_balance()
        
        assert balance == mock_balance
        exchange_manager.async_exchange.fetch_balance.assert_called_once()


class TestTradeSignal:
    """Test cases for TradeSignal dataclass."""
    
    def test_trade_signal_creation(self):
        """Test creating a trade signal."""
        signal = TradeSignal(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.001,
            order_type=OrderType.LIMIT,
            price=50000.0,
            strategy_name="test_strategy"
        )
        
        assert signal.symbol == "BTC/USDT"
        assert signal.side == OrderSide.BUY
        assert signal.quantity == 0.001
        assert signal.order_type == OrderType.LIMIT
        assert signal.price == 50000.0
        assert signal.strategy_name == "test_strategy"
        assert signal.time_in_force == "GTC"
        assert isinstance(signal.timestamp, datetime)


class TestOrder:
    """Test cases for Order dataclass."""
    
    def test_order_creation(self):
        """Test creating an order."""
        order = Order(
            id="test_order_123",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0,
            status=OrderStatus.PENDING
        )
        
        assert order.id == "test_order_123"
        assert order.symbol == "BTC/USDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == 0.001
        assert order.price == 50000.0
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0.0
        assert order.remaining_quantity == 0.001
        assert order.average_price is None
        assert order.commission == 0.0
        assert isinstance(order.timestamp, datetime)
    
    def test_order_post_init(self):
        """Test order post-initialization."""
        order = Order(
            id="test_order_123",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0,
            status=OrderStatus.PENDING,
            remaining_quantity=0.0  # Should be overridden
        )
        
        assert order.remaining_quantity == 0.001  # Should be set to quantity


if __name__ == "__main__":
    pytest.main([__file__]) 