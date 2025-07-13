import asyncio
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import ccxt
import ccxt.async_support as ccxt_async

from src.utils.logger import logger
from src.utils.config_loader import config
from src.execution.position_manager import PositionSide, PositionStatus


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class TradeSignal:
    """Trade signal from strategy."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    strategy_name: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __getitem__(self, key):
        """Enable dict-like access for compatibility with risk manager."""
        return getattr(self, key)
    
    def get(self, key, default=None):
        """Enable dict-like get method for compatibility with risk manager."""
        return getattr(self, key, default)
    
    def __contains__(self, key):
        """Enable 'in' operator for compatibility with risk manager."""
        return hasattr(self, key)


@dataclass
class Order:
    """Order representation."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    status: OrderStatus
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_price: Optional[float] = None
    commission: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    exchange_order_id: Optional[str] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.quantity


class ExchangeManager:
    """Manages exchange connections and API interactions."""
    
    def __init__(self, exchange_name: str = "binance"):
        self.exchange_name = exchange_name
        self.exchange = None
        self.async_exchange = None
        self._setup_exchange()
    
    def _setup_exchange(self):
        """Setup exchange connection with API credentials."""
        try:
            exchange_config = config.get_exchange_config(self.exchange_name)
            
            # Setup synchronous exchange
            exchange_class = getattr(ccxt, self.exchange_name)
            self.exchange = exchange_class({
                'apiKey': exchange_config['apiKey'],
                'secret': exchange_config['secret'],
                'sandbox': exchange_config['sandbox'],
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })
            
            # Setup asynchronous exchange
            async_exchange_class = getattr(ccxt_async, self.exchange_name)
            self.async_exchange = async_exchange_class({
                'apiKey': exchange_config['apiKey'],
                'secret': exchange_config['secret'],
                'sandbox': exchange_config['sandbox'],
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })
            
            logger.info(f"Exchange {self.exchange_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange {self.exchange_name}: {e}")
            raise
    
    async def load_markets(self):
        """Load market information."""
        try:
            await self.async_exchange.load_markets()
            logger.info(f"Markets loaded for {self.exchange_name}")
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            raise
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker information."""
        try:
            ticker = await self.async_exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            raise
    
    async def get_balance(self) -> Dict[str, Any]:
        """Get account balance."""
        try:
            balance = await self.async_exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            raise


class OrderManager:
    """Core order management system for crypto trading bot."""
    
    def __init__(self, exchange_name: str = "binance"):
        self.exchange_manager = ExchangeManager(exchange_name)
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.max_retries = config.get("ORDER_MAX_RETRIES", 3)
        self.retry_delay = config.get("ORDER_RETRY_DELAY", 1.0)
        self.max_slippage = config.get("MAX_SLIPPAGE", 0.02)  # 2%
        self.order_timeout = config.get("ORDER_TIMEOUT", 300)  # 5 minutes
        
        # Risk management integration
        self.risk_manager = None  # Will be set by main application
        self.position_manager = None  # Will be set by main application
        
        logger.info("OrderManager initialized successfully")
    
    def set_risk_manager(self, risk_manager):
        """Set risk manager reference."""
        self.risk_manager = risk_manager
        logger.info("Risk manager connected to order manager")
    
    def set_position_manager(self, position_manager):
        """Set position manager reference."""
        self.position_manager = position_manager
        logger.info("Position manager connected to order manager")
    
    async def submit_order(self, signal: TradeSignal) -> Order:
        """
        Submit a new order based on trade signal.
        
        Args:
            signal: Trade signal from strategy
            
        Returns:
            Order object with initial status
        """
        # Validate signal
        if not self._validate_signal(signal):
            raise ValueError(f"Invalid trade signal: {signal}")
        
        # Pre-trade risk check
        if self.risk_manager:
            is_allowed, reason = self.risk_manager.check_order_risk(signal)
            if not is_allowed:
                logger.warning(f"Order rejected by risk manager: {reason}")
                raise ValueError(f"Order rejected by risk manager: {reason}")
        
        # Create order object
        order = Order(
            id=str(uuid.uuid4()),
            symbol=signal.symbol,
            side=signal.side,
            order_type=signal.order_type,
            quantity=signal.quantity,
            price=signal.price,
            status=OrderStatus.PENDING,
            timestamp=datetime.now()
        )
        
        # Submit to exchange
        try:
            await self._submit_to_exchange(order, signal)
            self.active_orders[order.id] = order
            logger.info(f"Order submitted: {order.id} - {signal.side.value} {signal.quantity} {signal.symbol}")
            return order
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            self.order_history.append(order)
            logger.error(f"Order submission failed: {order.id} - {e}")
            raise
    
    async def _submit_to_exchange(self, order: Order, signal: TradeSignal):
        """Submit order to exchange with retry logic."""
        for attempt in range(self.max_retries):
            try:
                # Prepare order parameters
                order_params = self._prepare_order_params(order, signal)
                
                # Submit order
                if signal.order_type == OrderType.MARKET:
                    result = await self.exchange_manager.async_exchange.create_market_order(
                        symbol=signal.symbol,
                        side=signal.side.value,
                        amount=signal.quantity,
                        params=order_params
                    )
                elif signal.order_type == OrderType.LIMIT:
                    result = await self.exchange_manager.async_exchange.create_limit_order(
                        symbol=signal.symbol,
                        side=signal.side.value,
                        amount=signal.quantity,
                        price=signal.price,
                        params=order_params
                    )
                elif signal.order_type == OrderType.STOP:
                    result = await self.exchange_manager.async_exchange.create_order(
                        symbol=signal.symbol,
                        type='stop',
                        side=signal.side.value,
                        amount=signal.quantity,
                        price=signal.stop_price,
                        params=order_params
                    )
                else:
                    raise ValueError(f"Unsupported order type: {signal.order_type}")
                
                # Update order with exchange response
                order.exchange_order_id = result.get('id')
                order.status = OrderStatus.PENDING
                logger.info(f"Order submitted to exchange: {order.exchange_order_id}")
                return
                
            except Exception as e:
                logger.warning(f"Order submission attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
    
    def _prepare_order_params(self, order: Order, signal: TradeSignal) -> Dict[str, Any]:
        """Prepare order parameters for exchange submission."""
        params = {}
        
        # Add time in force for limit orders
        if signal.order_type == OrderType.LIMIT:
            params['timeInForce'] = signal.time_in_force
        
        # Add slippage protection for market orders
        if signal.order_type == OrderType.MARKET:
            params['maxSlippage'] = self.max_slippage
        
        return params
    
    def _validate_signal(self, signal: TradeSignal) -> bool:
        """Validate trade signal parameters."""
        if not signal.symbol or not signal.quantity or signal.quantity <= 0:
            return False
        
        if signal.order_type == OrderType.LIMIT and not signal.price:
            return False
        
        if signal.order_type == OrderType.STOP and not signal.stop_price:
            return False
        
        return True
    
    async def track_order(self, order_id: str) -> Order:
        """
        Track order status and update internal state.
        
        Args:
            order_id: Internal order ID
            
        Returns:
            Updated order object
        """
        if order_id not in self.active_orders:
            raise ValueError(f"Order {order_id} not found in active orders")
        
        order = self.active_orders[order_id]
        
        try:
            # Get order status from exchange
            exchange_order = await self.exchange_manager.async_exchange.fetch_order(
                order.exchange_order_id,
                order.symbol
            )
            
            # Update order status
            self._update_order_status(order, exchange_order)
            
            # Handle completed orders
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                await self._handle_completed_order(order)
            
            logger.debug(f"Order {order_id} status: {order.status.value}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to track order {order_id}: {e}")
            raise
    
    def _update_order_status(self, order: Order, exchange_order: Dict[str, Any]):
        """Update order status based on exchange response."""
        exchange_status = exchange_order.get('status', 'unknown')
        
        # Map exchange status to internal status
        status_mapping = {
            'open': OrderStatus.PENDING,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'rejected': OrderStatus.REJECTED,
            'expired': OrderStatus.EXPIRED
        }
        
        order.status = status_mapping.get(exchange_status, OrderStatus.PENDING)
        order.filled_quantity = exchange_order.get('filled', 0.0)
        order.remaining_quantity = exchange_order.get('remaining', order.quantity)
        order.average_price = exchange_order.get('average')
        order.commission = exchange_order.get('fee', {}).get('cost', 0.0)
    
    async def _handle_completed_order(self, order: Order):
        """Handle completed order (filled, cancelled, rejected)."""
        # Remove from active orders
        if order.id in self.active_orders:
            del self.active_orders[order.id]
        
        # Add to history
        self.order_history.append(order)
        
        # Update position manager and risk manager for filled orders
        if order.status == OrderStatus.FILLED:
            await self._handle_filled_order(order)
        elif order.status == OrderStatus.PARTIALLY_FILLED:
            await self._handle_partial_fill(order)
        
        # Log completion
        if order.status == OrderStatus.FILLED:
            logger.info(f"Order filled: {order.id} - {order.filled_quantity} @ {order.average_price}")
        elif order.status == OrderStatus.CANCELLED:
            logger.info(f"Order cancelled: {order.id}")
        elif order.status == OrderStatus.REJECTED:
            logger.error(f"Order rejected: {order.id} - {order.error_message}")
    
    async def _handle_filled_order(self, order: Order):
        """Handle fully filled order."""
        if not self.position_manager:
            logger.warning("Position manager not connected")
            return
        
        try:
            # Convert order side to position side
            position_side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT
            
            # Add or update position
            if order.side == OrderSide.BUY:
                # Opening or adding to long position
                position = self.position_manager.add_position(
                    symbol=order.symbol,
                    side=position_side,
                    quantity=order.filled_quantity,
                    entry_price=order.average_price,
                    order_id=order.id,
                    strategy_name=getattr(order, 'strategy_name', 'unknown')
                )
            else:  # SELL
                # Check if we have an existing position to close
                existing_positions = self.position_manager.get_positions_by_symbol(order.symbol)
                long_positions = [p for p in existing_positions if p.side == PositionSide.LONG and p.status != PositionStatus.CLOSED]
                
                if long_positions:
                    # Close existing long position
                    position_to_close = long_positions[0]  # Close the first one
                    realized_pnl = self.position_manager.close_position(
                        position_to_close.id,
                        order.average_price,
                        order.filled_quantity,
                        order.id
                    )
                    logger.info(f"Closed position {position_to_close.id}: PnL {realized_pnl:.2f}")
                else:
                    # Opening short position
                    position = self.position_manager.add_position(
                        symbol=order.symbol,
                        side=position_side,
                        quantity=order.filled_quantity,
                        entry_price=order.average_price,
                        order_id=order.id,
                        strategy_name=getattr(order, 'strategy_name', 'unknown')
                    )
            
            # Notify risk manager
            if self.risk_manager:
                self.risk_manager.update_on_trade(
                    symbol=order.symbol,
                    side=order.side.value,
                    quantity=order.filled_quantity,
                    price=order.average_price,
                    order_id=order.id
                )
                
        except Exception as e:
            logger.error(f"Failed to handle filled order {order.id}: {e}")
    
    async def _handle_partial_fill(self, order: Order):
        """Handle partially filled order."""
        if not self.position_manager:
            logger.warning("Position manager not connected")
            return
        
        try:
            # Calculate the fill amount for this update
            previous_filled = sum(o.filled_quantity for o in self.order_history if o.id == order.id)
            current_fill = order.filled_quantity - previous_filled
            
            if current_fill <= 0:
                return
            
            # Convert order side to position side
            position_side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT
            
            # Handle partial fill similar to full fill
            if order.side == OrderSide.BUY:
                # Adding to long position
                position = self.position_manager.add_position(
                    symbol=order.symbol,
                    side=position_side,
                    quantity=current_fill,
                    entry_price=order.average_price,
                    order_id=order.id,
                    strategy_name=getattr(order, 'strategy_name', 'unknown')
                )
            else:  # SELL
                # Partial close of existing position
                existing_positions = self.position_manager.get_positions_by_symbol(order.symbol)
                long_positions = [p for p in existing_positions if p.side == PositionSide.LONG and p.status != PositionStatus.CLOSED]
                
                if long_positions:
                    position_to_close = long_positions[0]
                    realized_pnl = self.position_manager.close_position(
                        position_to_close.id,
                        order.average_price,
                        current_fill,
                        order.id
                    )
                    logger.info(f"Partially closed position {position_to_close.id}: PnL {realized_pnl:.2f}")
            
            # Notify risk manager
            if self.risk_manager:
                self.risk_manager.update_on_trade(
                    symbol=order.symbol,
                    side=order.side.value,
                    quantity=current_fill,
                    price=order.average_price,
                    order_id=order.id
                )
                
        except Exception as e:
            logger.error(f"Failed to handle partial fill for order {order.id}: {e}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.
        
        Args:
            order_id: Internal order ID
            
        Returns:
            True if cancellation was successful
        """
        if order_id not in self.active_orders:
            raise ValueError(f"Order {order_id} not found in active orders")
        
        order = self.active_orders[order_id]
        
        try:
            # Cancel on exchange
            await self.exchange_manager.async_exchange.cancel_order(
                order.exchange_order_id,
                order.symbol
            )
            
            # Update order status
            order.status = OrderStatus.CANCELLED
            await self._handle_completed_order(order)
            
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def replace_order(self, order_id: str, new_price: float, new_quantity: Optional[float] = None) -> Order:
        """
        Replace an existing order with new parameters.
        
        Args:
            order_id: Internal order ID
            new_price: New price for the order
            new_quantity: New quantity (optional)
            
        Returns:
            New order object
        """
        if order_id not in self.active_orders:
            raise ValueError(f"Order {order_id} not found in active orders")
        
        old_order = self.active_orders[order_id]
        
        # Cancel old order
        await self.cancel_order(order_id)
        
        # Create new signal
        new_signal = TradeSignal(
            symbol=old_order.symbol,
            side=old_order.side,
            quantity=new_quantity or old_order.remaining_quantity,
            order_type=old_order.order_type,
            price=new_price,
            strategy_name="order_replacement"
        )
        
        # Submit new order
        return await self.submit_order(new_signal)
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current order status."""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            await self.track_order(order_id)
            return order.status
        else:
            # Check history
            for order in self.order_history:
                if order.id == order_id:
                    return order.status
            raise ValueError(f"Order {order_id} not found")
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        return list(self.active_orders.values())
    
    def get_order_history(self, limit: int = 100) -> List[Order]:
        """Get recent order history."""
        return self.order_history[-limit:]
    
    async def cleanup_expired_orders(self):
        """Clean up orders that have exceeded timeout."""
        current_time = datetime.now()
        expired_orders = []
        
        for order_id, order in self.active_orders.items():
            if (current_time - order.timestamp).total_seconds() > self.order_timeout:
                expired_orders.append(order_id)
        
        for order_id in expired_orders:
            logger.warning(f"Cleaning up expired order: {order_id}")
            await self.cancel_order(order_id)
    
    async def batch_submit_orders(self, signals: List[TradeSignal]) -> List[Order]:
        """
        Submit multiple orders in batch.
        
        Args:
            signals: List of trade signals
            
        Returns:
            List of submitted orders
        """
        orders = []
        for signal in signals:
            try:
                order = await self.submit_order(signal)
                orders.append(order)
            except Exception as e:
                logger.error(f"Failed to submit order in batch: {e}")
        
        logger.info(f"Batch submitted {len(orders)} orders")
        return orders
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get order manager performance metrics."""
        total_orders = len(self.order_history)
        filled_orders = len([o for o in self.order_history if o.status == OrderStatus.FILLED])
        cancelled_orders = len([o for o in self.order_history if o.status == OrderStatus.CANCELLED])
        rejected_orders = len([o for o in self.order_history if o.status == OrderStatus.REJECTED])
        
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'cancelled_orders': cancelled_orders,
            'rejected_orders': rejected_orders,
            'success_rate': filled_orders / total_orders if total_orders > 0 else 0,
            'active_orders': len(self.active_orders)
        }
    
    async def shutdown(self):
        """Clean shutdown of order manager."""
        logger.info("Shutting down order manager...")
        
        # Cancel all active orders
        for order_id in list(self.active_orders.keys()):
            try:
                await self.cancel_order(order_id)
            except Exception as e:
                logger.error(f"Failed to cancel order during shutdown: {e}")
        
        # Close exchange connection
        if self.exchange_manager.async_exchange:
            await self.exchange_manager.async_exchange.close()
        
        logger.info("Order manager shutdown complete")


# Example usage and testing
async def main():
    """Example usage of OrderManager."""
    order_manager = OrderManager("binance")
    
    # Create a test signal
    signal = TradeSignal(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.001,
        order_type=OrderType.LIMIT,
        price=50000.0,
        strategy_name="test_strategy"
    )
    
    try:
        # Submit order
        order = await order_manager.submit_order(signal)
        print(f"Order submitted: {order.id}")
        
        # Track order
        await order_manager.track_order(order.id)
        print(f"Order status: {order.status.value}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await order_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
