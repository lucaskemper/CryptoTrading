# Manager Integration Guide

This guide explains how to properly integrate the Position Manager, Order Manager, and Risk Manager to create a robust, automated trading system with real-time risk management and position tracking.

## Overview

The three core managers work together to provide a complete trade lifecycle management system:

- **Order Manager**: Handles order submission, execution, and tracking
- **Position Manager**: Tracks positions, calculates PnL, and manages portfolio metrics
- **Risk Manager**: Monitors risk limits, enforces safety rules, and triggers forced exits

## Integration Architecture

### 1. Manager Dependencies

Each manager has references to the other managers for seamless communication:

```python
# Initialize managers
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
```

### 2. Trade Lifecycle Flow

#### A. Pre-Trade: Risk Check Before Order Submission

```python
# Order Manager → Risk Manager
is_allowed, reason = risk_manager.check_order_risk(signal)
if not is_allowed:
    raise ValueError(f"Order rejected by risk manager: {reason}")
```

**Risk checks include:**
- Position size limits
- Exposure limits
- Drawdown limits
- Correlation limits
- Circuit breaker status

#### B. Order Execution and Position Updates

```python
# Order Manager → Position Manager (after order fill)
if order.status == OrderStatus.FILLED:
    await self._handle_filled_order(order)

# Position Manager → Risk Manager
risk_manager.update_on_trade(symbol, side, quantity, price, order_id)
```

**Position updates include:**
- Creating new positions
- Updating existing positions
- Closing positions
- Partial fills handling

#### C. Ongoing Risk Monitoring

```python
# Continuous monitoring loop
async def _risk_monitoring_loop(self):
    while self.is_running:
        # Check stop-losses and take-profits
        await self.risk_manager.execute_forced_exits()
        
        # Update portfolio metrics
        portfolio_metrics = self.position_manager.get_portfolio_metrics()
        
        # Check risk levels
        risk_level = self.risk_manager.get_risk_level()
        
        await asyncio.sleep(30)  # Check every 30 seconds
```

#### D. Forced Exits and Circuit Breakers

```python
# Risk Manager → Order Manager
async def execute_forced_exits(self):
    # Check stop-losses
    stop_loss_positions = self.check_stop_losses()
    for position_id, side, quantity in stop_loss_positions:
        await self._submit_exit_order(position_id, side, quantity, "stop_loss")
    
    # Check circuit breakers
    if not self._check_circuit_breakers():
        await self._execute_circuit_breaker()
```

## Key Integration Points

### 1. Order Manager Integration

**Enhanced `submit_order` method:**
```python
async def submit_order(self, signal: TradeSignal) -> Order:
    # Pre-trade risk check
    if self.risk_manager:
        is_allowed, reason = self.risk_manager.check_order_risk(signal)
        if not is_allowed:
            raise ValueError(f"Order rejected by risk manager: {reason}")
    
    # Submit order to exchange
    order = await self._submit_to_exchange(order, signal)
    
    # Handle completion
    await self._handle_completed_order(order)
    
    return order
```

**Order completion handling:**
```python
async def _handle_filled_order(self, order: Order):
    # Convert order side to position side
    position_side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT
    
    if order.side == OrderSide.BUY:
        # Opening or adding to long position
        position = self.position_manager.add_position(...)
    else:  # SELL
        # Close existing long position or open short
        if existing_positions:
            realized_pnl = self.position_manager.close_position(...)
        else:
            position = self.position_manager.add_position(...)
    
    # Notify risk manager
    if self.risk_manager:
        self.risk_manager.update_on_trade(...)
```

### 2. Position Manager Integration

**Trade update method:**
```python
def update_on_trade(self, symbol: str, side: str, quantity: float, price: float, order_id: str):
    # Convert side string to PositionSide enum
    position_side = PositionSide.LONG if side == 'buy' else PositionSide.SHORT
    
    if side == 'buy':
        # Opening or adding to long position
        self.add_position(symbol, position_side, quantity, price, order_id)
    else:  # sell
        # Close existing long position or open short
        if existing_positions:
            realized_pnl = self.close_position(position_to_close.id, price, quantity, order_id)
        else:
            self.add_position(symbol, position_side, quantity, price, order_id)
    
    # Notify risk manager
    if self.risk_manager:
        self.risk_manager.update_on_trade(symbol, side, quantity, price, order_id)
```

**Position updates with risk monitoring:**
```python
def update_position(self, position_id: str, new_price: float, order_id: Optional[str] = None):
    # Update position price and PnL
    position.update_price(new_price)
    
    # Check stop-loss and take-profit
    self._check_exit_conditions(position)
    
    # Notify risk manager
    if self.risk_manager:
        self.risk_manager.update_position(position.symbol, position.side.value, 
                                       position.quantity, new_price)
```

### 3. Risk Manager Integration

**Trade update handling:**
```python
def update_on_trade(self, symbol: str, side: str, quantity: float, price: float, order_id: str):
    # Update position tracking
    self.update_position(symbol, side, quantity, price)
    
    # Check for risk events after trade
    self._check_post_trade_risk_events(symbol, side, quantity, price)
    
    # Update portfolio metrics
    if self.position_manager:
        portfolio_metrics = self.position_manager.get_portfolio_metrics()
        self.update_portfolio_metrics(portfolio_metrics.total_market_value, 
                                   portfolio_metrics.daily_pnl)
```

**Forced exit execution:**
```python
async def execute_forced_exits(self):
    # Check stop-losses
    stop_loss_positions = self.check_stop_losses()
    for position_id, side, quantity in stop_loss_positions:
        await self._submit_exit_order(position_id, side, quantity, "stop_loss")
    
    # Check take-profits
    take_profit_positions = self.check_take_profits()
    for position_id, side, quantity in take_profit_positions:
        await self._submit_exit_order(position_id, side, quantity, "take_profit")
    
    # Check circuit breakers
    if not self._check_circuit_breakers():
        await self._execute_circuit_breaker()
```

## Event-Driven Architecture

### 1. Callback Methods

Each manager implements callback methods that are called by other managers:

**Order Manager callbacks:**
- `_handle_filled_order()`: Called when order is filled
- `_handle_partial_fill()`: Called when order is partially filled

**Position Manager callbacks:**
- `update_on_trade()`: Called when trade occurs
- `_check_exit_conditions()`: Called when position is updated

**Risk Manager callbacks:**
- `update_on_trade()`: Called when trade occurs
- `execute_forced_exits()`: Called periodically for risk monitoring

### 2. Event Flow

```
Strategy → Order Manager → Risk Manager (pre-check)
                ↓
        Exchange (order execution)
                ↓
        Order Manager → Position Manager (position update)
                ↓
        Position Manager → Risk Manager (risk update)
                ↓
        Risk Manager → Order Manager (forced exits if needed)
```

## Risk Management Integration

### 1. Pre-Trade Risk Checks

**Position size limits:**
```python
def _check_position_size_limits(self, signal) -> bool:
    current_size = self._get_position_size(signal.symbol)
    new_size = current_size + (signal.quantity * signal.price)
    max_size = self.config.get('MAX_POSITION_SIZE', 1000000)
    return new_size <= max_size
```

**Exposure limits:**
```python
def _check_exposure_limits(self, signal) -> bool:
    current_exposure = self._get_total_exposure()
    trade_value = signal.quantity * signal.price
    max_exposure = self.config.get('MAX_TOTAL_EXPOSURE', 5000000)
    return (current_exposure + trade_value) <= max_exposure
```

### 2. Post-Trade Risk Monitoring

**Stop-loss monitoring:**
```python
def check_stop_losses(self) -> List[Tuple[str, str, float]]:
    positions_to_close = []
    for position in open_positions:
        if position.stop_loss is None:
            continue
        
        if position.side == PositionSide.LONG and position.current_price <= position.stop_loss:
            positions_to_close.append((position.id, 'sell', position.quantity))
        elif position.side == PositionSide.SHORT and position.current_price >= position.stop_loss:
            positions_to_close.append((position.id, 'buy', position.quantity))
    
    return positions_to_close
```

**Circuit breaker checks:**
```python
def _check_circuit_breakers(self) -> bool:
    portfolio_metrics = self.position_manager.get_portfolio_metrics()
    
    # Check drawdown limit
    if portfolio_metrics.max_drawdown_percentage > 20:  # 20% drawdown
        return False
    
    # Check daily loss limit
    if portfolio_metrics.daily_pnl_percentage < -10:  # 10% daily loss
        return False
    
    return True
```

## Best Practices

### 1. Thread Safety

All managers use thread-safe operations:

```python
# Position Manager uses locks
def update_position(self, position_id: str, new_price: float):
    with self._price_update_lock:
        return self._update_position_unsafe(position_id, new_price)

# Risk Manager uses async operations
async def execute_forced_exits(self):
    # Async operations for order submission
    await self._submit_exit_order(position_id, side, quantity, reason)
```

### 2. Error Handling

Comprehensive error handling throughout:

```python
try:
    # Manager operations
    await self.order_manager.submit_order(signal)
except Exception as e:
    logger.error(f"Failed to submit order: {e}")
    # Handle error appropriately
```

### 3. Logging and Audit Trail

All critical operations are logged:

```python
logger.info(f"Order submitted: {order.id} - {signal.side.value} {signal.quantity} {signal.symbol}")
logger.warning(f"Order rejected by risk manager: {reason}")
logger.error(f"Failed to handle filled order {order.id}: {e}")
```

### 4. Configuration Management

Risk limits are configurable:

```python
# In config.yaml
RISK_MANAGEMENT:
  MAX_POSITION_SIZE: 1000000  # $1M per position
  MAX_TOTAL_EXPOSURE: 5000000  # $5M total exposure
  MAX_DRAWDOWN: 20  # 20% max drawdown
  DAILY_LOSS_LIMIT: 10  # 10% daily loss limit
```

## Example Usage

### Complete Trading System

```python
class TradingSystem:
    def __init__(self):
        # Initialize managers
        self.position_manager = PositionManager()
        self.order_manager = OrderManager("binance")
        self.risk_manager = RiskManager()
        
        # Wire managers together
        self._connect_managers()
    
    def _connect_managers(self):
        # Order Manager connections
        self.order_manager.set_risk_manager(self.risk_manager)
        self.order_manager.set_position_manager(self.position_manager)
        
        # Position Manager connections
        self.position_manager.set_order_manager(self.order_manager)
        self.position_manager.set_risk_manager(self.risk_manager)
        
        # Risk Manager connections
        self.risk_manager.set_position_manager(self.position_manager)
        self.risk_manager.set_order_manager(self.order_manager)
    
    async def submit_trade_signal(self, signal: TradeSignal) -> bool:
        try:
            # Pre-trade risk check
            if self.risk_manager:
                is_allowed, reason = self.risk_manager.check_order_risk(signal)
                if not is_allowed:
                    logger.warning(f"Trade signal rejected by risk manager: {reason}")
                    return False
            
            # Submit order
            order = await self.order_manager.submit_order(signal)
            logger.info(f"Order submitted successfully: {order.id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit trade signal: {e}")
            return False
    
    async def _risk_monitoring_loop(self):
        while self.is_running:
            try:
                # Check for forced exits
                await self.risk_manager.execute_forced_exits()
                
                # Update portfolio metrics
                portfolio_metrics = self.position_manager.get_portfolio_metrics()
                logger.info(f"Portfolio PnL: ${portfolio_metrics.total_pnl:.2f}")
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(30)
```

## Testing Integration

### Unit Tests

Test each manager's integration points:

```python
def test_order_manager_integration():
    order_manager = OrderManager()
    risk_manager = RiskManager()
    position_manager = PositionManager()
    
    # Wire managers
    order_manager.set_risk_manager(risk_manager)
    order_manager.set_position_manager(position_manager)
    
    # Test order submission with risk check
    signal = TradeSignal(symbol="BTC/USDT", side=OrderSide.BUY, quantity=0.001)
    order = await order_manager.submit_order(signal)
    
    assert order.status == OrderStatus.FILLED
    assert len(position_manager.get_open_positions()) == 1
```

### Integration Tests

Test the complete trade lifecycle:

```python
async def test_complete_trade_lifecycle():
    trading_system = TradingSystem()
    await trading_system.start()
    
    # Submit trade signal
    signal = TradeSignal(symbol="BTC/USDT", side=OrderSide.BUY, quantity=0.001)
    success = await trading_system.submit_trade_signal(signal)
    
    assert success == True
    
    # Check position was created
    positions = trading_system.position_manager.get_open_positions()
    assert len(positions) == 1
    
    # Check risk manager was updated
    risk_events = trading_system.risk_manager.get_risk_events(limit=1)
    assert len(risk_events) > 0
    
    await trading_system.stop()
```

## Conclusion

By properly integrating the Position Manager, Order Manager, and Risk Manager, you create a robust, automated trading system that:

1. **Enforces risk limits** before every trade
2. **Tracks positions accurately** with real-time PnL
3. **Monitors risk continuously** and triggers forced exits when needed
4. **Provides comprehensive audit trails** for all operations
5. **Handles errors gracefully** with proper logging and recovery

This integration ensures that every trade is risk-checked, every position is tracked, and risk events are handled automatically and transparently—making your system production-ready and auditable. 