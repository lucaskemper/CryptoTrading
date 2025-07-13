# Position Manager Documentation

## What’s New / Enhancements

The Position Manager now includes:
- **Closed Position Analytics**: Accurate win rate and performance tracking
- **Enhanced Sector Mapping**: Comprehensive crypto sector classification
- **Advanced Export Capabilities**: CSV, JSON, and DataFrame exports
- **Historical Performance Analysis**: Full backtesting and research capabilities
- **Thread Safety**: Concurrency support for multi-threaded environments
- **Real-Time Price Updates**: Live price feed integration
- **Database Statistics & Cleanup**: Performance monitoring and smart data retention

---

## Overview

The Position Manager is a comprehensive system for tracking and managing trading positions in the crypto trading bot. It provides real-time position tracking, portfolio analytics, risk management integration, automated stop-loss/take-profit functionality, and advanced research/export features.

## Key Features

### 1. Real-Time Position Tracking
- Symbol, side, quantity, entry price, and entry time tracking
- Current price and market value calculations
- Associated stop-loss and take-profit levels
- Realized and unrealized PnL calculations
- Support for multi-leg and multi-asset positions

### 2. Position Lifecycle Management
- Open new positions when orders are filled
- Update positions on partial fills, scaling, or additional trades
- Close positions when exit orders are filled
- Handle partial closes and adjust quantities accordingly
- Automatic position status updates

### 3. Portfolio State and Exposure
- Aggregate portfolio metrics (total exposure, market value, leverage)
- Current unrealized and realized PnL
- Exposure by asset and sector (with enhanced sector mapping)
- Performance metrics (win rate, drawdown, Sharpe ratio)

### 4. Integration with Execution and Risk
- Receive execution updates from order manager
- Notify risk manager of position changes
- Support stop-loss and take-profit triggers
- Real-time risk monitoring

### 5. Analytics and Reporting
- Performance metrics (per-position and portfolio-level)
- Win/loss rates, average holding time, turnover
- Audit trails with entry/exit times and order references
- Backtesting support with position history replay
- **Closed Position Analytics**: Win rate, realized PnL, and more
- **Historical Performance Analysis**: Full backtesting and research

### 6. Data Access and API
- Query current and historical positions
- Filter by asset, status, or time
- Dashboard hooks for portfolio visualization
- Serialization (JSON, CSV, DataFrame, database) for persistence

### 7. Error Handling and Robustness
- Detect and resolve inconsistencies
- Log all position changes and errors
- Graceful handling of exchange downtime
- State reconciliation on recovery

### 8. Thread Safety and Concurrency
- RLock for position operations
- Separate lock for price updates
- Safe multi-threaded position management

### 9. Real-Time Price Updates
- Batch and individual price updates
- Price cache and rolling price history
- Market data integration
- Continuous price update loop

### 10. Database Statistics and Cleanup
- **Get stats:**
```python
stats = position_manager.get_database_stats()
print(f"Total positions: {stats['total_positions']}")
```
- **Cleanup:**
```python
position_manager.cleanup_old_data(days=90)
```

---

## Architecture

### Core Components

#### Position Class
```python
@dataclass
class Position:
    id: str
    symbol: str
    side: PositionSide  # LONG or SHORT
    position_type: PositionType  # SINGLE, PAIR, or MULTI_LEG
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    last_update_time: datetime
    status: PositionStatus  # OPEN, CLOSED, PARTIALLY_CLOSED
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    commission: float
    strategy_name: str
    order_ids: List[str]
    related_positions: List[str]  # For pair/multi-leg positions
    metadata: Dict[str, Any]
```

#### PortfolioMetrics Class
```python
@dataclass
class PortfolioMetrics:
    total_market_value: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_pnl: float
    total_pnl_percentage: float
    daily_pnl: float
    daily_pnl_percentage: float
    exposure_by_asset: Dict[str, float]
    exposure_by_sector: Dict[str, float]
    position_count: int
    open_position_count: int
    win_rate: float
    average_holding_time: timedelta
    max_drawdown: float
    max_drawdown_percentage: float
    sharpe_ratio: float
```

### Database Schema

The position manager uses SQLite for persistence with three main tables:
1. **positions** - Current and historical positions
2. **position_history** - Detailed action log for each position
3. **portfolio_snapshots** - Periodic portfolio state snapshots

---

## Enhancements in Detail

### Closed Position Analytics
- **Method:** `_get_closed_positions_recent(days: int = 30)`
- **Features:**
  - Configurable time window
  - Accurate win rate calculation
  - Realized PnL tracking
  - Performance analytics
- **Example:**
```python
closed_positions = position_manager._get_closed_positions_recent(days=30)
win_rate = sum(1 for p in closed_positions if p.total_pnl > 0) / len(closed_positions)
```

### Enhanced Sector Mapping
- **Comprehensive mapping** for Layer 1, Layer 2, DeFi, Gaming, NFT, Stablecoins, Exchange, Privacy, AI, Oracles, Meme coins, and more.
- **Example:**
```python
metrics = position_manager.get_portfolio_metrics()
for sector, exposure in metrics.exposure_by_sector.items():
    print(f"{sector}: ${exposure:,.2f}")
```

### Advanced Export Capabilities
- **Formats:** JSON, CSV, pandas DataFrame
- **Include closed positions:** `include_closed=True`
- **Example:**
```python
json_data = position_manager.export_positions(format="json", include_closed=True)
csv_data = position_manager.export_positions(format="csv", include_closed=True)
df = position_manager.export_positions(format="dataframe", include_closed=True)
```

### Historical Performance Analysis
- **Position history with filtering:**
```python
history = position_manager.get_position_history(
    symbol="BTC/USDT",
    strategy="stat_arb",
    start_date=datetime.now() - timedelta(days=30),
    status="closed"
)
```
- **Comprehensive analytics:**
```python
analytics = position_manager.get_position_analytics(
    symbol="BTC/USDT",
    strategy="stat_arb",
    days=30
)
```

### Thread Safety and Concurrency
- **Thread-safe methods:**
```python
position_manager.add_position(...)
position_manager.update_position(...)
position_manager.close_position(...)
```
- **Features:** RLock, price update lock, atomic DB ops, error handling

### Real-Time Price Updates
- **Batch update:**
```python
await position_manager.update_all_prices()
```
- **From feed:**
```python
position_manager.update_price_from_feed("BTC/USDT", 50000.0)
```
- **Continuous loop:**
```python
position_manager.start_price_update_loop(update_interval=30)
```

### Database Statistics and Cleanup
- **Get stats:**
```python
stats = position_manager.get_database_stats()
print(f"Total positions: {stats['total_positions']}")
```
- **Cleanup:**
```python
position_manager.cleanup_old_data(days=90)
```

---

## Usage Examples

### Basic Position Management
```python
from execution.position_manager import PositionManager, PositionSide, PositionType

position_manager = PositionManager()
position = position_manager.add_position(
    symbol="BTC/USDT",
    side=PositionSide.LONG,
    quantity=0.1,
    entry_price=50000.0,
    order_id="order_123",
    strategy_name="trend_following",
    stop_loss=48000.0,
    take_profit=52000.0
)
position_manager.update_position(position.id, 51000.0)
realized_pnl = position_manager.close_position(position.id, 51500.0)
```

### Multi-Leg Position (Arbitrage)
```python
long_position = position_manager.add_position(
    symbol="BTC/USDT",
    side=PositionSide.LONG,
    quantity=0.1,
    entry_price=50000.0,
    order_id="order_001",
    strategy_name="statistical_arbitrage",
    position_type=PositionType.PAIR,
    metadata={"exchange": "binance", "pair_id": "arb_001"}
)
short_position = position_manager.add_position(
    symbol="BTC/USDT",
    side=PositionSide.SHORT,
    quantity=0.1,
    entry_price=50100.0,
    order_id="order_002",
    strategy_name="statistical_arbitrage",
    position_type=PositionType.PAIR,
    related_positions=[long_position.id],
    metadata={"exchange": "kraken", "pair_id": "arb_001"}
)
long_position.related_positions.append(short_position.id)
position_manager.database.save_position(long_position)
```

### Portfolio Analytics
```python
metrics = position_manager.get_portfolio_metrics()
print(f"Total Market Value: ${metrics.total_market_value:.2f}")
print(f"Total PnL: ${metrics.total_pnl:.2f}")
print(f"Win Rate: {metrics.win_rate:.2f}%")
print(f"Max Drawdown: {metrics.max_drawdown_percentage:.2f}%")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
report = position_manager.get_performance_report()
print(json.dumps(report, indent=2))
```

### Position Queries
```python
open_positions = position_manager.get_open_positions()
btc_positions = position_manager.get_positions_by_symbol("BTC/USDT")
arb_positions = position_manager.get_positions_by_strategy("statistical_arbitrage")
position = position_manager.get_position_by_id("position_id")
```

### Data Export
```python
json_export = position_manager.export_positions("json")
history = position_manager.get_position_history(
    symbol="BTC/USDT",
    strategy="trend_following",
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now()
)
```

---

## Integration with Other Components

### Order Manager Integration
```python
position_manager.set_order_manager(order_manager)
order_manager.set_position_manager(position_manager)

def on_order_filled(order):
    position = position_manager.add_position(
        symbol=order.symbol,
        side=PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT,
        quantity=order.filled_quantity,
        entry_price=order.average_price,
        order_id=order.id,
        strategy_name=order.strategy_name
    )
```

### Risk Manager Integration
```python
position_manager.set_risk_manager(risk_manager)
# Risk manager receives position updates automatically
```

### Market Data Integration
```python
position_manager.set_market_data_manager(market_data_manager)
async def update_prices():
    await position_manager.update_all_prices()
```

---

## Configuration

### Database Configuration
```python
position_manager = PositionManager(db_path="data/custom_positions.db")
```

### Performance Settings
```python
position_manager.price_history = defaultdict(lambda: deque(maxlen=2000))
position_manager.daily_pnl_history = deque(maxlen=365)  # One year
```

---

## Error Handling
- Invalid position operations are logged and handled gracefully
- Database errors are caught and logged
- Price update failures don't crash the system
- Inconsistent state is detected and resolved

---

## Performance Considerations
- Position objects are kept in memory for fast access
- Price history is limited to prevent memory bloat
- Database queries are optimized for common operations
- Indexes on frequently queried fields
- Batch operations for bulk updates
- Connection pooling for database access
- Asynchronous price updates
- Efficient PnL calculations
- Minimal blocking operations

---

## Testing

Run the comprehensive test suite:
```bash
python -m pytest tests/test_position_manager_enhanced.py -v
```
The tests cover:
- Position creation and updates
- PnL calculations for long/short positions
- Stop-loss and take-profit triggers
- Multi-leg position management
- Portfolio analytics
- Database persistence
- Error handling
- All enhancements (sector mapping, export, thread safety, real-time updates, stats/cleanup)

---

## Monitoring and Logging
- **INFO**: Position operations, portfolio updates
- **WARNING**: Invalid operations, data inconsistencies
- **ERROR**: Database failures, integration issues
- **DEBUG**: Detailed position calculations

---

## Best Practices
1. Always set stop-loss and take-profit for risk management
2. Use appropriate position types (SINGLE, PAIR, MULTI_LEG)
3. Track related positions for arbitrage strategies
4. Monitor portfolio metrics regularly
5. Export data for analysis and backup
6. Connect all components (order, risk, market data managers)
7. Handle order fills consistently
8. Update prices regularly
9. Monitor risk limits through integration
10. Log all operations for audit trails
11. Limit position history to prevent memory issues
12. Use batch operations for bulk updates
13. Optimize database queries for common operations
14. Monitor database size and clean up old data
15. Use appropriate indexes for frequent queries

---

## Troubleshooting
- **Position not found errors**: Check ID, verify not auto-closed, check DB
- **PnL calculation errors**: Verify side, check price updates, validate prices
- **Database errors**: Check permissions, disk space, corruption
- **Integration issues**: Verify connections, check for circular dependencies, monitor logs

---

## Future Enhancements
- Real-time streaming of position updates
- Advanced analytics (VaR, stress testing)
- Multi-exchange position aggregation
- Machine learning integration for position sizing
- Web dashboard for position monitoring
- In-memory caching for frequently accessed data
- Parallel processing for bulk operations
- Database optimization for large datasets
- Real-time notifications for position events

---

## API Reference

### PositionManager Methods
- `add_position()` - Create new position
- `update_position()` - Update position price
- `close_position()` - Close position (full or partial)
- `get_position_by_id()` - Get position by ID
- `get_open_positions()` - Get all open positions
- `get_positions_by_symbol()` - Get positions by symbol
- `get_positions_by_strategy()` - Get positions by strategy
- `get_portfolio_metrics()` - Get portfolio analytics
- `get_performance_report()` - Get detailed performance report
- `export_positions()` - Export position data
- `set_order_manager()` - Connect order manager
- `set_risk_manager()` - Connect risk manager
- `set_market_data_manager()` - Connect market data manager
- `sync_with_order_manager()` - Sync with order state

### Position Properties
- `id`, `symbol`, `side`, `quantity`, `entry_price`, `current_price`
- `market_value`, `entry_value`, `unrealized_pnl`, `realized_pnl`, `total_pnl`, `pnl_percentage`, `holding_time`
- `status`, `stop_loss`, `take_profit`, `strategy_name`, `order_ids`, `related_positions`, `metadata` 