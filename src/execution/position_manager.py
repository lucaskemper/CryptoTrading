import asyncio
import time
import sqlite3
import json
import uuid
import csv
import io
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, deque
import numpy as np
import pandas as pd

from src.utils.logger import logger
from src.utils.config_loader import config


class PositionSide(Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    """Position status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    PARTIALLY_CLOSED = "partially_closed"


class PositionType(Enum):
    """Position type enumeration."""
    SINGLE = "single"
    PAIR = "pair"  # For statistical arbitrage
    MULTI_LEG = "multi_leg"  # For complex strategies


@dataclass
class Position:
    """Position representation with comprehensive tracking."""
    id: str
    symbol: str
    side: PositionSide
    position_type: PositionType
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    last_update_time: datetime
    status: PositionStatus
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    commission: float = 0.0
    strategy_name: str = "unknown"
    order_ids: List[str] = field(default_factory=list)
    related_positions: List[str] = field(default_factory=list)  # For pair/multi-leg positions
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self._update_pnl()
    
    def _update_pnl(self):
        """Update PnL calculations."""
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - self.current_price) * self.quantity
        
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
    
    @property
    def market_value(self) -> float:
        """Calculate current market value of position."""
        return self.quantity * self.current_price
    
    @property
    def entry_value(self) -> float:
        """Calculate entry value of position."""
        return self.quantity * self.entry_price
    
    @property
    def pnl_percentage(self) -> float:
        """Calculate PnL as percentage of entry value."""
        if self.entry_value == 0:
            return 0.0
        return (self.total_pnl / self.entry_value) * 100
    
    @property
    def holding_time(self) -> timedelta:
        """Calculate how long position has been held."""
        return datetime.now() - self.entry_time
    
    def update_price(self, new_price: float):
        """Update current price and recalculate PnL."""
        self.current_price = new_price
        self.last_update_time = datetime.now()
        self._update_pnl()
    
    def add_order_id(self, order_id: str):
        """Add order ID to position tracking."""
        if order_id not in self.order_ids:
            self.order_ids.append(order_id)
    
    def partial_close(self, close_quantity: float, close_price: float, commission: float = 0.0):
        """Handle partial position close."""
        if close_quantity > self.quantity:
            raise ValueError("Close quantity cannot exceed position quantity")
        
        # Calculate realized PnL for closed portion
        if self.side == PositionSide.LONG:
            realized_pnl = (close_price - self.entry_price) * close_quantity
        else:  # SHORT
            realized_pnl = (self.entry_price - close_price) * close_quantity
        
        self.realized_pnl += realized_pnl - commission
        self.quantity -= close_quantity
        self.commission += commission
        
        if self.quantity == 0:
            self.status = PositionStatus.CLOSED
        else:
            self.status = PositionStatus.PARTIALLY_CLOSED
        
        self._update_pnl()
    
    def full_close(self, close_price: float, commission: float = 0.0):
        """Handle full position close."""
        self.partial_close(self.quantity, close_price, commission)
        self.status = PositionStatus.CLOSED


@dataclass
class PortfolioMetrics:
    """Portfolio-level metrics and analytics."""
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
    timestamp: datetime = field(default_factory=datetime.now)


class PositionDatabase:
    """Database handler for position persistence and analytics."""
    
    def __init__(self, db_path: str = "data/positions.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Positions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS positions (
                        id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        position_type TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        entry_time TEXT NOT NULL,
                        current_price REAL NOT NULL,
                        last_update_time TEXT NOT NULL,
                        status TEXT NOT NULL,
                        unrealized_pnl REAL NOT NULL,
                        realized_pnl REAL NOT NULL,
                        total_pnl REAL NOT NULL,
                        stop_loss REAL,
                        take_profit REAL,
                        commission REAL NOT NULL,
                        strategy_name TEXT NOT NULL,
                        order_ids TEXT,
                        related_positions TEXT,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Position history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS position_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        price REAL NOT NULL,
                        pnl REAL NOT NULL,
                        action TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        order_id TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Portfolio snapshots table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        total_market_value REAL NOT NULL,
                        total_unrealized_pnl REAL NOT NULL,
                        total_realized_pnl REAL NOT NULL,
                        total_pnl REAL NOT NULL,
                        total_pnl_percentage REAL NOT NULL,
                        daily_pnl REAL NOT NULL,
                        daily_pnl_percentage REAL NOT NULL,
                        position_count INTEGER NOT NULL,
                        open_position_count INTEGER NOT NULL,
                        win_rate REAL NOT NULL,
                        max_drawdown REAL NOT NULL,
                        max_drawdown_percentage REAL NOT NULL,
                        sharpe_ratio REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info(f"Position database initialized at {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize position database: {e}")
    
    def save_position(self, position: Position):
        """Save position to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO positions 
                    (id, symbol, side, position_type, quantity, entry_price, entry_time,
                     current_price, last_update_time, status, unrealized_pnl, realized_pnl,
                     total_pnl, stop_loss, take_profit, commission, strategy_name,
                     order_ids, related_positions, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    position.id,
                    position.symbol,
                    position.side.value,
                    position.position_type.value,
                    position.quantity,
                    position.entry_price,
                    position.entry_time.isoformat(),
                    position.current_price,
                    position.last_update_time.isoformat(),
                    position.status.value,
                    position.unrealized_pnl,
                    position.realized_pnl,
                    position.total_pnl,
                    position.stop_loss,
                    position.take_profit,
                    position.commission,
                    position.strategy_name,
                    json.dumps(position.order_ids),
                    json.dumps(position.related_positions),
                    json.dumps(position.metadata)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to save position to database: {e}")
    
    def load_position(self, position_id: str) -> Optional[Position]:
        """Load position from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM positions WHERE id = ?
                ''', (position_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_position(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to load position from database: {e}")
            return None
    
    def _row_to_position(self, row) -> Position:
        """Convert database row to Position object."""
        return Position(
            id=row[0],
            symbol=row[1],
            side=PositionSide(row[2]),
            position_type=PositionType(row[3]),
            quantity=row[4],
            entry_price=row[5],
            entry_time=datetime.fromisoformat(row[6]),
            current_price=row[7],
            last_update_time=datetime.fromisoformat(row[8]),
            status=PositionStatus(row[9]),
            unrealized_pnl=row[10],
            realized_pnl=row[11],
            total_pnl=row[12],
            stop_loss=row[13],
            take_profit=row[14],
            commission=row[15],
            strategy_name=row[16],
            order_ids=json.loads(row[17]) if row[17] else [],
            related_positions=json.loads(row[18]) if row[18] else [],
            metadata=json.loads(row[19]) if row[19] else {}
        )
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM positions WHERE status IN (?, ?)
                ''', (PositionStatus.OPEN.value, PositionStatus.PARTIALLY_CLOSED.value))
                
                positions = []
                for row in cursor.fetchall():
                    positions.append(self._row_to_position(row))
                return positions
                
        except Exception as e:
            logger.error(f"Failed to get open positions from database: {e}")
            return []
    
    def log_position_action(self, position_id: str, symbol: str, side: str, 
                          quantity: float, price: float, pnl: float, 
                          action: str, order_id: Optional[str] = None):
        """Log position action to history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO position_history 
                    (position_id, symbol, side, quantity, price, pnl, action, timestamp, order_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    position_id,
                    symbol,
                    side,
                    quantity,
                    price,
                    pnl,
                    action,
                    datetime.now().isoformat(),
                    order_id
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log position action to database: {e}")
    
    def save_portfolio_snapshot(self, metrics: PortfolioMetrics):
        """Save portfolio snapshot to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO portfolio_snapshots 
                    (total_market_value, total_unrealized_pnl, total_realized_pnl,
                     total_pnl, total_pnl_percentage, daily_pnl, daily_pnl_percentage,
                     position_count, open_position_count, win_rate, max_drawdown,
                     max_drawdown_percentage, sharpe_ratio, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.total_market_value,
                    metrics.total_unrealized_pnl,
                    metrics.total_realized_pnl,
                    metrics.total_pnl,
                    metrics.total_pnl_percentage,
                    metrics.daily_pnl,
                    metrics.daily_pnl_percentage,
                    metrics.position_count,
                    metrics.open_position_count,
                    metrics.win_rate,
                    metrics.max_drawdown,
                    metrics.max_drawdown_percentage,
                    metrics.sharpe_ratio,
                    metrics.timestamp.isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to save portfolio snapshot to database: {e}")


class PositionManager:
    """Comprehensive position management system for crypto trading bot."""
    
    def __init__(self, db_path: str = "data/positions.db"):
        self.database = PositionDatabase(db_path)
        self.positions: Dict[str, Position] = {}
        self.position_groups: Dict[str, List[str]] = defaultdict(list)  # For pair/multi-leg positions
        self.price_cache: Dict[str, float] = {}
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Performance tracking
        self.initial_portfolio_value: Optional[float] = None
        self.peak_portfolio_value: float = 0.0
        self.daily_pnl_history: deque = deque(maxlen=252)  # One year of trading days
        
        # External dependencies
        self.order_manager = None
        self.risk_manager = None
        self.market_data_manager = None
        
        # Thread safety
        self._lock = threading.RLock()
        self._price_update_lock = threading.Lock()
        
        # Load existing positions
        self._load_existing_positions()
        
        logger.info("PositionManager initialized successfully")
    
    def set_order_manager(self, order_manager):
        """Set order manager reference."""
        self.order_manager = order_manager
        logger.info("Order manager connected to position manager")
    
    def set_risk_manager(self, risk_manager):
        """Set risk manager reference."""
        self.risk_manager = risk_manager
        logger.info("Risk manager connected to position manager")
    
    def set_market_data_manager(self, market_data_manager):
        """Set market data manager reference."""
        self.market_data_manager = market_data_manager
        logger.info("Market data manager connected to position manager")
    
    def update_on_trade(self, symbol: str, side: str, quantity: float, price: float, order_id: str):
        """
        Update position manager when a trade occurs.
        This method is called by the order manager when an order is filled.
        
        Args:
            symbol: Trading symbol
            side: Trade side ('buy' or 'sell')
            quantity: Trade quantity
            price: Trade price
            order_id: Order ID
        """
        try:
            # Convert side string to PositionSide enum
            position_side = PositionSide.LONG if side == 'buy' else PositionSide.SHORT
            
            # Update position
            if side == 'buy':
                # Opening or adding to long position
                self.add_position(
                    symbol=symbol,
                    side=position_side,
                    quantity=quantity,
                    entry_price=price,
                    order_id=order_id,
                    strategy_name="order_manager"
                )
            else:  # sell
                # Check if we have an existing position to close
                existing_positions = self.get_positions_by_symbol(symbol)
                long_positions = [p for p in existing_positions if p.side == PositionSide.LONG and p.status != PositionStatus.CLOSED]
                
                if long_positions:
                    # Close existing long position
                    position_to_close = long_positions[0]  # Close the first one
                    realized_pnl = self.close_position(
                        position_to_close.id,
                        price,
                        quantity,
                        order_id
                    )
                    logger.info(f"Closed position {position_to_close.id}: PnL {realized_pnl:.2f}")
                else:
                    # Opening short position
                    self.add_position(
                        symbol=symbol,
                        side=position_side,
                        quantity=quantity,
                        entry_price=price,
                        order_id=order_id,
                        strategy_name="order_manager"
                    )
            
            # Notify risk manager
            if self.risk_manager:
                self.risk_manager.update_on_trade(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    order_id=order_id
                )
                
        except Exception as e:
            logger.error(f"Failed to update position manager on trade: {e}")
    
    def _load_existing_positions(self):
        """Load existing positions from database."""
        try:
            open_positions = self.database.get_open_positions()
            for position in open_positions:
                self.positions[position.id] = position
                if position.related_positions:
                    self.position_groups[position.id] = position.related_positions
            
            logger.info(f"Loaded {len(open_positions)} existing positions")
            
        except Exception as e:
            logger.error(f"Failed to load existing positions: {e}")
    
    def add_position(self, symbol: str, side: PositionSide, quantity: float, 
                    entry_price: float, order_id: str, strategy_name: str = "unknown",
                    position_type: PositionType = PositionType.SINGLE,
                    stop_loss: Optional[float] = None, 
                    take_profit: Optional[float] = None,
                    related_positions: List[str] = None,
                    metadata: Dict[str, Any] = None) -> Position:
        """Add a new position after order fill (thread-safe)."""
        with self._lock:
            return self._add_position_unsafe(symbol, side, quantity, entry_price, order_id,
                                           strategy_name, position_type, stop_loss, 
                                           take_profit, related_positions, metadata)
    
    def _add_position_unsafe(self, symbol: str, side: PositionSide, quantity: float, 
                            entry_price: float, order_id: str, strategy_name: str = "unknown",
                            position_type: PositionType = PositionType.SINGLE,
                            stop_loss: Optional[float] = None, 
                            take_profit: Optional[float] = None,
                            related_positions: List[str] = None,
                            metadata: Dict[str, Any] = None) -> Position:
        """
        Add a new position after order fill.
        
        Args:
            symbol: Trading symbol
            side: Position side (long/short)
            quantity: Position quantity
            entry_price: Entry price
            order_id: Associated order ID
            strategy_name: Strategy that generated the position
            position_type: Type of position
            stop_loss: Stop loss price
            take_profit: Take profit price
            related_positions: Related position IDs for pair/multi-leg
            metadata: Additional position metadata
            
        Returns:
            Created Position object
        """
        position_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        position = Position(
            id=position_id,
            symbol=symbol,
            side=side,
            position_type=position_type,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=current_time,
            current_price=entry_price,
            last_update_time=current_time,
            status=PositionStatus.OPEN,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=strategy_name,
            order_ids=[order_id],
            related_positions=related_positions or [],
            metadata=metadata or {}
        )
        
        # Store position
        self.positions[position_id] = position
        self.database.save_position(position)
        
        # Update position groups if related positions
        if related_positions:
            self.position_groups[position_id] = related_positions
            for related_id in related_positions:
                if related_id in self.positions:
                    self.positions[related_id].related_positions.append(position_id)
                    self.database.save_position(self.positions[related_id])
        
        # Log position action
        self.database.log_position_action(
            position_id, symbol, side.value, quantity, entry_price, 0.0, "OPEN", order_id
        )
        
        # Update price cache
        self.price_cache[symbol] = entry_price
        self._update_price_history(symbol, entry_price)
        
        # Notify risk manager
        if self.risk_manager:
            self.risk_manager.update_position(symbol, side.value, quantity, entry_price)
        
        logger.info(f"Added position: {symbol} {side.value} {quantity} @ {entry_price}")
        return position
    
    def update_position(self, position_id: str, new_price: float, 
                       order_id: Optional[str] = None):
        """Update position with new price and recalculate PnL (thread-safe)."""
        with self._price_update_lock:
            return self._update_position_unsafe(position_id, new_price, order_id)
    
    def _update_position_unsafe(self, position_id: str, new_price: float, 
                               order_id: Optional[str] = None):
        """
        Update position with new price and recalculate PnL.
        
        Args:
            position_id: Position ID to update
            new_price: New current price
            order_id: Associated order ID (optional)
        """
        if position_id not in self.positions:
            logger.warning(f"Position {position_id} not found for update")
            return
        
        position = self.positions[position_id]
        old_pnl = position.total_pnl
        
        # Update position
        position.update_price(new_price)
        if order_id:
            position.add_order_id(order_id)
        
        # Save to database
        self.database.save_position(position)
        
        # Update price cache and history
        self.price_cache[position.symbol] = new_price
        self._update_price_history(position.symbol, new_price)
        
        # Log significant PnL changes
        pnl_change = position.total_pnl - old_pnl
        if abs(pnl_change) > 0.01:  # Log if PnL changed by more than $0.01
            self.database.log_position_action(
                position_id, position.symbol, position.side.value,
                position.quantity, new_price, pnl_change, "UPDATE", order_id
            )
        
        # Check stop-loss and take-profit
        self._check_exit_conditions(position)
    
    def close_position(self, position_id: str, close_price: float, 
                      close_quantity: Optional[float] = None,
                      order_id: Optional[str] = None) -> float:
        """Close position or partial close (thread-safe)."""
        with self._lock:
            return self._close_position_unsafe(position_id, close_price, close_quantity, order_id)
    
    def _close_position_unsafe(self, position_id: str, close_price: float, 
                              close_quantity: Optional[float] = None,
                              order_id: Optional[str] = None) -> float:
        """
        Close position or partial close.
        
        Args:
            position_id: Position ID to close
            close_price: Close price
            close_quantity: Quantity to close (None for full close)
            order_id: Associated order ID
            
        Returns:
            Realized PnL from the close
        """
        if position_id not in self.positions:
            logger.warning(f"Position {position_id} not found for close")
            return 0.0
        
        position = self.positions[position_id]
        close_qty = close_quantity or position.quantity
        
        if close_qty > position.quantity:
            raise ValueError(f"Close quantity {close_qty} exceeds position quantity {position.quantity}")
        
        # Calculate realized PnL
        if position.side == PositionSide.LONG:
            realized_pnl = (close_price - position.entry_price) * close_qty
        else:  # SHORT
            realized_pnl = (position.entry_price - close_price) * close_qty
        
        # Update position
        position.partial_close(close_qty, close_price)
        
        # Save to database
        self.database.save_position(position)
        
        # Log close action
        self.database.log_position_action(
            position_id, position.symbol, position.side.value,
            close_qty, close_price, realized_pnl, "CLOSE", order_id
        )
        
        # Remove from active positions if fully closed
        if position.status == PositionStatus.CLOSED:
            del self.positions[position_id]
            if position_id in self.position_groups:
                del self.position_groups[position_id]
        
        # Notify risk manager
        if self.risk_manager:
            self.risk_manager.close_position(position.symbol, position.side.value, close_qty, close_price)
        
        logger.info(f"Closed position {position_id}: {close_qty} @ {close_price}, PnL: {realized_pnl:.2f}")
        return realized_pnl
    
    def get_open_positions(self) -> List[Position]:
        """Get all currently open positions."""
        return [pos for pos in self.positions.values() 
                if pos.status in [PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED]]
    
    def get_position_by_id(self, position_id: str) -> Optional[Position]:
        """Get position by ID."""
        return self.positions.get(position_id)
    
    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get all positions for a specific symbol."""
        return [pos for pos in self.positions.values() if pos.symbol == symbol]
    
    def get_positions_by_strategy(self, strategy_name: str) -> List[Position]:
        """Get all positions for a specific strategy."""
        return [pos for pos in self.positions.values() if pos.strategy_name == strategy_name]
    
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics."""
        open_positions = self.get_open_positions()
        
        if not open_positions:
            return self._empty_portfolio_metrics()
        
        # Basic calculations
        total_market_value = sum(pos.market_value for pos in open_positions)
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in open_positions)
        total_realized_pnl = sum(pos.realized_pnl for pos in open_positions)
        total_pnl = total_unrealized_pnl + total_realized_pnl
        
        # Calculate daily PnL
        today = datetime.now().date()
        daily_pnl = sum(pos.unrealized_pnl for pos in open_positions 
                       if pos.last_update_time.date() == today)
        
        # Exposure by asset
        exposure_by_asset = defaultdict(float)
        for pos in open_positions:
            exposure_by_asset[pos.symbol] += pos.market_value
        
        # Calculate win rate
        closed_positions = self._get_closed_positions_recent()
        win_count = sum(1 for pos in closed_positions if pos.total_pnl > 0)
        win_rate = win_count / len(closed_positions) if closed_positions else 0.0
        
        # Calculate average holding time
        holding_times = [pos.holding_time for pos in open_positions]
        avg_holding_time = sum(holding_times, timedelta()) / len(holding_times) if holding_times else timedelta()
        
        # Calculate drawdown
        max_drawdown, max_drawdown_pct = self._calculate_drawdown()
        
        # Calculate Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Calculate total PnL percentage
        total_entry_value = sum(pos.entry_value for pos in open_positions)
        total_pnl_percentage = (total_pnl / total_entry_value * 100) if total_entry_value > 0 else 0.0
        
        # Daily PnL percentage
        daily_pnl_percentage = (daily_pnl / total_entry_value * 100) if total_entry_value > 0 else 0.0
        
        return PortfolioMetrics(
            total_market_value=total_market_value,
            total_unrealized_pnl=total_unrealized_pnl,
            total_realized_pnl=total_realized_pnl,
            total_pnl=total_pnl,
            total_pnl_percentage=total_pnl_percentage,
            daily_pnl=daily_pnl,
            daily_pnl_percentage=daily_pnl_percentage,
            exposure_by_asset=dict(exposure_by_asset),
            exposure_by_sector=self._calculate_sector_exposure(exposure_by_asset),
            position_count=len(open_positions),
            open_position_count=len([p for p in open_positions if p.status == PositionStatus.OPEN]),
            win_rate=win_rate,
            average_holding_time=avg_holding_time,
            max_drawdown=max_drawdown,
            max_drawdown_percentage=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio
        )
    
    def _empty_portfolio_metrics(self) -> PortfolioMetrics:
        """Return empty portfolio metrics when no positions exist."""
        return PortfolioMetrics(
            total_market_value=0.0,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            total_pnl=0.0,
            total_pnl_percentage=0.0,
            daily_pnl=0.0,
            daily_pnl_percentage=0.0,
            exposure_by_asset={},
            exposure_by_sector={},
            position_count=0,
            open_position_count=0,
            win_rate=0.0,
            average_holding_time=timedelta(),
            max_drawdown=0.0,
            max_drawdown_percentage=0.0,
            sharpe_ratio=0.0
        )
    
    def _get_closed_positions_recent(self, days: int = 30) -> List[Position]:
        """Get recently closed positions for analytics."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM positions 
                    WHERE status = ? AND last_update_time >= ?
                    ORDER BY last_update_time DESC
                ''', (PositionStatus.CLOSED.value, cutoff_date.isoformat()))
                
                positions = []
                for row in cursor.fetchall():
                    positions.append(self.database._row_to_position(row))
                
                logger.info(f"Retrieved {len(positions)} closed positions from last {days} days")
                return positions
                
        except Exception as e:
            logger.error(f"Failed to get closed positions: {e}")
            return []
    
    def _calculate_drawdown(self) -> Tuple[float, float]:
        """Calculate maximum drawdown."""
        if not self.daily_pnl_history:
            return 0.0, 0.0
        
        cumulative_pnl = np.cumsum(list(self.daily_pnl_history))
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        
        max_drawdown = float(np.max(drawdown))
        max_drawdown_pct = (max_drawdown / running_max[-1] * 100) if running_max[-1] > 0 else 0.0
        
        return max_drawdown, max_drawdown_pct
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        if len(self.daily_pnl_history) < 2:
            return 0.0
        
        returns = list(self.daily_pnl_history)
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming daily returns)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return float(sharpe)
    
    def _calculate_sector_exposure(self, asset_exposure: Dict[str, float]) -> Dict[str, float]:
        """Calculate exposure by sector with comprehensive crypto sector mapping."""
        # Comprehensive sector mapping for crypto assets
        sector_mapping = {
            # Layer 1 Blockchains
            'BTC': 'layer1',
            'ETH': 'layer1',
            'SOL': 'layer1',
            'ADA': 'layer1',
            'DOT': 'layer1',
            'AVAX': 'layer1',
            'MATIC': 'layer1',
            'LINK': 'layer1',
            'ATOM': 'layer1',
            'NEAR': 'layer1',
            'FTM': 'layer1',
            'ALGO': 'layer1',
            'ICP': 'layer1',
            
            # Layer 2 Solutions
            'ARB': 'layer2',
            'OP': 'layer2',
            'IMX': 'layer2',
            'ZKS': 'layer2',
            'STX': 'layer2',
            
            # DeFi Tokens
            'UNI': 'defi',
            'AAVE': 'defi',
            'COMP': 'defi',
            'CRV': 'defi',
            'SUSHI': 'defi',
            'YFI': 'defi',
            'SNX': 'defi',
            'MKR': 'defi',
            'BAL': 'defi',
            '1INCH': 'defi',
            'CAKE': 'defi',
            'SAND': 'defi',
            'MANA': 'defi',
            'AXS': 'defi',
            'ENJ': 'defi',
            
            # Gaming & Metaverse
            'GALA': 'gaming',
            'CHZ': 'gaming',
            'ENJ': 'gaming',
            'MANA': 'gaming',
            'SAND': 'gaming',
            'AXS': 'gaming',
            'ILV': 'gaming',
            'ALICE': 'gaming',
            
            # NFT & Collectibles
            'APE': 'nft',
            'FLOW': 'nft',
            'RARI': 'nft',
            'LOOKS': 'nft',
            
            # Stablecoins
            'USDT': 'stablecoin',
            'USDC': 'stablecoin',
            'DAI': 'stablecoin',
            'BUSD': 'stablecoin',
            'FRAX': 'stablecoin',
            'TUSD': 'stablecoin',
            'USDP': 'stablecoin',
            
            # Exchange Tokens
            'BNB': 'exchange',
            'OKB': 'exchange',
            'FTT': 'exchange',
            'HT': 'exchange',
            'KCS': 'exchange',
            'GT': 'exchange',
            
            # Privacy & Security
            'XMR': 'privacy',
            'ZEC': 'privacy',
            'DASH': 'privacy',
            'XNO': 'privacy',
            
            # AI & Big Data
            'OCEAN': 'ai_data',
            'FET': 'ai_data',
            'AGIX': 'ai_data',
            'RLC': 'ai_data',
            'GRT': 'ai_data',
            
            # Infrastructure & Oracles
            'LINK': 'infrastructure',
            'BAND': 'infrastructure',
            'API3': 'infrastructure',
            'UMA': 'infrastructure',
            'PENDLE': 'infrastructure',
            
            # Meme Coins
            'DOGE': 'meme',
            'SHIB': 'meme',
            'PEPE': 'meme',
            'FLOKI': 'meme',
            'BONK': 'meme',
            'WIF': 'meme'
        }
        
        sector_exposure = defaultdict(float)
        for asset, exposure in asset_exposure.items():
            # Extract base asset from symbol (e.g., "BTC/USDT" -> "BTC")
            base_asset = asset.split('/')[0] if '/' in asset else asset
            sector = sector_mapping.get(base_asset, 'other')
            sector_exposure[sector] += exposure
        
        return dict(sector_exposure)
    
    def _update_price_history(self, symbol: str, price: float):
        """Update price history for analytics."""
        self.price_history[symbol].append(price)
    
    def _check_exit_conditions(self, position: Position):
        """Check if position should be closed due to stop-loss or take-profit."""
        if position.status != PositionStatus.OPEN:
            return
        
        should_close = False
        close_reason = ""
        
        # Check stop-loss
        if position.stop_loss is not None:
            if (position.side == PositionSide.LONG and position.current_price <= position.stop_loss) or \
               (position.side == PositionSide.SHORT and position.current_price >= position.stop_loss):
                should_close = True
                close_reason = "stop_loss"
        
        # Check take-profit
        if position.take_profit is not None:
            if (position.side == PositionSide.LONG and position.current_price >= position.take_profit) or \
               (position.side == PositionSide.SHORT and position.current_price <= position.take_profit):
                should_close = True
                close_reason = "take_profit"
        
        if should_close:
            logger.info(f"Auto-closing position {position.id} due to {close_reason}")
            # Close the entire position when stop-loss or take-profit is triggered
            self.close_position(position.id, position.current_price, close_quantity=position.quantity)
    
    def sync_with_order_manager(self):
        """Reconcile positions with order manager state."""
        if not self.order_manager:
            return
        
        # This would typically check order manager for any discrepancies
        # and update positions accordingly
        logger.info("Syncing positions with order manager")
    
    def get_position_history(self, symbol: Optional[str] = None, 
                           strategy: Optional[str] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get historical position data for analysis."""
        try:
            query = '''
                SELECT * FROM positions 
                WHERE 1=1
            '''
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if strategy:
                query += " AND strategy_name = ?"
                params.append(strategy)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            if start_date:
                query += " AND entry_time >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND entry_time <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY entry_time DESC"
            
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                history = []
                for row in cursor.fetchall():
                    position = self.database._row_to_position(row)
                    history.append(self._position_to_dict(position))
                
                logger.info(f"Retrieved {len(history)} historical positions")
                return history
                
        except Exception as e:
            logger.error(f"Failed to get position history: {e}")
            return []
    
    def get_position_analytics(self, symbol: Optional[str] = None,
                             strategy: Optional[str] = None,
                             days: int = 30) -> Dict[str, Any]:
        """Get comprehensive position analytics."""
        try:
            start_date = datetime.now() - timedelta(days=days)
            positions = self.get_position_history(
                symbol=symbol, 
                strategy=strategy, 
                start_date=start_date
            )
            
            if not positions:
                return self._empty_analytics()
            
            # Calculate analytics
            total_positions = len(positions)
            winning_positions = len([p for p in positions if p['total_pnl'] > 0])
            losing_positions = len([p for p in positions if p['total_pnl'] < 0])
            
            total_pnl = sum(p['total_pnl'] for p in positions)
            total_volume = sum(p['entry_value'] for p in positions)
            avg_holding_time = np.mean([p['holding_time_days'] for p in positions])
            
            # Calculate returns
            returns = [p['pnl_percentage'] for p in positions]
            avg_return = np.mean(returns) if returns else 0.0
            std_return = np.std(returns) if returns else 0.0
            
            # Calculate Sharpe ratio
            sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
            
            # Calculate max drawdown
            cumulative_pnl = np.cumsum([p['total_pnl'] for p in positions])
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = running_max - cumulative_pnl
            max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
            
            # Strategy performance
            strategy_performance = defaultdict(lambda: {
                'count': 0, 'total_pnl': 0.0, 'win_rate': 0.0
            })
            
            for pos in positions:
                strategy = pos['strategy_name']
                strategy_performance[strategy]['count'] += 1
                strategy_performance[strategy]['total_pnl'] += pos['total_pnl']
                if pos['total_pnl'] > 0:
                    strategy_performance[strategy]['wins'] = strategy_performance[strategy].get('wins', 0) + 1
            
            # Calculate win rates
            for strategy in strategy_performance:
                count = strategy_performance[strategy]['count']
                wins = strategy_performance[strategy].get('wins', 0)
                strategy_performance[strategy]['win_rate'] = wins / count if count > 0 else 0.0
            
            return {
                'summary': {
                    'total_positions': total_positions,
                    'winning_positions': winning_positions,
                    'losing_positions': losing_positions,
                    'win_rate': winning_positions / total_positions if total_positions > 0 else 0.0,
                    'total_pnl': total_pnl,
                    'total_volume': total_volume,
                    'avg_holding_time_days': avg_holding_time,
                    'avg_return_percent': avg_return,
                    'std_return_percent': std_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown
                },
                'strategy_performance': dict(strategy_performance),
                'positions': positions,
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Failed to get position analytics: {e}")
            return self._empty_analytics()
    
    def _empty_analytics(self) -> Dict[str, Any]:
        """Return empty analytics structure."""
        return {
            'summary': {
                'total_positions': 0,
                'winning_positions': 0,
                'losing_positions': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'total_volume': 0.0,
                'avg_holding_time_days': 0.0,
                'avg_return_percent': 0.0,
                'std_return_percent': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            },
            'strategy_performance': {},
            'positions': [],
            'period_days': 0
        }
    
    def export_positions(self, format: str = "json", include_closed: bool = False) -> str:
        """Export positions to various formats."""
        positions = self.get_open_positions()
        if include_closed:
            positions.extend(self._get_closed_positions_recent(days=365))  # Last year
        
        if format == "json":
            return json.dumps([self._position_to_dict(pos) for pos in positions], indent=2)
        elif format == "csv":
            return self._export_to_csv(positions)
        elif format == "dataframe":
            return self._export_to_dataframe(positions)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_to_csv(self, positions: List[Position]) -> str:
        """Export positions to CSV format."""
        if not positions:
            return ""
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'id', 'symbol', 'side', 'position_type', 'quantity', 'entry_price',
            'current_price', 'entry_time', 'last_update_time', 'status',
            'unrealized_pnl', 'realized_pnl', 'total_pnl', 'pnl_percentage',
            'market_value', 'entry_value', 'stop_loss', 'take_profit',
            'commission', 'strategy_name', 'holding_time_days',
            'order_ids', 'related_positions'
        ])
        
        # Write data rows
        for pos in positions:
            writer.writerow([
                pos.id,
                pos.symbol,
                pos.side.value,
                pos.position_type.value,
                pos.quantity,
                pos.entry_price,
                pos.current_price,
                pos.entry_time.isoformat(),
                pos.last_update_time.isoformat(),
                pos.status.value,
                pos.unrealized_pnl,
                pos.realized_pnl,
                pos.total_pnl,
                pos.pnl_percentage,
                pos.market_value,
                pos.entry_value,
                pos.stop_loss or '',
                pos.take_profit or '',
                pos.commission,
                pos.strategy_name,
                pos.holding_time.days,
                ';'.join(pos.order_ids),
                ';'.join(pos.related_positions)
            ])
        
        return output.getvalue()
    
    def _export_to_dataframe(self, positions: List[Position]) -> pd.DataFrame:
        """Export positions to pandas DataFrame."""
        if not positions:
            return pd.DataFrame()
        
        data = []
        for pos in positions:
            data.append({
                'id': pos.id,
                'symbol': pos.symbol,
                'side': pos.side.value,
                'position_type': pos.position_type.value,
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'entry_time': pos.entry_time,
                'last_update_time': pos.last_update_time,
                'status': pos.status.value,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl,
                'total_pnl': pos.total_pnl,
                'pnl_percentage': pos.pnl_percentage,
                'market_value': pos.market_value,
                'entry_value': pos.entry_value,
                'stop_loss': pos.stop_loss,
                'take_profit': pos.take_profit,
                'commission': pos.commission,
                'strategy_name': pos.strategy_name,
                'holding_time_days': pos.holding_time.days,
                'order_ids': ';'.join(pos.order_ids),
                'related_positions': ';'.join(pos.related_positions)
            })
        
        return pd.DataFrame(data)
    
    def _position_to_dict(self, position: Position) -> Dict[str, Any]:
        """Convert position to dictionary for export."""
        return {
            "id": position.id,
            "symbol": position.symbol,
            "side": position.side.value,
            "quantity": position.quantity,
            "entry_price": position.entry_price,
            "current_price": position.current_price,
            "entry_time": position.entry_time.isoformat(),
            "unrealized_pnl": position.unrealized_pnl,
            "realized_pnl": position.realized_pnl,
            "total_pnl": position.total_pnl,
            "pnl_percentage": position.pnl_percentage,
            "market_value": position.market_value,
            "status": position.status.value,
            "strategy_name": position.strategy_name,
            "holding_time": str(position.holding_time)
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        metrics = self.get_portfolio_metrics()
        open_positions = self.get_open_positions()
        
        # Top performers
        top_positions = sorted(open_positions, key=lambda p: p.total_pnl, reverse=True)[:5]
        worst_positions = sorted(open_positions, key=lambda p: p.total_pnl)[:5]
        
        return {
            "portfolio_metrics": {
                "total_market_value": metrics.total_market_value,
                "total_pnl": metrics.total_pnl,
                "total_pnl_percentage": metrics.total_pnl_percentage,
                "daily_pnl": metrics.daily_pnl,
                "position_count": metrics.position_count,
                "win_rate": metrics.win_rate,
                "max_drawdown": metrics.max_drawdown,
                "sharpe_ratio": metrics.sharpe_ratio
            },
            "exposure": {
                "by_asset": metrics.exposure_by_asset,
                "by_sector": metrics.exposure_by_sector
            },
            "top_positions": [self._position_to_dict(p) for p in top_positions],
            "worst_positions": [self._position_to_dict(p) for p in worst_positions],
            "timestamp": datetime.now().isoformat()
        }
    
    async def update_all_prices(self):
        """Update prices for all open positions with real-time market data."""
        if not self.market_data_manager:
            logger.warning("Market data manager not available for price updates")
            return
        
        open_positions = self.get_open_positions()
        if not open_positions:
            return
        
        symbols = list(set(pos.symbol for pos in open_positions))
        
        # Batch update prices for efficiency
        try:
            # Get current prices from market data manager
            current_prices = await self._fetch_current_prices(symbols)
            
            # Update positions with new prices
            updated_count = 0
            for symbol, price in current_prices.items():
                if price is not None:
                    positions = self.get_positions_by_symbol(symbol)
                    for position in positions:
                        self.update_position(position.id, price)
                        updated_count += 1
            
            logger.info(f"Updated prices for {updated_count} positions across {len(current_prices)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to update prices: {e}")
    
    async def _fetch_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch current prices from market data manager."""
        current_prices = {}
        
        for symbol in symbols:
            try:
                # This would integrate with your market data manager
                # For now, return cached prices or None
                price = self.price_cache.get(symbol)
                if price is not None:
                    current_prices[symbol] = price
                else:
                    # Simulate price fetch - replace with actual market data manager call
                    logger.debug(f"No cached price for {symbol}, skipping update")
                    
            except Exception as e:
                logger.error(f"Failed to fetch price for {symbol}: {e}")
        
        return current_prices
    
    def start_price_update_loop(self, update_interval: int = 30):
        """Start continuous price update loop (non-blocking)."""
        async def price_update_loop():
            while True:
                try:
                    await self.update_all_prices()
                    await asyncio.sleep(update_interval)
                except Exception as e:
                    logger.error(f"Error in price update loop: {e}")
                    await asyncio.sleep(5)  # Brief pause on error
        
        # Start the loop in the background
        asyncio.create_task(price_update_loop())
        logger.info(f"Started price update loop with {update_interval}s interval")
    
    def update_price_from_feed(self, symbol: str, price: float, timestamp: Optional[datetime] = None):
        """Update price from external feed (thread-safe)."""
        with self._price_update_lock:
            # Update price cache
            self.price_cache[symbol] = price
            self._update_price_history(symbol, price)
            
            # Update all positions for this symbol
            positions = self.get_positions_by_symbol(symbol)
            for position in positions:
                self._update_position_unsafe(position.id, price)
            
            logger.debug(f"Updated {len(positions)} positions for {symbol} @ {price}")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol."""
        return self.price_cache.get(symbol)
    
    def get_price_history(self, symbol: str, limit: int = 100) -> List[float]:
        """Get price history for a symbol."""
        history = self.price_history.get(symbol, deque())
        return list(history)[-limit:] if history else []
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up old position data from database."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                
                # Clean up old position history
                cursor.execute('''
                    DELETE FROM position_history 
                    WHERE timestamp < ?
                ''', (cutoff_date.isoformat(),))
                history_deleted = cursor.rowcount
                
                # Clean up old portfolio snapshots (keep daily snapshots)
                cursor.execute('''
                    DELETE FROM portfolio_snapshots 
                    WHERE timestamp < ? AND id NOT IN (
                        SELECT id FROM portfolio_snapshots 
                        WHERE timestamp >= ? 
                        GROUP BY DATE(timestamp) 
                        HAVING MAX(timestamp)
                    )
                ''', (cutoff_date.isoformat(), cutoff_date.isoformat()))
                snapshots_deleted = cursor.rowcount
                
                # Clean up old closed positions (keep last 1000)
                cursor.execute('''
                    DELETE FROM positions 
                    WHERE status = ? AND last_update_time < ? 
                    AND id NOT IN (
                        SELECT id FROM positions 
                        WHERE status = ? 
                        ORDER BY last_update_time DESC 
                        LIMIT 1000
                    )
                ''', (PositionStatus.CLOSED.value, cutoff_date.isoformat(), PositionStatus.CLOSED.value))
                positions_deleted = cursor.rowcount
                
                conn.commit()
                
                logger.info(f"Cleaned up {history_deleted} history records, "
                          f"{snapshots_deleted} snapshots, {positions_deleted} closed positions")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                
                # Count records in each table
                cursor.execute("SELECT COUNT(*) FROM positions")
                total_positions = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM positions WHERE status = ?", (PositionStatus.OPEN.value,))
                open_positions = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM position_history")
                history_records = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM portfolio_snapshots")
                snapshots = cursor.fetchone()[0]
                
                # Get database size
                cursor.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                db_size = page_count * page_size
                
                return {
                    'total_positions': total_positions,
                    'open_positions': open_positions,
                    'closed_positions': total_positions - open_positions,
                    'history_records': history_records,
                    'snapshots': snapshots,
                    'database_size_bytes': db_size,
                    'database_size_mb': db_size / (1024 * 1024)
                }
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    async def shutdown(self):
        """Clean shutdown of position manager."""
        logger.info("Shutting down position manager")
        
        # Save final portfolio snapshot
        metrics = self.get_portfolio_metrics()
        self.database.save_portfolio_snapshot(metrics)
        
        # Close database connections
        # Database connection is handled by sqlite3 context managers


# Example usage and testing
async def main():
    """Example usage of PositionManager."""
    position_manager = PositionManager()
    
    # Add a position
    position = position_manager.add_position(
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        quantity=0.1,
        entry_price=50000.0,
        order_id="order_123",
        strategy_name="statistical_arbitrage",
        stop_loss=48000.0,
        take_profit=52000.0
    )
    
    # Update position price
    position_manager.update_position(position.id, 51000.0)
    
    # Get portfolio metrics
    metrics = position_manager.get_portfolio_metrics()
    print(f"Total PnL: ${metrics.total_pnl:.2f}")
    
    # Close position
    realized_pnl = position_manager.close_position(position.id, 51500.0)
    print(f"Realized PnL: ${realized_pnl:.2f}")
    
    # Get performance report
    report = position_manager.get_performance_report()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
