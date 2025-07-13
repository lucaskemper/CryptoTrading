import asyncio
import time
import sqlite3
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, deque
import uuid
import numpy as np
import pandas as pd

from src.utils.logger import logger
from src.utils.config_loader import config
from src.execution.position_manager import PositionSide


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskEventType(Enum):
    """Risk event type enumeration."""
    ORDER_BLOCKED = "order_blocked"
    POSITION_LIMIT_BREACH = "position_limit_breach"
    DRAWDOWN_BREACH = "drawdown_breach"
    EXPOSURE_LIMIT_BREACH = "exposure_limit_breach"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    TAKE_PROFIT_TRIGGERED = "take_profit_triggered"


@dataclass
class RiskEvent:
    """Risk event record."""
    id: str
    event_type: RiskEventType
    symbol: str
    message: str
    risk_level: RiskLevel
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Position representation."""
    symbol: str
    side: str  # 'buy' (long) or 'sell' (short)
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def market_value(self) -> float:
        """Calculate current market value of position."""
        return self.quantity * self.current_price
    
    @property
    def pnl_percentage(self) -> float:
        """Calculate PnL as percentage of entry value."""
        entry_value = self.quantity * self.entry_price
        if entry_value == 0:
            return 0.0
        return (self.unrealized_pnl / entry_value) * 100


@dataclass
class PortfolioMetrics:
    """Portfolio risk metrics."""
    total_value: float
    total_pnl: float
    total_pnl_percentage: float
    daily_pnl: float
    daily_pnl_percentage: float
    max_drawdown: float
    max_drawdown_percentage: float
    exposure_by_asset: Dict[str, float]
    exposure_by_sector: Dict[str, float]
    correlation_matrix: Dict[str, Dict[str, float]]
    var_95: float  # Value at Risk (95% confidence)
    sharpe_ratio: float
    timestamp: datetime = field(default_factory=datetime.now)


class RiskDatabase:
    """Database handler for risk events and metrics."""
    
    def __init__(self, db_path: str = "data/risk_events.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Risk events table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS risk_events (
                        id TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        message TEXT NOT NULL,
                        risk_level TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        details TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Portfolio metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS portfolio_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        total_value REAL NOT NULL,
                        total_pnl REAL NOT NULL,
                        total_pnl_percentage REAL NOT NULL,
                        daily_pnl REAL NOT NULL,
                        daily_pnl_percentage REAL NOT NULL,
                        max_drawdown REAL NOT NULL,
                        max_drawdown_percentage REAL NOT NULL,
                        var_95 REAL NOT NULL,
                        sharpe_ratio REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Position snapshots table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS position_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        current_price REAL NOT NULL,
                        unrealized_pnl REAL NOT NULL,
                        stop_loss REAL,
                        take_profit REAL,
                        timestamp TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info(f"Risk database initialized at {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize risk database: {e}")
    
    def log_risk_event(self, event: RiskEvent):
        """Log risk event to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO risk_events 
                    (id, event_type, symbol, message, risk_level, timestamp, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.id,
                    event.event_type.value,
                    event.symbol,
                    event.message,
                    event.risk_level.value,
                    event.timestamp.isoformat(),
                    json.dumps(event.details)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log risk event to database: {e}")
    
    def log_portfolio_metrics(self, metrics: PortfolioMetrics):
        """Log portfolio metrics to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO portfolio_metrics 
                    (total_value, total_pnl, total_pnl_percentage, daily_pnl, 
                     daily_pnl_percentage, max_drawdown, max_drawdown_percentage,
                     var_95, sharpe_ratio, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.total_value,
                    metrics.total_pnl,
                    metrics.total_pnl_percentage,
                    metrics.daily_pnl,
                    metrics.daily_pnl_percentage,
                    metrics.max_drawdown,
                    metrics.max_drawdown_percentage,
                    metrics.var_95,
                    metrics.sharpe_ratio,
                    metrics.timestamp.isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log portfolio metrics to database: {e}")
    
    def get_risk_events(self, limit: int = 100, days: int = 30) -> List[Dict[str, Any]]:
        """Get risk events from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, event_type, symbol, message, risk_level, timestamp, details
                    FROM risk_events 
                    WHERE timestamp >= datetime('now', '-{} days')
                    ORDER BY timestamp DESC 
                    LIMIT ?
                '''.format(days), (limit,))
                
                events = []
                for row in cursor.fetchall():
                    events.append({
                        'id': row[0],
                        'event_type': row[1],
                        'symbol': row[2],
                        'message': row[3],
                        'risk_level': row[4],
                        'timestamp': row[5],
                        'details': json.loads(row[6]) if row[6] else {}
                    })
                return events
                
        except Exception as e:
            logger.error(f"Failed to get risk events from database: {e}")
            return []


class RiskManager:
    """Enhanced risk management system for crypto trading bot."""
    
    def __init__(self, portfolio_manager=None, market_data_manager=None, db_path: str = "data/risk_events.db"):
        # External dependencies
        self.portfolio_manager = portfolio_manager
        self.market_data_manager = market_data_manager
        self.position_manager = None
        self.order_manager = None
        self.db = RiskDatabase(db_path)
        self.risk_config = config.get("risk", {})
        
        # Set initial portfolio value from config or use default
        self.initial_portfolio_value = self.risk_config.get('initial_portfolio_value', 10000)
        self.current_portfolio_value = 0.0  # Start with 0, will be set by set_initial_portfolio_value
        
        # Internal state
        self.positions: Dict[str, Position] = {}
        self.risk_events: List[RiskEvent] = []
        self.trading_paused = False
        self.consecutive_losses = 0
        self.last_loss_time = None
        self.daily_drawdown = 0.0
        self.total_drawdown = 0.0
        self.max_drawdown = 0.0
        self.max_drawdown_percentage = 0.0
        self.max_daily_drawdown = self.risk_config.get('max_daily_drawdown', 0.05)
        self.max_total_drawdown = self.risk_config.get('max_total_drawdown', 0.15)
        self.max_open_positions = self.risk_config.get('max_open_positions', 10)
        self.max_positions_per_asset = self.risk_config.get('max_positions_per_asset', 3)
        self.max_order_size = self.risk_config.get('max_order_size', 0.05)
        self.max_position_size = self.risk_config.get('max_position_size', 0.1)
        self.max_total_exposure = self.risk_config.get('max_total_exposure', 0.8)
        self.max_single_asset_exposure = self.risk_config.get('max_single_asset_exposure', 0.3)
        self.max_correlated_exposure = self.risk_config.get('max_correlated_exposure', 0.5)
        self.risk_per_trade = self.risk_config.get('risk_per_trade', 0.02)
        self.default_stop_loss = self.risk_config.get('stop_loss_percentage', 0.05)
        self.default_take_profit = self.risk_config.get('take_profit_percentage', 0.1)
        self.max_consecutive_losses = self.risk_config.get('max_consecutive_losses', 5)
        self.volatility_threshold = self.risk_config.get('volatility_threshold', 0.1)
        self.max_leverage = self.risk_config.get('max_leverage', 1.0)
        self.min_correlation_threshold = self.risk_config.get('min_correlation_threshold', 0.7)
        self.sector_exposure_limit = self.risk_config.get('sector_exposure_limit', 0.4)
        
        # Sector classification
        self.sector_classification = {
            "BTC": "Layer1",
            "ETH": "Layer1", 
            "SOL": "Layer1",
            "ADA": "Layer1",
            "DOT": "Layer1",
            "AVAX": "Layer1",
            "MATIC": "Layer2",
            "OP": "Layer2",
            "ARB": "Layer2",
            "USDT": "Stablecoin",
            "USDC": "Stablecoin",
            "DAI": "Stablecoin",
            "UNI": "DeFi",
            "AAVE": "DeFi",
            "COMP": "DeFi",
            "LINK": "Oracle",
            "BAND": "Oracle"
        }
        
        # State tracking
        self.positions: Dict[str, Position] = {}
        self.risk_events: List[RiskEvent] = []
        self.trading_paused = False
        self.pause_reason = ""
        self.consecutive_losses = 0
        self.daily_pnl_history = deque(maxlen=30)  # 30 days
        self.portfolio_metrics: Optional[PortfolioMetrics] = None
        
        # Performance tracking
        self.start_time = datetime.now()
        self.peak_portfolio_value = 0.0
        
        # Price history for correlation analysis
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        logger.info("Enhanced RiskManager initialized successfully")
    
    def set_initial_portfolio_value(self, value: float):
        """Set initial portfolio value for drawdown calculations."""
        self.initial_portfolio_value = value
        self.peak_portfolio_value = value
        logger.info(f"Initial portfolio value set to: ${value:,.2f}")
    
    def _get_portfolio_value(self) -> float:
        """Get current portfolio value from portfolio manager or fallback."""
        # Always use current_portfolio_value if set
        if hasattr(self, 'current_portfolio_value'):
            logger.debug(f"[RISK DEBUG] Using current_portfolio_value: {self.current_portfolio_value}")
            if self.current_portfolio_value > 0:
                return self.current_portfolio_value
            else:
                logger.warning("[RISK WARNING] current_portfolio_value is 0. Fallback to initial_portfolio_value.")
                return self.initial_portfolio_value
        # Fallback to sum of position values + initial value
        total_value = self.initial_portfolio_value
        for position in self.positions.values():
            total_value += position.unrealized_pnl
        logger.debug(f"[RISK DEBUG] Fallback portfolio value: {total_value}")
        return total_value
    
    def update_portfolio_value(self, portfolio_value: float):
        """Update the current portfolio value for risk calculations."""
        self.current_portfolio_value = portfolio_value
    
    def _extract_signal_data(self, signal) -> Dict[str, Any]:
        """Extract signal data consistently whether it's a dict or object."""
        if isinstance(signal, dict):
            return {
                'symbol': signal.get('symbol', ''),
                'side': signal.get('side', ''),
                'quantity': signal.get('quantity', 0.0),
                'confidence': signal.get('confidence', 0.5),
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit')
            }
        else:
            # Assume it's an object with attributes
            return {
                'symbol': getattr(signal, 'symbol', ''),
                'side': getattr(signal, 'side', ''),
                'quantity': getattr(signal, 'quantity', 0.0),
                'confidence': getattr(signal, 'confidence', 0.5),
                'stop_loss': getattr(signal, 'stop_loss', None),
                'take_profit': getattr(signal, 'take_profit', None)
            }
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price from market data manager."""
        if self.market_data_manager:
            try:
                return self.market_data_manager.get_latest_price(symbol)
            except Exception as e:
                logger.error(f"Failed to get current price for {symbol}: {e}")
        
        # Fallback to position's current price if available
        for position in self.positions.values():
            if position.symbol == symbol:
                return position.current_price
        
        # Final fallback
        return 100.0
    
    def _update_price_history(self, symbol: str, price: float):
        """Update price history for correlation analysis."""
        self.price_history[symbol].append(price)
    
    def _calculate_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix from price history."""
        if len(self.price_history) < 2:
            return {}
        
        try:
            # Convert price histories to pandas DataFrame
            price_data = {}
            min_length = min(len(history) for history in self.price_history.values())
            
            if min_length < 10:  # Need at least 10 data points
                return {}
            
            for symbol, history in self.price_history.items():
                if len(history) >= min_length:
                    price_data[symbol] = list(history)[-min_length:]
            
            if len(price_data) < 2:
                return {}
            
            df = pd.DataFrame(price_data)
            correlation_matrix = df.corr().to_dict()
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Failed to calculate correlation matrix: {e}")
            return {}
    
    def _calculate_sector_exposure(self, portfolio_value: float) -> Dict[str, float]:
        """Calculate exposure by sector."""
        sector_exposure = defaultdict(float)
        
        for position in self.positions.values():
            symbol_base = position.symbol.split('/')[0]  # Extract base asset
            sector = self.sector_classification.get(symbol_base, "Other")
            sector_exposure[sector] += position.market_value / portfolio_value
        
        return dict(sector_exposure)
    
    def check_order_risk(self, signal) -> Tuple[bool, str]:
        """
        Pre-trade risk check for a new order.
        
        Args:
            signal: Trade signal from strategy
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        try:
            # Check if trading is paused
            if self.trading_paused:
                return False, f"Trading paused: {self.pause_reason}"
            
            # Check position size limits
            if not self._check_position_size_limits(signal):
                return False, "Position size exceeds limits"
            
            # Check exposure limits
            if not self._check_exposure_limits(signal):
                return False, "Exposure limits exceeded"
            
            # Check position count limits
            if not self._check_position_count_limits(signal):
                return False, "Position count limits exceeded"
            
            # Check correlation limits (before drawdown to ensure it gets tested)
            if not self._check_correlation_limits(signal):
                return False, "Correlation limits exceeded"
            
            # Check circuit breakers (before drawdown to ensure volatility gets tested)
            if not self._check_circuit_breakers():
                return False, "Circuit breaker triggered"
            
            # Check drawdown limits (last to avoid blocking other tests)
            if not self._check_drawdown_limits():
                return False, "Drawdown limits exceeded"
            
            # Validate stop-loss and take-profit
            if not self._validate_risk_levels(signal):
                return False, "Invalid risk levels"
            
            return True, "Order approved"
            
        except Exception as e:
            logger.error(f"Error in order risk check: {e}")
            return False, f"Risk check error: {str(e)}"
    
    def _check_correlation_limits(self, signal) -> bool:
        """Check correlation limits for new position (patched: sum all correlated exposures)."""
        if len(self.positions) == 0:
            return True
        
        # Extract signal data
        signal_data = self._extract_signal_data(signal)
        symbol = signal_data['symbol']
        quantity = signal_data['quantity']
        
        # Get current correlation matrix
        correlation_matrix = self._calculate_correlation_matrix()
        if not correlation_matrix:
            return True
        
        symbol_base = symbol.split('/')[0]
        portfolio_value = self._get_portfolio_value()
        new_order_value = quantity * self._get_current_price(symbol)
        new_order_exposure = new_order_value / portfolio_value if portfolio_value > 0 else 0.0
        
        # Sum exposures of all highly correlated positions
        total_correlated_exposure = new_order_exposure
        correlated_assets = []
        for position in self.positions.values():
            pos_symbol_base = position.symbol.split('/')[0]
            if symbol_base in correlation_matrix and pos_symbol_base in correlation_matrix[symbol_base]:
                correlation = abs(correlation_matrix[symbol_base][pos_symbol_base])
                if correlation > 0.8:  # High correlation threshold
                    correlated_assets.append((pos_symbol_base, correlation))
                    total_correlated_exposure += position.market_value / portfolio_value if portfolio_value > 0 else 0.0
        
        if total_correlated_exposure > self.max_correlated_exposure:
            self._log_risk_event(
                RiskEventType.EXPOSURE_LIMIT_BREACH,
                symbol,
                f"Correlated exposure limit exceeded: total {total_correlated_exposure:.2f} (threshold {self.max_correlated_exposure:.2f}), correlated assets: {correlated_assets}",
                RiskLevel.HIGH
            )
            return False
        
        return True
    
    def _check_position_size_limits(self, signal) -> bool:
        """Check if order size is within limits."""
        # Extract signal data
        signal_data = self._extract_signal_data(signal)
        symbol = signal_data['symbol']
        quantity = signal_data['quantity']
        
        # Calculate portfolio value
        portfolio_value = self._get_portfolio_value()
        
        # Calculate order value
        order_value = quantity * self._get_current_price(symbol)
        
        # Check maximum order size
        max_order_value = portfolio_value * self.max_order_size
        # DEBUG LOGGING
        logger.debug(f"[RISK DEBUG] portfolio_value={portfolio_value}, max_order_size={self.max_order_size}, max_order_value={max_order_value}, order_value={order_value}")
        if order_value > max_order_value:
            self._log_risk_event(
                RiskEventType.ORDER_BLOCKED,
                symbol,
                f"Order value ${order_value:,.2f} exceeds max order size ${max_order_value:,.2f}",
                RiskLevel.HIGH
            )
            return False
        
        # Check risk per trade
        risk_amount = portfolio_value * self.risk_per_trade
        if order_value > risk_amount:
            self._log_risk_event(
                RiskEventType.ORDER_BLOCKED,
                symbol,
                f"Order value ${order_value:,.2f} exceeds risk per trade ${risk_amount:,.2f}",
                RiskLevel.MEDIUM
            )
            return False
        return True
    
    def _check_exposure_limits(self, signal) -> bool:
        """Check exposure limits for the asset."""
        # Extract signal data
        signal_data = self._extract_signal_data(signal)
        symbol = signal_data['symbol']
        quantity = signal_data['quantity']
        
        portfolio_value = self._get_portfolio_value()
        
        # Calculate current exposure to this asset
        current_exposure = self._get_asset_exposure(symbol)
        new_exposure = current_exposure + (quantity * self._get_current_price(symbol))
        
        # Check single asset exposure limit
        max_asset_exposure = portfolio_value * self.max_single_asset_exposure
        if new_exposure > max_asset_exposure:
            self._log_risk_event(
                RiskEventType.EXPOSURE_LIMIT_BREACH,
                symbol,
                f"Asset exposure ${new_exposure:,.2f} exceeds limit ${max_asset_exposure:,.2f}",
                RiskLevel.HIGH
            )
            return False
        
        # Check total exposure limit
        total_exposure = self._get_total_exposure()
        if total_exposure > portfolio_value * self.max_total_exposure:
            self._log_risk_event(
                RiskEventType.EXPOSURE_LIMIT_BREACH,
                symbol,
                f"Total exposure ${total_exposure:,.2f} exceeds limit",
                RiskLevel.CRITICAL
            )
            return False
        
        return True
    
    def _check_position_count_limits(self, signal) -> bool:
        """Check position count limits."""
        # Extract signal data
        signal_data = self._extract_signal_data(signal)
        symbol = signal_data['symbol']
        
        # Check total open positions
        if len(self.positions) >= self.max_open_positions:
            self._log_risk_event(
                RiskEventType.POSITION_LIMIT_BREACH,
                symbol,
                f"Maximum open positions ({self.max_open_positions}) reached",
                RiskLevel.MEDIUM
            )
            return False
        
        # Check positions per asset
        asset_positions = sum(1 for pos in self.positions.values() if pos.symbol == symbol)
        if asset_positions >= self.max_positions_per_asset:
            self._log_risk_event(
                RiskEventType.POSITION_LIMIT_BREACH,
                symbol,
                f"Maximum positions per asset ({self.max_positions_per_asset}) reached",
                RiskLevel.MEDIUM
            )
            return False
        
        return True
    
    def _check_drawdown_limits(self) -> bool:
        """Check drawdown limits."""
        if not self.portfolio_metrics:
            return True
        
        # Check daily drawdown
        if self.portfolio_metrics.daily_pnl_percentage < -self.max_daily_drawdown * 100:
            self._log_risk_event(
                RiskEventType.DRAWDOWN_BREACH,
                "PORTFOLIO",
                f"Daily drawdown {self.portfolio_metrics.daily_pnl_percentage:.2f}% exceeds limit",
                RiskLevel.CRITICAL
            )
            return False
        
        # Check total drawdown
        if self.portfolio_metrics.max_drawdown_percentage > self.max_total_drawdown * 100:
            self._log_risk_event(
                RiskEventType.DRAWDOWN_BREACH,
                "PORTFOLIO",
                f"Total drawdown {self.portfolio_metrics.max_drawdown_percentage:.2f}% exceeds limit",
                RiskLevel.CRITICAL
            )
            return False
        
        return True
    
    def _check_circuit_breakers(self) -> bool:
        """Check circuit breaker conditions."""
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self._pause_trading(f"Maximum consecutive losses ({self.max_consecutive_losses}) reached")
            return False
        
        # Check volatility (simplified - should use proper volatility calculation)
        if self.portfolio_metrics and abs(self.portfolio_metrics.daily_pnl_percentage) > self.volatility_threshold * 100:
            self._pause_trading(f"High volatility detected: {self.portfolio_metrics.daily_pnl_percentage:.2f}%")
            return False
        
        return True
    
    def _validate_risk_levels(self, signal) -> bool:
        """Validate stop-loss and take-profit levels."""
        # Extract signal data
        signal_data = self._extract_signal_data(signal)
        symbol = signal_data['symbol']
        side = signal_data['side']
        stop_loss = signal_data['stop_loss']
        
        # Ensure stop-loss is set
        if stop_loss is None:
            # Set default stop-loss
            current_price = self._get_current_price(symbol)
            if side == 'buy':
                stop_loss = current_price * (1 - self.default_stop_loss)
            else:
                stop_loss = current_price * (1 + self.default_stop_loss)
        
        # Validate stop-loss level
        current_price = self._get_current_price(symbol)
        if side == 'buy':
            if stop_loss >= current_price:
                return False
        else:
            if stop_loss <= current_price:
                return False
        
        return True
    
    def update_position(self, symbol: str, side: str, quantity: float, price: float):
        """Update position after trade execution."""
        position_id = f"{symbol}_{side}"
        
        # Update price history for correlation analysis
        self._update_price_history(symbol, price)
        
        if position_id in self.positions:
            # Update existing position
            position = self.positions[position_id]
            # Calculate new average price and quantity
            total_quantity = position.quantity + quantity
            total_value = (position.quantity * position.entry_price) + (quantity * price)
            position.entry_price = total_value / total_quantity
            position.quantity = total_quantity
            position.current_price = price
        else:
            # Create new position
            position = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                unrealized_pnl=0.0
            )
            self.positions[position_id] = position
        
        logger.info(f"Position updated: {position_id} - Qty: {position.quantity}, Avg Price: ${position.entry_price:.4f}")
    
    def close_position(self, symbol: str, side: str, quantity: float, price: float):
        """Close a position and calculate PnL. 'buy' = long, 'sell' = short."""
        position_id = f"{symbol}_{side}"
        
        # Update price history
        self._update_price_history(symbol, price)
        
        if position_id in self.positions:
            position = self.positions[position_id]
            
            # Calculate realized PnL
            if side == 'buy':  # Closing long position
                realized_pnl = (price - position.entry_price) * quantity
            else:  # Closing short position
                realized_pnl = (position.entry_price - price) * quantity
            
            # Update consecutive losses
            if realized_pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            # Remove position if fully closed
            if quantity >= position.quantity:
                del self.positions[position_id]
                logger.info(f"Position closed: {position_id} - PnL: ${realized_pnl:.2f}")
            else:
                # Partial close
                position.quantity -= quantity
                logger.info(f"Position partially closed: {position_id} - PnL: ${realized_pnl:.2f}")
            
            return realized_pnl
        
        return 0.0
    
    def update_portfolio_metrics(self, portfolio_value: float, daily_pnl: float):
        """Update portfolio risk metrics."""
        # Update peak value
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
        
        # Calculate drawdown
        max_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
        
        # Update daily PnL history
        self.daily_pnl_history.append(daily_pnl)
        
        # Calculate metrics
        total_pnl = portfolio_value - self.initial_portfolio_value
        total_pnl_percentage = (total_pnl / self.initial_portfolio_value) * 100 if self.initial_portfolio_value > 0 else 0
        daily_pnl_percentage = (daily_pnl / portfolio_value) * 100 if portfolio_value > 0 else 0
        
        # Calculate exposure by asset
        exposure_by_asset = self._calculate_asset_exposure(portfolio_value)
        
        # Calculate sector exposure
        exposure_by_sector = self._calculate_sector_exposure(portfolio_value)
        
        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix()
        
        # Calculate VaR (simplified)
        var_95 = self._calculate_var_95()
        
        # Calculate Sharpe ratio (simplified)
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        self.portfolio_metrics = PortfolioMetrics(
            total_value=portfolio_value,
            total_pnl=total_pnl,
            total_pnl_percentage=total_pnl_percentage,
            daily_pnl=daily_pnl,
            daily_pnl_percentage=daily_pnl_percentage,
            max_drawdown=max_drawdown,
            max_drawdown_percentage=max_drawdown * 100,
            exposure_by_asset=exposure_by_asset,
            exposure_by_sector=exposure_by_sector,
            correlation_matrix=correlation_matrix,
            var_95=var_95,
            sharpe_ratio=sharpe_ratio
        )
        
        # Log metrics to database
        self.db.log_portfolio_metrics(self.portfolio_metrics)
        
        # Check for drawdown breaches
        if max_drawdown > self.max_total_drawdown:
            self._log_risk_event(
                RiskEventType.DRAWDOWN_BREACH,
                "PORTFOLIO",
                f"Total drawdown {max_drawdown*100:.2f}% exceeds limit",
                RiskLevel.CRITICAL
            )
    
    def check_stop_losses(self) -> List[Tuple[str, str, float]]:
        """Check for stop-loss triggers and return list of positions to close."""
        positions_to_close = []
        
        try:
            if not self.position_manager:
                return positions_to_close
            
            open_positions = self.position_manager.get_open_positions()
            
            for position in open_positions:
                if position.stop_loss is None:
                    continue
                
                # Check if stop-loss is triggered
                if position.side == PositionSide.LONG and position.current_price <= position.stop_loss:
                    positions_to_close.append((position.id, 'sell', position.quantity))
                    self._log_risk_event(
                        RiskEventType.STOP_LOSS_TRIGGERED,
                        position.symbol,
                        f"Stop-loss triggered for {position.symbol} at {position.current_price}",
                        RiskLevel.HIGH,
                        {'position_id': position.id, 'stop_loss': position.stop_loss, 'current_price': position.current_price}
                    )
                elif position.side == PositionSide.SHORT and position.current_price >= position.stop_loss:
                    positions_to_close.append((position.id, 'buy', position.quantity))
                    self._log_risk_event(
                        RiskEventType.STOP_LOSS_TRIGGERED,
                        position.symbol,
                        f"Stop-loss triggered for {position.symbol} at {position.current_price}",
                        RiskLevel.HIGH,
                        {'position_id': position.id, 'stop_loss': position.stop_loss, 'current_price': position.current_price}
                    )
            
            return positions_to_close
            
        except Exception as e:
            logger.error(f"Failed to check stop-losses: {e}")
            return []
    
    def check_take_profits(self) -> List[Tuple[str, str, float]]:
        """Check for take-profit triggers and return list of positions to close."""
        positions_to_close = []
        
        try:
            if not self.position_manager:
                return positions_to_close
            
            open_positions = self.position_manager.get_open_positions()
            
            for position in open_positions:
                if position.take_profit is None:
                    continue
                
                # Check if take-profit is triggered
                if position.side == PositionSide.LONG and position.current_price >= position.take_profit:
                    positions_to_close.append((position.id, 'sell', position.quantity))
                    self._log_risk_event(
                        RiskEventType.TAKE_PROFIT_TRIGGERED,
                        position.symbol,
                        f"Take-profit triggered for {position.symbol} at {position.current_price}",
                        RiskLevel.LOW,
                        {'position_id': position.id, 'take_profit': position.take_profit, 'current_price': position.current_price}
                    )
                elif position.side == PositionSide.SHORT and position.current_price <= position.take_profit:
                    positions_to_close.append((position.id, 'buy', position.quantity))
                    self._log_risk_event(
                        RiskEventType.TAKE_PROFIT_TRIGGERED,
                        position.symbol,
                        f"Take-profit triggered for {position.symbol} at {position.current_price}",
                        RiskLevel.LOW,
                        {'position_id': position.id, 'take_profit': position.take_profit, 'current_price': position.current_price}
                    )
            
            return positions_to_close
            
        except Exception as e:
            logger.error(f"Failed to check take-profits: {e}")
            return []
    
    async def execute_forced_exits(self):
        """Execute forced exits for risk management."""
        try:
            if not self.order_manager:
                logger.warning("Order manager not connected for forced exits")
                return
            
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
                
        except Exception as e:
            logger.error(f"Failed to execute forced exits: {e}")
    
    async def _submit_exit_order(self, position_id: str, side: str, quantity: float, reason: str):
        """Submit exit order for risk management."""
        try:
            if not self.position_manager:
                return
            
            position = self.position_manager.get_position_by_id(position_id)
            if not position:
                logger.warning(f"Position {position_id} not found for exit order")
                return
            
            # Create exit signal
            from execution.order_manager import TradeSignal, OrderSide, OrderType
            
            exit_signal = TradeSignal(
                symbol=position.symbol,
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                quantity=quantity,
                order_type=OrderType.MARKET,  # Use market order for immediate execution
                strategy_name=f"risk_manager_{reason}"
            )
            
            # Submit order
            order = await self.order_manager.submit_order(exit_signal)
            logger.info(f"Exit order submitted for {position_id}: {side} {quantity} {position.symbol} ({reason})")
            
        except Exception as e:
            logger.error(f"Failed to submit exit order for {position_id}: {e}")
    
    async def _execute_circuit_breaker(self):
        """Execute circuit breaker - close all positions."""
        try:
            if not self.position_manager or not self.order_manager:
                logger.warning("Position or order manager not connected for circuit breaker")
                return
            
            open_positions = self.position_manager.get_open_positions()
            
            for position in open_positions:
                side = 'buy' if position.side == PositionSide.SHORT else 'sell'
                await self._submit_exit_order(position.id, side, position.quantity, "circuit_breaker")
            
            # Pause trading
            self._pause_trading("Circuit breaker triggered - all positions closed")
            
            self._log_risk_event(
                RiskEventType.CIRCUIT_BREAKER_TRIGGERED,
                "ALL",
                "Circuit breaker triggered - all positions closed",
                RiskLevel.CRITICAL,
                {'positions_closed': len(open_positions)}
            )
            
        except Exception as e:
            logger.error(f"Failed to execute circuit breaker: {e}")
    
    def _pause_trading(self, reason: str):
        """Pause trading due to risk event."""
        self.trading_paused = True
        self.pause_reason = reason
        logger.warning(f"Trading paused: {reason}")
        
        self._log_risk_event(
            RiskEventType.CIRCUIT_BREAKER_TRIGGERED,
            "SYSTEM",
            f"Trading paused: {reason}",
            RiskLevel.CRITICAL
        )
    
    def resume_trading(self):
        """Resume trading after risk conditions are resolved."""
        self.trading_paused = False
        self.pause_reason = ""
        logger.info("Trading resumed")
    
    def _log_risk_event(self, event_type: RiskEventType, symbol: str, message: str, risk_level: RiskLevel, details: Dict[str, Any] = None):
        """Log a risk event to memory and database."""
        event = RiskEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            symbol=symbol,
            message=message,
            risk_level=risk_level,
            timestamp=datetime.now(),
            details=details or {}
        )
        
        self.risk_events.append(event)
        
        # Log to database
        self.db.log_risk_event(event)
        
        # Log based on risk level
        if risk_level == RiskLevel.CRITICAL:
            logger.critical(f"RISK EVENT: {message}")
        elif risk_level == RiskLevel.HIGH:
            logger.error(f"RISK EVENT: {message}")
        elif risk_level == RiskLevel.MEDIUM:
            logger.warning(f"RISK EVENT: {message}")
        else:
            logger.info(f"RISK EVENT: {message}")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        return {
            "trading_status": {
                "paused": self.trading_paused,
                "pause_reason": self.pause_reason,
                "consecutive_losses": self.consecutive_losses
            },
            "portfolio_metrics": self.portfolio_metrics.__dict__ if self.portfolio_metrics else None,
            "positions": {
                pos_id: {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "pnl_percentage": pos.pnl_percentage,
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit
                }
                for pos_id, pos in self.positions.items()
            },
            "risk_limits": {
                "max_position_size": self.max_position_size,
                "max_order_size": self.max_order_size,
                "risk_per_trade": self.risk_per_trade,
                "max_total_exposure": self.max_total_exposure,
                "max_single_asset_exposure": self.max_single_asset_exposure,
                "max_daily_drawdown": self.max_daily_drawdown,
                "max_total_drawdown": self.max_total_drawdown,
                "max_open_positions": self.max_open_positions,
                "max_positions_per_asset": self.max_positions_per_asset,
                "max_consecutive_losses": self.max_consecutive_losses,
                "volatility_threshold": self.volatility_threshold
            },
            "recent_risk_events": [
                {
                    "event_type": event.event_type.value,
                    "symbol": event.symbol,
                    "message": event.message,
                    "risk_level": event.risk_level.value,
                    "timestamp": event.timestamp.isoformat()
                }
                for event in self.risk_events[-10:]  # Last 10 events
            ],
            "sector_classification": self.sector_classification,
            "correlation_matrix": self._calculate_correlation_matrix()
        }
    
    # Helper methods (simplified implementations)
    def _get_asset_exposure(self, symbol: str) -> float:
        """Get current exposure to specific asset."""
        exposure = 0.0
        for position in self.positions.values():
            if position.symbol == symbol:
                exposure += position.market_value
        return exposure
    
    def _get_total_exposure(self) -> float:
        """Get total portfolio exposure."""
        try:
            if self.position_manager:
                # Handle both object and dict position_manager
                if hasattr(self.position_manager, 'get_portfolio_metrics'):
                    portfolio_metrics = self.position_manager.get_portfolio_metrics()
                    return portfolio_metrics.total_market_value
                elif isinstance(self.position_manager, dict):
                    return self.position_manager.get('total_market_value', 0.0)
                else:
                    logger.warning(f"Unexpected position_manager type: {type(self.position_manager)}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get total exposure: {e}")
            return 0.0

    def _calculate_asset_exposure(self, portfolio_value: float) -> Dict[str, float]:
        """Calculate exposure by asset as percentage of portfolio."""
        exposure = {}
        for position in self.positions.values():
            if position.symbol not in exposure:
                exposure[position.symbol] = 0.0
            exposure[position.symbol] += position.market_value / portfolio_value
        return exposure
    
    def _calculate_var_95(self) -> float:
        """Calculate Value at Risk (95% confidence)."""
        if len(self.daily_pnl_history) < 2:
            return 0.0
        
        # Simplified VaR calculation
        returns = list(self.daily_pnl_history)
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = variance ** 0.5
        
        # 95% VaR = mean - 1.645 * std_dev
        return mean_return - 1.645 * std_dev
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        if len(self.daily_pnl_history) < 2:
            return 0.0
        
        returns = list(self.daily_pnl_history)
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return 0.0
        
        # Assuming risk-free rate of 0 for crypto
        return mean_return / std_dev

    def validate_signal(self, signal: Dict, current_positions: Dict) -> bool:
        """
        Validate a trading signal against risk management rules.
        
        Args:
            signal: Trading signal dictionary containing symbol, side, etc.
            current_positions: Current portfolio positions
            
        Returns:
            bool: True if signal passes risk checks, False otherwise
        """
        try:
            # Extract signal information
            symbol = signal.get('symbol', '')
            side = signal.get('side', '')
            confidence = signal.get('confidence', 0.5)
            
            if not symbol or not side:
                logger.warning(f"Invalid signal format: {signal}")
                return False
            
            # Check if trading is allowed
            if not self.is_trading_allowed():
                logger.warning("Trading is currently paused by risk manager")
                return False
            
            # Check basic signal validity
            if side not in ['buy', 'sell']:
                logger.warning(f"Invalid signal side: {side}")
                return False
            
            # Check confidence threshold
            min_confidence = self.risk_config.get('MIN_SIGNAL_CONFIDENCE', 0.3)
            if confidence < min_confidence:
                logger.debug(f"Signal confidence too low: {confidence} < {min_confidence}")
                return False
            
            # Check position size limits
            if not self._check_position_size_limits(signal):
                logger.warning(f"Position size limit check failed for {symbol}")
                return False
            
            # Check exposure limits
            if not self._check_exposure_limits(signal):
                logger.warning(f"Exposure limit check failed for {symbol}")
                return False
            
            # Check position count limits
            if not self._check_position_count_limits(signal):
                logger.warning(f"Position count limit check failed for {symbol}")
                return False
            
            # Check drawdown limits
            if not self._check_drawdown_limits():
                logger.warning("Drawdown limit check failed")
                return False
            
            # Check circuit breakers
            if not self._check_circuit_breakers():
                logger.warning("Circuit breaker check failed")
                return False
            
            # Check risk levels
            if not self._validate_risk_levels(signal):
                logger.warning(f"Risk level validation failed for {symbol}")
                return False
            
            logger.debug(f"Signal validation passed for {symbol} {side}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    def get_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        return self.positions.copy()
    
    def get_risk_events(self, limit: int = 100) -> List[RiskEvent]:
        """Get recent risk events."""
        return self.risk_events[-limit:]
    
    def get_historical_risk_events(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get historical risk events from database."""
        return self.db.get_risk_events(limit=100, days=days)
    
    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        return not self.trading_paused
    
    def get_risk_level(self) -> RiskLevel:
        """Get current risk level based on portfolio metrics."""
        if not self.portfolio_metrics:
            return RiskLevel.LOW
        
        # Determine risk level based on drawdown and volatility
        if self.portfolio_metrics.max_drawdown_percentage > 10:
            return RiskLevel.CRITICAL
        elif self.portfolio_metrics.max_drawdown_percentage > 5:
            return RiskLevel.HIGH
        elif self.portfolio_metrics.max_drawdown_percentage > 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def set_position_manager(self, position_manager):
        """Set position manager reference."""
        self.position_manager = position_manager
        logger.info("Position manager connected to risk manager")
    
    def set_order_manager(self, order_manager):
        """Set order manager reference."""
        self.order_manager = order_manager
        logger.info("Order manager connected to risk manager")
    
    def update_on_trade(self, symbol: str, side: str, quantity: float, price: float, order_id: str):
        """
        Update risk manager when a trade occurs.
        This method is called by the order manager when an order is filled.
        
        Args:
            symbol: Trading symbol
            side: Trade side ('buy' or 'sell')
            quantity: Trade quantity
            price: Trade price
            order_id: Order ID
        """
        try:
            # Update position tracking
            self.update_position(symbol, side, quantity, price)
            
            # Check for risk events after trade
            self._check_post_trade_risk_events(symbol, side, quantity, price)
            
            # Update portfolio metrics
            if self.position_manager:
                try:
                    if hasattr(self.position_manager, 'get_portfolio_metrics'):
                        portfolio_metrics = self.position_manager.get_portfolio_metrics()
                        self.update_portfolio_metrics(portfolio_metrics.total_market_value, portfolio_metrics.daily_pnl)
                    elif isinstance(self.position_manager, dict):
                        # Use fallback values if position_manager is a dict
                        total_market_value = self.position_manager.get('total_market_value', 0.0)
                        daily_pnl = self.position_manager.get('daily_pnl', 0.0)
                        self.update_portfolio_metrics(total_market_value, daily_pnl)
                    else:
                        logger.warning(f"Unexpected position_manager type: {type(self.position_manager)}")
                except Exception as e:
                    logger.error(f"Failed to update portfolio metrics: {e}")
            
            logger.info(f"Risk manager updated on trade: {side} {quantity} {symbol} @ {price}")
            
        except Exception as e:
            logger.error(f"Failed to update risk manager on trade: {e}")
    
    def _check_post_trade_risk_events(self, symbol: str, side: str, quantity: float, price: float):
        """Check for risk events after a trade."""
        try:
            # Check position limits
            if not self._check_position_size_limits_after_trade(symbol, side, quantity):
                self._log_risk_event(
                    RiskEventType.POSITION_LIMIT_BREACH,
                    symbol,
                    f"Position size limit breached after {side} {quantity} {symbol}",
                    RiskLevel.HIGH,
                    {'side': side, 'quantity': quantity, 'price': price}
                )
            
            # Check exposure limits
            if not self._check_exposure_limits_after_trade(symbol, side, quantity, price):
                self._log_risk_event(
                    RiskEventType.EXPOSURE_LIMIT_BREACH,
                    symbol,
                    f"Exposure limit breached after {side} {quantity} {symbol}",
                    RiskLevel.HIGH,
                    {'side': side, 'quantity': quantity, 'price': price}
                )
            
            # Check drawdown limits
            if not self._check_drawdown_limits():
                self._log_risk_event(
                    RiskEventType.DRAWDOWN_BREACH,
                    symbol,
                    f"Drawdown limit breached after {side} {quantity} {symbol}",
                    RiskLevel.CRITICAL,
                    {'side': side, 'quantity': quantity, 'price': price}
                )
                
        except Exception as e:
            logger.error(f"Failed to check post-trade risk events: {e}")
    
    def _check_position_size_limits_after_trade(self, symbol: str, side: str, quantity: float) -> bool:
        """Check position size limits after a trade."""
        try:
            # Get current position size for this symbol
            current_position_size = self._get_position_size(symbol)
            
            # Check against limits
            max_position_size = self.risk_config.get('MAX_POSITION_SIZE', 1000000)  # $1M default
            if current_position_size > max_position_size:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check position size limits: {e}")
            return False
    
    def _check_exposure_limits_after_trade(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        """Check exposure limits after a trade."""
        try:
            # Calculate new exposure
            trade_value = quantity * price
            current_exposure = self._get_total_exposure()
            new_exposure = current_exposure + trade_value
            
            # Check against limits
            max_exposure = self.risk_config.get('MAX_TOTAL_EXPOSURE', 5000000)  # $5M default
            if new_exposure > max_exposure:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check exposure limits: {e}")
            return False
    
    def _get_position_size(self, symbol: str) -> float:
        """Get current position size for a symbol."""
        try:
            if self.position_manager:
                # Handle both object and dict position_manager
                if hasattr(self.position_manager, 'get_positions_by_symbol'):
                    positions = self.position_manager.get_positions_by_symbol(symbol)
                    return sum(pos.market_value for pos in positions)
                elif isinstance(self.position_manager, dict):
                    # Fallback to local positions tracking
                    return sum(pos.market_value for pos in self.positions.values() if pos.symbol == symbol)
                else:
                    logger.warning(f"Unexpected position_manager type: {type(self.position_manager)}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get position size for {symbol}: {e}")
            return 0.0
