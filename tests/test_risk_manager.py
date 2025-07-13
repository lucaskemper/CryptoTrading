import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
import os
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from execution.risk_manager import (
    RiskManager, RiskLevel, RiskEventType, Position, 
    PortfolioMetrics, RiskEvent, RiskDatabase
)
from execution.order_manager import TradeSignal, OrderSide, OrderType
from src.execution.position_manager import PositionSide


class MockPortfolioManager:
    """Mock portfolio manager for testing."""
    
    def __init__(self, total_value: float = 10000.0):
        self.total_value = total_value
    
    def get_total_value(self) -> float:
        return self.total_value


class MockMarketDataManager:
    """Mock market data manager for testing."""
    
    def __init__(self):
        self.prices = {
            "ETH/USDT": 2000.0,
            "BTC/USDT": 50000.0,
            "SOL/USDT": 100.0,
            "USDT/USDT": 1.0
        }
    
    def get_latest_price(self, symbol: str) -> float:
        return self.prices.get(symbol, 100.0)


class TestRiskManagerEnhanced(unittest.TestCase):
    """Enhanced test cases for RiskManager with portfolio integration and advanced features."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_risk_events.db")
        
        # Mock config
        with patch('execution.risk_manager.config') as mock_config:
            mock_config.get.return_value = {
                "max_position_size": 0.1,
                "max_order_size": 0.05,
                "risk_per_trade": 0.02,
                "stop_loss_percentage": 0.05,
                "take_profit_percentage": 0.1,
                "max_total_exposure": 0.8,
                "max_single_asset_exposure": 0.3,
                "max_correlated_exposure": 0.5,
                "max_daily_drawdown": 0.05,
                "max_total_drawdown": 0.15,
                "max_open_positions": 10,
                "max_positions_per_asset": 3,
                "max_consecutive_losses": 5,
                "volatility_threshold": 0.1
            }
            
            # Create mock managers
            self.portfolio_manager = MockPortfolioManager(10000.0)
            self.market_data_manager = MockMarketDataManager()
            
            # Initialize risk manager with dependencies
            self.risk_manager = RiskManager(
                portfolio_manager=self.portfolio_manager,
                market_data_manager=self.market_data_manager,
                db_path=self.db_path
            )
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_portfolio_integration(self):
        """Test portfolio manager integration."""
        # Test portfolio value retrieval
        portfolio_value = self.risk_manager._get_portfolio_value()
        self.assertEqual(portfolio_value, 10000.0)
        
        # Test current price retrieval
        eth_price = self.risk_manager._get_current_price("ETH/USDT")
        self.assertEqual(eth_price, 2000.0)
        
        # Test fallback when managers are not available
        risk_manager_no_deps = RiskManager()
        fallback_value = risk_manager_no_deps._get_portfolio_value()
        self.assertEqual(fallback_value, 100000.0)  # Initial value from config
    
    def test_sector_classification(self):
        """Test sector classification functionality."""
        # Test sector exposure calculation
        self.risk_manager.update_position("ETH/USDT", "buy", 2.0, 2000.0)
        self.risk_manager.update_position("SOL/USDT", "buy", 10.0, 100.0)
        
        portfolio_value = self.risk_manager._get_portfolio_value()
        sector_exposure = self.risk_manager._calculate_sector_exposure(portfolio_value)
        
        # ETH and SOL should both be Layer1
        self.assertIn("Layer1", sector_exposure)
        self.assertGreater(sector_exposure["Layer1"], 0)
    
    def test_correlation_analysis(self):
        """Test correlation analysis functionality."""
        # Add price history for correlation analysis
        for i in range(20):  # Need at least 10 data points
            self.risk_manager._update_price_history("ETH/USDT", 2000.0 + i)
            self.risk_manager._update_price_history("BTC/USDT", 50000.0 + i * 10)
        
        correlation_matrix = self.risk_manager._calculate_correlation_matrix()
        
        # Should have correlation data
        self.assertIn("ETH/USDT", correlation_matrix)
        self.assertIn("BTC/USDT", correlation_matrix)
        
        # Correlation with self should be 1.0
        self.assertEqual(correlation_matrix["ETH/USDT"]["ETH/USDT"], 1.0)
    
    def test_correlation_limits(self):
        """Test correlation-based exposure limits."""
        # Lower the max correlated exposure for this test
        self.risk_manager.max_correlated_exposure = 0.05
        # Increase order size and risk per trade limits for this test
        self.risk_manager.max_order_size = 0.5
        self.risk_manager.risk_per_trade = 0.5
        
        # Add highly correlated price history
        for i in range(20):
            self.risk_manager._update_price_history("ETH/USDT", 2000.0 + i)
            self.risk_manager._update_price_history("BTC/USDT", 50000.0 + i * 25)  # Highly correlated
        
        # Add existing position
        self.risk_manager.update_position("ETH/USDT", "buy", 2.0, 2000.0)
        
        # Try to add correlated position with a size that triggers correlation but not size/risk checks
        signal = TradeSignal(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.04,  # Increased to ensure correlated exposure is breached (0.04 BTC * $50k = $2,000 = 20% of $10k)
            order_type=OrderType.MARKET,
            strategy_name="test"
        )
        
        is_allowed, reason = self.risk_manager.check_order_risk(signal)
        # Should be blocked due to correlation limits
        self.assertFalse(is_allowed)
        self.assertIn("correlation", reason.lower())
    
    def test_database_logging(self):
        """Test database logging functionality."""
        # Log a risk event
        self.risk_manager._log_risk_event(
            RiskEventType.ORDER_BLOCKED,
            "ETH/USDT",
            "Test database logging",
            RiskLevel.HIGH,
            details={"test": "data"}
        )
        
        # Check that event was logged to database
        historical_events = self.risk_manager.get_historical_risk_events(days=1)
        self.assertGreater(len(historical_events), 0)
        
        # Verify event details
        event = historical_events[0]
        self.assertEqual(event["event_type"], "order_blocked")
        self.assertEqual(event["symbol"], "ETH/USDT")
        self.assertEqual(event["message"], "Test database logging")
        self.assertEqual(event["risk_level"], "high")
        self.assertEqual(event["details"], {"test": "data"})
    
    def test_portfolio_metrics_logging(self):
        """Test portfolio metrics database logging."""
        self.risk_manager.set_initial_portfolio_value(10000.0)
        self.risk_manager.update_portfolio_metrics(10500.0, 500.0)
        
        # Metrics should be logged to database
        # (We can't easily test this without database inspection, but we can verify no errors)
        self.assertIsNotNone(self.risk_manager.portfolio_metrics)
    
    def test_rapid_drawdown_scenario(self):
        """Test rapid drawdown scenario."""
        self.risk_manager.set_initial_portfolio_value(10000.0)
        
        # Simulate rapid losses
        self.risk_manager.update_portfolio_metrics(9500.0, -500.0)  # -5% daily
        self.risk_manager.update_portfolio_metrics(9000.0, -500.0)  # -5% daily
        self.risk_manager.update_portfolio_metrics(8500.0, -500.0)  # -5% daily
        
        # Check that trading is paused due to drawdown
        signal = TradeSignal(
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            order_type=OrderType.MARKET,
            strategy_name="test"
        )
        
        is_allowed, reason = self.risk_manager.check_order_risk(signal)
        self.assertFalse(is_allowed)
        self.assertIn("drawdown", reason.lower())
    
    def test_partial_position_close(self):
        """Test partial position closing."""
        # Create position
        self.risk_manager.update_position("ETH/USDT", "buy", 2.0, 2000.0)
        
        # Partial close
        pnl = self.risk_manager.close_position("ETH/USDT", "buy", 1.0, 2200.0)
        
        # Should have profit
        self.assertEqual(pnl, 200.0)  # (2200 - 2000) * 1.0
        
        # Position should still exist with remaining quantity
        position = self.risk_manager.positions.get("ETH/USDT_buy")
        self.assertIsNotNone(position)
        self.assertEqual(position.quantity, 1.0)
        
        # Full close
        pnl = self.risk_manager.close_position("ETH/USDT", "buy", 1.0, 2300.0)
        self.assertEqual(pnl, 300.0)  # (2300 - 2000) * 1.0
        
        # Position should be removed
        self.assertNotIn("ETH/USDT_buy", self.risk_manager.positions)
    
    def test_volatile_market_scenario(self):
        """Test volatile market scenario."""
        self.risk_manager.set_initial_portfolio_value(10000.0)
        
        # Lower volatility threshold for this test
        self.risk_manager.volatility_threshold = 0.05  # 5% instead of default
        
        # Simulate high volatility with positive daily PnL to avoid drawdown
        self.risk_manager.update_portfolio_metrics(12000.0, 2000.0)  # +20% daily
        self.risk_manager.update_portfolio_metrics(13000.0, 1000.0)  # +7.69% daily, still volatile but positive
        
        # Check that circuit breaker is triggered for volatility
        signal = TradeSignal(
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            order_type=OrderType.MARKET,
            strategy_name="test"
        )
        
        is_allowed, reason = self.risk_manager.check_order_risk(signal)
        self.assertFalse(is_allowed)
        self.assertIn("circuit breaker", reason.lower())
    
    def test_consecutive_losses_circuit_breaker(self):
        """Test consecutive losses circuit breaker."""
        # Simulate consecutive losses
        for i in range(5):  # Max consecutive losses
            self.risk_manager.consecutive_losses = i + 1
            
            signal = TradeSignal(
                symbol="ETH/USDT",
                side=OrderSide.BUY,
                quantity=0.1,
                order_type=OrderType.MARKET,
                strategy_name="test"
            )
            
            is_allowed, reason = self.risk_manager.check_order_risk(signal)
            
            if i < 4:  # Should still be allowed
                self.assertTrue(is_allowed)
            else:  # Should be blocked
                # Accept either 'circuit breaker triggered' or 'consecutive' in the message
                self.assertTrue(
                    "consecutive" in reason.lower() or "circuit breaker" in reason.lower(),
                    f"Expected 'consecutive' or 'circuit breaker' in reason, got: {reason}"
                )
    
    def test_circuit_breaker_resume(self):
        """Test circuit breaker resume functionality."""
        # Trigger circuit breaker
        self.risk_manager.consecutive_losses = 5
        self.risk_manager._pause_trading("Test pause")
        
        self.assertTrue(self.risk_manager.trading_paused)
        
        # Resume trading
        self.risk_manager.resume_trading()
        self.assertFalse(self.risk_manager.trading_paused)
        self.assertEqual(self.risk_manager.pause_reason, "")
    
    def test_edge_case_exact_stop_loss(self):
        """Test edge case where price exactly hits stop-loss."""
        # Mock position manager
        mock_position_manager = Mock()
        mock_position = Mock()
        mock_position.id = "test_position_id"
        mock_position.symbol = "ETH/USDT"
        mock_position.side = PositionSide.LONG  # Use Enum to match risk manager expectation
        mock_position.quantity = 2.0
        mock_position.current_price = 1900.0
        mock_position.stop_loss = 1900.0
        mock_position.take_profit = None
        
        # Ensure the mock position manager returns our mock position
        mock_position_manager.get_open_positions.return_value = [mock_position]
        self.risk_manager.position_manager = mock_position_manager
        
        # Test stop-loss check
        positions_to_close = self.risk_manager.check_stop_losses()
    
        # Should trigger stop-loss (current_price <= stop_loss for LONG position)
        self.assertEqual(len(positions_to_close), 1)
        position_id, side, quantity = positions_to_close[0]
        self.assertEqual(position_id, "test_position_id")
        self.assertEqual(side, "sell")
        self.assertEqual(quantity, 2.0)
    
    def test_edge_case_slippage_handling(self):
        """Test edge case handling with price slippage."""
        # Create position
        self.risk_manager.update_position("ETH/USDT", "buy", 2.0, 2000.0)
        
        # Close with slippage (price moved against us)
        pnl = self.risk_manager.close_position("ETH/USDT", "buy", 2.0, 1950.0)  # 2.5% slippage
        
        # Should have loss
        self.assertEqual(pnl, -100.0)  # (1950 - 2000) * 2.0
        
        # Consecutive losses should be incremented
        self.assertEqual(self.risk_manager.consecutive_losses, 1)
    
    def test_stress_test_multiple_positions(self):
        """Stress test with multiple positions and rapid updates."""
        # Add multiple positions
        positions = [
            ("ETH/USDT", "buy", 1.0, 2000.0),
            ("BTC/USDT", "buy", 0.01, 50000.0),
            ("SOL/USDT", "sell", 5.0, 100.0),
        ]
        
        for symbol, side, quantity, price in positions:
            self.risk_manager.update_position(symbol, side, quantity, price)
        
        # Verify all positions exist
        self.assertEqual(len(self.risk_manager.positions), 3)
        
        # Rapid portfolio updates
        for i in range(10):
            portfolio_value = 10000.0 + i * 100
            daily_pnl = 100.0 if i % 2 == 0 else -50.0
            self.risk_manager.update_portfolio_metrics(portfolio_value, daily_pnl)
        
        # Should still be functional
        self.assertIsNotNone(self.risk_manager.portfolio_metrics)
        self.assertEqual(len(self.risk_manager.positions), 3)
    
    def test_risk_report_enhanced(self):
        """Test enhanced risk report with new features."""
        # Add some test data
        self.risk_manager.set_initial_portfolio_value(10000.0)
        self.risk_manager.update_position("ETH/USDT", "buy", 2.0, 2000.0)
        self.risk_manager.update_portfolio_metrics(10500.0, 500.0)
        
        report = self.risk_manager.get_risk_report()
        
        # Check enhanced features
        self.assertIn("sector_classification", report)
        self.assertIn("correlation_matrix", report)
        self.assertIsInstance(report["sector_classification"], dict)
        self.assertIsInstance(report["correlation_matrix"], dict)
    
    def test_historical_risk_events(self):
        """Test historical risk events retrieval."""
        # Log multiple events
        for i in range(5):
            self.risk_manager._log_risk_event(
                RiskEventType.ORDER_BLOCKED,
                f"ASSET{i}/USDT",
                f"Test event {i}",
                RiskLevel.MEDIUM
            )
        
        # Get historical events
        historical_events = self.risk_manager.get_historical_risk_events(days=1)
        
        # Should have events
        self.assertGreaterEqual(len(historical_events), 5)
        
        # Check event structure
        for event in historical_events:
            self.assertIn("id", event)
            self.assertIn("event_type", event)
            self.assertIn("symbol", event)
            self.assertIn("message", event)
            self.assertIn("risk_level", event)
            self.assertIn("timestamp", event)
    
    def test_database_error_handling(self):
        """Test database error handling."""
        # Create risk manager with invalid database path
        risk_manager_bad_db = RiskManager(db_path="/invalid/path/risk.db")
        
        # Should not crash when logging events
        risk_manager_bad_db._log_risk_event(
            RiskEventType.ORDER_BLOCKED,
            "ETH/USDT",
            "Test error handling",
            RiskLevel.HIGH
        )
        
        # Should return empty list for historical events
        events = risk_manager_bad_db.get_historical_risk_events()
        self.assertEqual(events, [])


if __name__ == '__main__':
    unittest.main() 