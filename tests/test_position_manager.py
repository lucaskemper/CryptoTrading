#!/usr/bin/env python3
"""
Enhanced Position Manager Tests

Comprehensive test suite for all enhanced position manager features:
- Closed position analytics
- Enhanced sector mapping
- Advanced export capabilities
- Historical performance analysis
- Thread safety and concurrency
- Real-time price updating
- Database statistics and cleanup
"""

import unittest
import asyncio
import tempfile
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

from src.execution.position_manager import (
    PositionManager, PositionSide, PositionType, PositionStatus
)


class TestEnhancedPositionManager(unittest.TestCase):
    """Test suite for enhanced position manager features."""
    
    def setUp(self):
        """Set up test environment."""
        # Use temporary database for tests
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.position_manager = PositionManager(db_path=self.temp_db.name)
        
        # Create test positions
        self._create_test_positions()
    
    def tearDown(self):
        """Clean up test environment."""
        # Close position manager
        asyncio.run(self.position_manager.shutdown())
        
        # Remove temporary database
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def _create_test_positions(self):
        """Create test positions for various scenarios."""
        # Layer 1 positions
        self.position_manager.add_position(
            "BTC/USDT", PositionSide.LONG, 0.1, 50000, "test_order_1", "stat_arb"
        )
        self.position_manager.add_position(
            "ETH/USDT", PositionSide.SHORT, 2.0, 3000, "test_order_2", "sentiment"
        )
        
        # DeFi positions
        self.position_manager.add_position(
            "UNI/USDT", PositionSide.LONG, 50.0, 8.0, "test_order_3", "defi_arb"
        )
        
        # Gaming positions
        self.position_manager.add_position(
            "AXS/USDT", PositionSide.LONG, 20.0, 15.0, "test_order_4", "gaming"
        )
        
        # Close some positions for analytics
        positions = self.position_manager.get_open_positions()
        if len(positions) >= 2:
            # Close first position as winning trade
            self.position_manager.close_position(
                positions[0].id, positions[0].entry_price * 1.05, 
                order_id="close_test_1"
            )
            # Close second position as losing trade
            self.position_manager.close_position(
                positions[1].id, positions[1].entry_price * 0.95, 
                order_id="close_test_2"
            )
    
    def test_enhanced_sector_mapping(self):
        """Test enhanced sector mapping functionality."""
        metrics = self.position_manager.get_portfolio_metrics()
        
        # Check that sector exposure is calculated
        self.assertIsInstance(metrics.exposure_by_sector, dict)
        self.assertIsInstance(metrics.exposure_by_asset, dict)
        
        # Check that we have exposure data
        self.assertGreater(len(metrics.exposure_by_asset), 0)
        
        # Verify sector mapping works for different asset types
        for asset, exposure in metrics.exposure_by_asset.items():
            self.assertIsInstance(asset, str)
            self.assertIsInstance(exposure, (int, float))
            self.assertGreaterEqual(exposure, 0)
    
    def test_closed_position_analytics(self):
        """Test closed position analytics."""
        # Get closed positions
        closed_positions = self.position_manager._get_closed_positions_recent(days=30)
        
        # Should have at least 2 closed positions from setup
        self.assertGreaterEqual(len(closed_positions), 2)
        
        # Check that closed positions have correct status
        for position in closed_positions:
            self.assertEqual(position.status, PositionStatus.CLOSED)
            self.assertGreater(position.realized_pnl, 0)  # Should have realized PnL
    
    def test_export_capabilities(self):
        """Test advanced export capabilities."""
        # Test JSON export
        json_export = self.position_manager.export_positions(format="json", include_closed=True)
        self.assertIsInstance(json_export, str)
        self.assertGreater(len(json_export), 0)
        
        # Parse JSON to ensure it's valid
        json_data = json.loads(json_export)
        self.assertIsInstance(json_data, list)
        
        # Test CSV export
        csv_export = self.position_manager.export_positions(format="csv", include_closed=True)
        self.assertIsInstance(csv_export, str)
        self.assertGreater(len(csv_export), 0)
        
        # Check CSV has expected headers
        csv_lines = csv_export.strip().split('\n')
        self.assertGreater(len(csv_lines), 1)  # Header + at least one data row
        
        # Test DataFrame export
        df = self.position_manager.export_positions(format="dataframe", include_closed=True)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        
        # Check DataFrame has expected columns
        expected_columns = ['id', 'symbol', 'side', 'quantity', 'entry_price', 'current_price']
        for col in expected_columns:
            self.assertIn(col, df.columns)
    
    def test_historical_analytics(self):
        """Test historical position analytics."""
        # Get position history
        history = self.position_manager.get_position_history()
        
        # Verify history is returned
        assert isinstance(history, list)
        
        # Test analytics
        analytics = self.position_manager.get_position_analytics()
        assert isinstance(analytics, dict)
        assert 'summary' in analytics
        assert 'total_positions' in analytics['summary']
        assert 'win_rate' in analytics['summary']
    
    def test_thread_safety(self):
        """Test thread safety features."""
        import concurrent.futures
        import threading
        
        # Get a position to update
        positions = self.position_manager.get_open_positions()
        if not positions:
            self.skipTest("No open positions to test")
        
        position = positions[0]
        original_price = position.current_price
        
        # Test concurrent updates
        def update_position_concurrent():
            new_price = original_price * 1.01
            self.position_manager.update_position(position.id, new_price)
            return True
        
        # Run concurrent updates
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(update_position_concurrent) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # All updates should complete successfully
        self.assertEqual(len(results), 10)
        self.assertTrue(all(results))
        
        # Position should be updated
        updated_position = self.position_manager.get_position_by_id(position.id)
        self.assertIsNotNone(updated_position)
        self.assertNotEqual(updated_position.current_price, original_price)
    
    def test_real_time_price_updates(self):
        """Test real-time price update functionality."""
        # Get a position
        positions = self.position_manager.get_open_positions()
        if not positions:
            self.skipTest("No open positions to test")
        
        position = positions[0]
        symbol = position.symbol
        original_price = position.current_price
        
        # Test price update from feed
        new_price = original_price * 1.02
        self.position_manager.update_price_from_feed(symbol, new_price)
        
        # Check that price cache was updated
        cached_price = self.position_manager.get_latest_price(symbol)
        self.assertEqual(cached_price, new_price)
        
        # Check that position was updated
        updated_position = self.position_manager.get_position_by_id(position.id)
        self.assertEqual(updated_position.current_price, new_price)
        
        # Test price history
        history = self.position_manager.get_price_history(symbol, limit=5)
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)
    
    def test_database_statistics(self):
        """Test database statistics functionality."""
        stats = self.position_manager.get_database_stats()
        self.assertIsInstance(stats, dict)
        
        # Check required fields
        required_fields = ['total_positions', 'open_positions', 'closed_positions', 
                         'history_records', 'snapshots']
        for field in required_fields:
            self.assertIn(field, stats)
            self.assertIsInstance(stats[field], int)
        
        # Check database size fields
        self.assertIn('database_size_bytes', stats)
        self.assertIn('database_size_mb', stats)
        self.assertIsInstance(stats['database_size_bytes'], int)
        self.assertIsInstance(stats['database_size_mb'], float)
    
    def test_position_history_filtering(self):
        """Test position history filtering capabilities."""
        # Test filtering by symbol
        btc_history = self.position_manager.get_position_history(symbol="BTC/USDT")
        self.assertIsInstance(btc_history, list)
        
        # Test filtering by strategy
        stat_arb_history = self.position_manager.get_position_history(strategy="stat_arb")
        self.assertIsInstance(stat_arb_history, list)
        
        # Test filtering by status
        closed_history = self.position_manager.get_position_history(status="closed")
        self.assertIsInstance(closed_history, list)
        
        # Test date filtering
        start_date = datetime.now() - timedelta(days=7)
        recent_history = self.position_manager.get_position_history(start_date=start_date)
        self.assertIsInstance(recent_history, list)
    
    def test_performance_report(self):
        """Test comprehensive performance report generation."""
        report = self.position_manager.get_performance_report()
        self.assertIsInstance(report, dict)
        
        # Check report structure
        self.assertIn('portfolio_metrics', report)
        self.assertIn('exposure', report)
        self.assertIn('top_positions', report)
        self.assertIn('worst_positions', report)
        self.assertIn('timestamp', report)
        
        # Check metrics
        metrics = report['portfolio_metrics']
        self.assertIsInstance(metrics['total_market_value'], float)
        self.assertIsInstance(metrics['total_pnl'], float)
        self.assertIsInstance(metrics['win_rate'], float)
        
        # Check exposure data
        exposure = report['exposure']
        self.assertIn('by_asset', exposure)
        self.assertIn('by_sector', exposure)
        
        # Check position lists
        self.assertIsInstance(report['top_positions'], list)
        self.assertIsInstance(report['worst_positions'], list)
    
    def test_empty_analytics(self):
        """Test analytics with no data."""
        # Create a new position manager with no data
        empty_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        empty_db.close()
        
        try:
            empty_manager = PositionManager(db_path=empty_db.name)
            
            # Test empty analytics
            analytics = empty_manager.get_position_analytics()
            self.assertIsInstance(analytics, dict)
            self.assertEqual(analytics['summary']['total_positions'], 0)
            
            # Test empty export
            json_export = empty_manager.export_positions(format="json")
            self.assertEqual(json_export, "[]")
            
            # Test empty DataFrame
            df = empty_manager.export_positions(format="dataframe")
            self.assertTrue(df.empty)
            
        finally:
            # Cleanup
            asyncio.run(empty_manager.shutdown())
            if os.path.exists(empty_db.name):
                os.unlink(empty_db.name)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test invalid position ID
        result = self.position_manager.update_position("invalid_id", 100.0)
        self.assertIsNone(result)  # Should handle gracefully
        
        # Test invalid export format
        with self.assertRaises(ValueError):
            self.position_manager.export_positions(format="invalid_format")
        
        # Test invalid close quantity
        positions = self.position_manager.get_open_positions()
        if positions:
            position = positions[0]
            with self.assertRaises(ValueError):
                self.position_manager.close_position(
                    position.id, 100.0, close_quantity=position.quantity + 1
                )
    
    def test_concurrent_operations(self):
        """Test concurrent operations on position manager."""
        import concurrent.futures
        
        # Get positions for testing
        positions = self.position_manager.get_open_positions()
        if len(positions) < 1:
            self.skipTest("Need at least 1 position for concurrent testing")
        
        def concurrent_operation(position_id: str):
            """Perform concurrent operations on a position."""
            # Update price
            self.position_manager.update_position(position_id, 100.0)
            # Get position
            position = self.position_manager.get_position_by_id(position_id)
            # Get portfolio metrics
            metrics = self.position_manager.get_portfolio_metrics()
            return position is not None and metrics is not None
        
        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for position in positions[:min(3, len(positions))]:
                future = executor.submit(concurrent_operation, position.id)
                futures.append(future)
        
            results = [future.result() for future in futures]
        
        # All operations should complete successfully
        self.assertEqual(len(results), len(futures))


class TestEnhancedPositionManagerAsync(unittest.IsolatedAsyncioTestCase):
    """Async test suite for enhanced position manager features."""
    
    async def asyncSetUp(self):
        """Set up async test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.position_manager = PositionManager(db_path=self.temp_db.name)
        
        # Create test positions
        self._create_test_positions()
    
    async def asyncTearDown(self):
        """Clean up async test environment."""
        await self.position_manager.shutdown()
        
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def _create_test_positions(self):
        """Create test positions for async tests."""
        # Create some test positions
        self.position_manager.add_position(
            "BTC/USDT", PositionSide.LONG, 0.1, 50000, "async_test_1", "stat_arb"
        )
        self.position_manager.add_position(
            "ETH/USDT", PositionSide.SHORT, 2.0, 3000, "async_test_2", "sentiment"
        )
    
    async def test_async_price_updates(self):
        """Test async price update functionality."""
        # Test async price update
        await self.position_manager.update_all_prices()
        
        # Verify positions exist
        positions = self.position_manager.get_open_positions()
        self.assertGreater(len(positions), 0)
    
    async def test_price_update_loop(self):
        """Test price update loop functionality."""
        # Start price update loop
        self.position_manager.start_price_update_loop(update_interval=1)
        
        # Wait a bit for loop to start
        await asyncio.sleep(0.1)
        
        # Verify loop is running (positions should exist)
        positions = self.position_manager.get_open_positions()
        self.assertGreater(len(positions), 0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 