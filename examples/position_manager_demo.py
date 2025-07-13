#!/usr/bin/env python3
"""
Enhanced Position Manager Demo

This demo showcases all the new features implemented in the position manager:
- Closed position analytics
- Enhanced sector mapping
- Advanced export capabilities (CSV, DataFrame)
- Historical performance analysis
- Thread safety and concurrency
- Real-time price updating
- Database statistics and cleanup
"""

import asyncio
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

from src.execution.position_manager import (
    PositionManager, PositionSide, PositionType, PositionStatus
)


class EnhancedPositionManagerDemo:
    """Demo class showcasing enhanced position manager features."""
    
    def __init__(self):
        self.position_manager = PositionManager(db_path="data/positions_enhanced.db")
        self.demo_positions = []
    
    async def run_comprehensive_demo(self):
        """Run the complete enhanced position manager demo."""
        print("🚀 Enhanced Position Manager Demo")
        print("=" * 50)
        
        # 1. Create demo positions with various strategies and assets
        await self._create_demo_positions()
        
        # 2. Demonstrate enhanced sector mapping
        self._demo_sector_mapping()
        
        # 3. Show advanced export capabilities
        self._demo_export_capabilities()
        
        # 4. Demonstrate historical analytics
        self._demo_historical_analytics()
        
        # 5. Show thread safety features
        await self._demo_thread_safety()
        
        # 6. Demonstrate real-time price updates
        await self._demo_real_time_updates()
        
        # 7. Show database statistics and cleanup
        self._demo_database_operations()
        
        # 8. Generate comprehensive performance report
        self._demo_performance_report()
        
        print("\n✅ Enhanced Position Manager Demo Complete!")
    
    async def _create_demo_positions(self):
        """Create a variety of demo positions."""
        print("\n📊 Creating Demo Positions...")
        
        # Layer 1 positions
        self._add_demo_position("BTC/USDT", PositionSide.LONG, 0.1, 50000, "stat_arb")
        self._add_demo_position("ETH/USDT", PositionSide.SHORT, 2.0, 3000, "sentiment")
        self._add_demo_position("SOL/USDT", PositionSide.LONG, 10.0, 100, "momentum")
        
        # DeFi positions
        self._add_demo_position("UNI/USDT", PositionSide.LONG, 50.0, 8.0, "defi_arb")
        self._add_demo_position("AAVE/USDT", PositionSide.SHORT, 5.0, 200, "defi_arb")
        
        # Gaming positions
        self._add_demo_position("AXS/USDT", PositionSide.LONG, 20.0, 15.0, "gaming")
        self._add_demo_position("SAND/USDT", PositionSide.LONG, 100.0, 0.8, "gaming")
        
        # Stablecoin positions
        self._add_demo_position("USDC/USDT", PositionSide.LONG, 1000.0, 1.0, "arbitrage")
        
        # Close some positions to demonstrate analytics
        await self._close_some_positions()
        
        print(f"✅ Created {len(self.demo_positions)} demo positions")
    
    def _add_demo_position(self, symbol: str, side: PositionSide, quantity: float, 
                          price: float, strategy: str):
        """Add a demo position."""
        position = self.position_manager.add_position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            order_id=f"demo_order_{len(self.demo_positions)}",
            strategy_name=strategy,
            stop_loss=price * 0.95 if side == PositionSide.LONG else price * 1.05,
            take_profit=price * 1.10 if side == PositionSide.LONG else price * 0.90
        )
        self.demo_positions.append(position.id)
    
    async def _close_some_positions(self):
        """Close some positions to demonstrate closed position analytics."""
        # Close first 3 positions with different outcomes
        positions = self.position_manager.get_open_positions()[:3]
        
        for i, position in enumerate(positions):
            # Simulate different close prices
            if i == 0:  # Winning trade
                close_price = position.entry_price * 1.05
            elif i == 1:  # Losing trade
                close_price = position.entry_price * 0.95
            else:  # Break-even
                close_price = position.entry_price
            
            self.position_manager.close_position(
                position.id, close_price, order_id=f"close_order_{i}"
            )
    
    def _demo_sector_mapping(self):
        """Demonstrate enhanced sector mapping."""
        print("\n🏢 Enhanced Sector Mapping Demo...")
        
        metrics = self.position_manager.get_portfolio_metrics()
        
        print("Portfolio Exposure by Sector:")
        for sector, exposure in metrics.exposure_by_sector.items():
            print(f"  {sector}: ${exposure:,.2f}")
        
        print("\nPortfolio Exposure by Asset:")
        for asset, exposure in metrics.exposure_by_asset.items():
            print(f"  {asset}: ${exposure:,.2f}")
    
    def _demo_export_capabilities(self):
        """Demonstrate advanced export capabilities."""
        print("\n📤 Advanced Export Capabilities Demo...")
        
        # JSON export
        json_export = self.position_manager.export_positions(format="json", include_closed=True)
        print(f"JSON Export: {len(json_export)} characters")
        
        # CSV export
        csv_export = self.position_manager.export_positions(format="csv", include_closed=True)
        print(f"CSV Export: {len(csv_export)} characters")
        
        # DataFrame export
        df = self.position_manager.export_positions(format="dataframe", include_closed=True)
        print(f"DataFrame Export: {len(df)} rows, {len(df.columns)} columns")
        
        # Show DataFrame info
        if not df.empty:
            print("\nDataFrame Info:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Data Types: {df.dtypes.to_dict()}")
    
    def _demo_historical_analytics(self):
        """Demonstrate historical position analytics."""
        print("\n📈 Historical Analytics Demo...")
        
        # Get position history
        history = self.position_manager.get_position_history(days=30)
        print(f"Position History: {len(history)} records")
        
        # Get comprehensive analytics
        analytics = self.position_manager.get_position_analytics(days=30)
        
        print("\nAnalytics Summary:")
        summary = analytics['summary']
        print(f"  Total Positions: {summary['total_positions']}")
        print(f"  Win Rate: {summary['win_rate']:.2%}")
        print(f"  Total PnL: ${summary['total_pnl']:,.2f}")
        print(f"  Sharpe Ratio: {summary['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: ${summary['max_drawdown']:,.2f}")
        
        print("\nStrategy Performance:")
        for strategy, perf in analytics['strategy_performance'].items():
            print(f"  {strategy}: {perf['count']} trades, "
                  f"${perf['total_pnl']:,.2f} PnL, {perf['win_rate']:.1%} win rate")
    
    async def _demo_thread_safety(self):
        """Demonstrate thread safety features."""
        print("\n🔒 Thread Safety Demo...")
        
        import concurrent.futures
        
        # Simulate concurrent position updates
        def update_position_concurrent(position_id: str, price: float):
            self.position_manager.update_position(position_id, price)
            return f"Updated {position_id} to {price}"
        
        # Get some open positions
        open_positions = self.position_manager.get_open_positions()
        if open_positions:
            position = open_positions[0]
            
            # Simulate concurrent updates
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for i in range(5):
                    new_price = position.entry_price * (1 + (i * 0.01))
                    future = executor.submit(update_position_concurrent, position.id, new_price)
                    futures.append(future)
                
                # Wait for all updates
                results = [future.result() for future in futures]
                print(f"Concurrent updates completed: {len(results)} operations")
    
    async def _demo_real_time_updates(self):
        """Demonstrate real-time price updates."""
        print("\n⚡ Real-Time Price Updates Demo...")
        
        # Simulate price updates from external feed
        open_positions = self.position_manager.get_open_positions()
        
        for position in open_positions[:3]:  # Update first 3 positions
            # Simulate price movement
            new_price = position.entry_price * 1.02
            self.position_manager.update_price_from_feed(position.symbol, new_price)
            print(f"Updated {position.symbol} price to ${new_price:,.2f}")
        
        # Show price history
        if open_positions:
            symbol = open_positions[0].symbol
            history = self.position_manager.get_price_history(symbol, limit=5)
            print(f"Price history for {symbol}: {history}")
    
    def _demo_database_operations(self):
        """Demonstrate database statistics and cleanup."""
        print("\n🗄️ Database Operations Demo...")
        
        # Get database statistics
        stats = self.position_manager.get_database_stats()
        print("Database Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # Show cleanup capabilities
        print("\nCleanup Operations:")
        print("  (Demo mode - no actual cleanup performed)")
        # self.position_manager.cleanup_old_data(days=30)  # Uncomment to actually cleanup
    
    def _demo_performance_report(self):
        """Generate and display comprehensive performance report."""
        print("\n📊 Comprehensive Performance Report...")
        
        report = self.position_manager.get_performance_report()
        
        print("Portfolio Metrics:")
        metrics = report['portfolio_metrics']
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        print("\nTop Performing Positions:")
        for i, pos in enumerate(report['top_positions'][:3], 1):
            print(f"  {i}. {pos['symbol']} {pos['side']}: ${pos['total_pnl']:,.2f}")
        
        print("\nWorst Performing Positions:")
        for i, pos in enumerate(report['worst_positions'][:3], 1):
            print(f"  {i}. {pos['symbol']} {pos['side']}: ${pos['total_pnl']:,.2f}")
    
    def export_demo_data(self, filename: str = "demo_export"):
        """Export demo data to various formats."""
        print(f"\n💾 Exporting Demo Data to {filename}...")
        
        # Export to different formats
        formats = ["json", "csv", "dataframe"]
        
        for fmt in formats:
            try:
                if fmt == "dataframe":
                    df = self.position_manager.export_positions(format=fmt, include_closed=True)
                    df.to_csv(f"{filename}.csv", index=False)
                    print(f"  DataFrame exported to {filename}.csv")
                else:
                    data = self.position_manager.export_positions(format=fmt, include_closed=True)
                    with open(f"{filename}.{fmt}", 'w') as f:
                        f.write(data)
                    print(f"  {fmt.upper()} exported to {filename}.{fmt}")
            except Exception as e:
                print(f"  Failed to export {fmt}: {e}")


async def main():
    """Run the enhanced position manager demo."""
    demo = EnhancedPositionManagerDemo()
    await demo.run_comprehensive_demo()
    
    # Export demo data
    demo.export_demo_data("enhanced_demo_export")
    
    # Clean shutdown
    await demo.position_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main()) 