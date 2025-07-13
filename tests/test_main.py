#!/usr/bin/env python3
"""
Test script for main.py trading bot
"""

import asyncio
import sys
import os
import pytest

# Add src to path
sys.path.append('src')

@pytest.mark.asyncio
async def test_main():
    """Test the main trading bot functionality."""
    try:
        from src.main import TradingBot
        
        print("🧪 Testing TradingBot initialization...")
        
        # Create trading bot instance
        bot = TradingBot()
        
        # Test initialization
        print("🔄 Initializing trading bot...")
        success = await bot.initialize()
        
        if success:
            print("✅ Trading bot initialized successfully")
            
            # Test data collection
            print("📡 Testing data collection...")
            await bot._start_data_collection()
            
            # Test signal generation
            print("🎯 Testing signal generation...")
            signals = await bot._generate_signals()
            print(f"✅ Generated {len(signals)} signals")
            
            # Test portfolio metrics
            print("📊 Testing portfolio metrics...")
            await bot._update_portfolio_metrics()
            
            # Test trading conditions
            print("🔍 Testing trading conditions...")
            conditions_met = bot._check_trading_conditions()
            print(f"✅ Trading conditions: {conditions_met}")
            
            # Shutdown
            print("🛑 Shutting down...")
            await bot.shutdown()
            
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Trading bot initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run test
    success = asyncio.run(test_main())
    
    if success:
        print("\n🎉 Trading bot test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Trading bot test failed!")
        sys.exit(1) 