#!/usr/bin/env python3
"""
Simple startup script for the crypto trading bot
"""

import os
import sys
import asyncio
import argparse

def main():
    """Main entry point for the trading bot."""
    parser = argparse.ArgumentParser(description='Crypto Trading Bot')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--simulation', action='store_true', help='Run in simulation mode (default)')
    parser.add_argument('--live', action='store_true', help='Run in live trading mode')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    if args.test:
        print("🧪 Running in test mode...")
        # Run the test script
        from tests.test_main import test_main
        success = asyncio.run(test_main())
        sys.exit(0 if success else 1)
    
    elif args.live:
        print("🚀 Starting live trading bot...")
        # Set environment variables for live trading
        os.environ['TRADING_ENABLED'] = 'true'
        os.environ['SIMULATION_MODE'] = 'false'
        
        # Import and run main
        from src.main import main
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    
    else:
        print("🎮 Starting simulation trading bot...")
        # Default to simulation mode
        os.environ['TRADING_ENABLED'] = 'false'
        os.environ['SIMULATION_MODE'] = 'true'
        
        # Import and run main
        from src.main import main
        exit_code = asyncio.run(main())
        sys.exit(exit_code)

if __name__ == "__main__":
    main() 