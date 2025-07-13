"""
Data Integration Demo

Demonstrates how to connect the statistical arbitrage strategy
to real-time and historical market data.
"""

import sys
import os
import time
from datetime import datetime, timedelta
import signal
import threading

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from strategy.data_integration import create_strategy_integration
from utils.logger import logger


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    print("\nReceived interrupt signal. Shutting down...")
    sys.exit(0)


def print_status(integration, interval=30):
    """Print status updates periodically."""
    while True:
        try:
            status = integration.get_strategy_status()
            print("\n" + "="*60)
            print("STRATEGY STATUS UPDATE")
            print("="*60)
            print(f"Running: {status['running']}")
            print(f"Cointegrated Pairs: {status['pairs_count']}")
            print(f"Open Positions: {status['positions_count']}")
            print(f"Total Signals: {status['signals_count']}")
            print(f"Last Update: {status['last_update']}")
            
            # Performance metrics
            perf = status['performance']
            print(f"\nPERFORMANCE METRICS:")
            print(f"  Total Trades: {perf['total_trades']}")
            print(f"  Win Rate: {perf['win_rate']:.2%}")
            print(f"  Total PnL: {perf['total_pnl']:.2f}")
            print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {perf['max_drawdown']:.2%}")
            
            # Open positions
            positions = integration.get_open_positions()
            if positions:
                print(f"\nOPEN POSITIONS:")
                for pair, pos in positions.items():
                    print(f"  {pair}: {pos.asset1} {pos.size1:.4f} @ {pos.entry_price1:.2f}, "
                          f"{pos.asset2} {pos.size2:.4f} @ {pos.entry_price2:.2f}, "
                          f"PnL: {pos.current_pnl:.2f}")
            
            # Recent signals
            recent_signals = integration.get_recent_signals(hours=1)
            if recent_signals:
                print(f"\nRECENT SIGNALS (last hour):")
                for signal in recent_signals[-5:]:  # Show last 5
                    print(f"  {signal.timestamp}: {signal.signal_type} for {signal.pair} "
                          f"(z-score: {signal.z_score:.3f})")
            
            print("="*60)
            time.sleep(interval)
            
        except Exception as e:
            logger.error(f"Error in status update: {e}")
            time.sleep(5)


def run_demo():
    """Run the data integration demo."""
    print("Starting Data Integration Demo")
    print("="*60)
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Strategy configuration
    strategy_config = {
        'z_score_threshold': 2.0,
        'lookback_period': 100,
        'correlation_threshold': 0.7,
        'cointegration_pvalue_threshold': 0.05,
        'min_spread_std': 0.001,
        'position_size_limit': 0.1,
        'stop_loss_pct': 0.05,
        'take_profit_pct': 0.1,
        'slippage': 0.001  # 0.1% slippage
    }
    
    try:
        # Create integration
        print("Creating strategy data integration...")
        integration = create_strategy_integration(strategy_config)
        
        # Start status update thread
        status_thread = threading.Thread(target=print_status, args=(integration, 30), daemon=True)
        status_thread.start()
        
        # Start integration
        print("Starting data integration...")
        integration.start()
        
        print("\nIntegration started successfully!")
        print("The strategy will now:")
        print("1. Load historical data for initialization")
        print("2. Find cointegrated pairs")
        print("3. Monitor real-time price data")
        print("4. Generate trading signals")
        print("5. Update position PnL")
        print("\nPress Ctrl+C to stop the demo")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Shutting down...")
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        raise
    finally:
        try:
            integration.stop()
            print("Integration stopped successfully.")
        except Exception as e:
            logger.error(f"Error stopping integration: {e}")


def run_quick_test():
    """Run a quick test of the integration without starting the full system."""
    print("Running Quick Integration Test")
    print("="*40)
    
    try:
        # Create integration
        integration = create_strategy_integration()
        
        # Test basic functionality
        print("Testing basic functionality...")
        
        # Test status
        status = integration.get_strategy_status()
        print(f"✓ Status retrieval: {status['running']}")
        
        # Test positions
        positions = integration.get_open_positions()
        print(f"✓ Open positions: {len(positions)}")
        
        # Test signals
        signals = integration.get_recent_signals(hours=24)
        print(f"✓ Recent signals: {len(signals)}")
        
        # Test force signal generation
        signals = integration.force_signal_generation()
        print(f"✓ Signal generation: {len(signals)} signals")
        
        print("\n✓ All basic tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Integration Demo')
    parser.add_argument('--test', action='store_true', 
                       help='Run quick test instead of full demo')
    
    args = parser.parse_args()
    
    if args.test:
        run_quick_test()
    else:
        run_demo() 