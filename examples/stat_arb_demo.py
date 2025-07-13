"""
Statistical Arbitrage Strategy Demo

This script demonstrates how to use the statistical arbitrage strategy
with realistic market data and trading scenarios.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from strategy.stat_arb import StatisticalArbitrage, create_stat_arb_strategy
from utils.logger import logger


def generate_synthetic_data(days: int = 30, assets: list = None) -> dict:
    """
    Generate synthetic price data for demonstration.
    
    Args:
        days: Number of days of data to generate
        assets: List of asset symbols
        
    Returns:
        Dictionary of price data for each asset
    """
    if assets is None:
        assets = ['ETH', 'SOL', 'BTC', 'ADA']
    
    np.random.seed(42)  # For reproducible results
    
    # Generate timestamps
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    price_data = {}
    
    # Base prices
    base_prices = {
        'ETH': 2000,
        'SOL': 100,
        'BTC': 40000,
        'ADA': 0.5
    }
    
    for asset in assets:
        base_price = base_prices.get(asset, 100)
        
        # Generate price series with trend and noise
        trend = np.linspace(0, 0.1, len(timestamps))  # 10% upward trend
        noise = np.random.normal(0, 0.02, len(timestamps))  # 2% volatility
        returns = trend + noise
        
        # Convert to prices
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        price_data[asset] = pd.Series(prices, index=timestamps)
    
    return price_data


def demonstrate_pair_selection():
    """Demonstrate pair selection and cointegration testing."""
    logger.info("=== Pair Selection Demo ===")
    
    # Create strategy
    config = {
        'z_score_threshold': 2.0,
        'lookback_period': 100,
        'correlation_threshold': 0.6,
        'cointegration_pvalue_threshold': 0.05
    }
    strategy = StatisticalArbitrage(config)
    
    # Generate synthetic data
    price_data = generate_synthetic_data(days=30)
    
    # Update strategy with price data
    for asset, prices in price_data.items():
        for timestamp, price in prices.items():
            strategy.update_price_data(asset, price, timestamp)
    
    # Find cointegrated pairs
    assets = list(price_data.keys())
    cointegrated_pairs = strategy.find_cointegrated_pairs(assets)
    
    logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs:")
    for asset1, asset2 in cointegrated_pairs:
        pair_key = f"{asset1}_{asset2}"
        pair_data = strategy.pairs_data[pair_key]
        logger.info(f"  {asset1}-{asset2}: correlation={pair_data.correlation:.3f}, "
                   f"p-value={pair_data.cointegration_pvalue:.3f}")
    
    return strategy, price_data


def demonstrate_signal_generation(strategy: StatisticalArbitrage, price_data: dict):
    """Demonstrate signal generation."""
    logger.info("\n=== Signal Generation Demo ===")
    
    # Update with latest prices to trigger signals
    current_time = datetime.now()
    for asset, prices in price_data.items():
        latest_price = prices.iloc[-1]
        strategy.update_price_data(asset, latest_price, current_time)
    
    # Generate signals
    signals = strategy.generate_signals()
    
    logger.info(f"Generated {len(signals)} signals:")
    for signal in signals:
        logger.info(f"  {signal.signal_type}: {signal.asset1} {signal.action1} "
                   f"{signal.size1:.4f}, {signal.asset2} {signal.action2} "
                   f"{signal.size2:.4f} (z-score: {signal.z_score:.3f})")
    
    return signals


def demonstrate_position_management(strategy: StatisticalArbitrage, signals: list):
    """Demonstrate position management."""
    logger.info("\n=== Position Management Demo ===")
    
    # Update positions with signals
    strategy.update_positions(signals)
    
    # Show open positions
    open_positions = strategy.get_open_positions()
    logger.info(f"Open positions: {len(open_positions)}")
    
    for pair, position in open_positions.items():
        logger.info(f"  {pair}: {position.asset1} {position.size1:.4f} @ {position.entry_price1:.2f}, "
                   f"{position.asset2} {position.size2:.4f} @ {position.entry_price2:.2f}")
    
    return open_positions


def demonstrate_performance_tracking(strategy: StatisticalArbitrage):
    """Demonstrate performance tracking."""
    logger.info("\n=== Performance Tracking Demo ===")
    
    # Simulate some trades for performance metrics
    performance = strategy.get_performance_summary()
    
    logger.info("Performance Summary:")
    for metric, value in performance.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")


def demonstrate_risk_management(strategy: StatisticalArbitrage, price_data: dict):
    """Demonstrate risk management features."""
    logger.info("\n=== Risk Management Demo ===")
    
    # Show position sizing for different assets
    for asset in ['ETH', 'SOL', 'BTC']:
        if asset in price_data:
            position_size = strategy._calculate_position_size(asset)
            logger.info(f"  {asset} position size: {position_size:.4f}")
    
    # Show risk parameters
    logger.info(f"  Stop loss: {strategy.stop_loss_pct:.1%}")
    logger.info(f"  Take profit: {strategy.take_profit_pct:.1%}")
    logger.info(f"  Z-score threshold: {strategy.z_score_threshold}")


def plot_spread_analysis(strategy: StatisticalArbitrage, price_data: dict):
    """Create plots for spread analysis."""
    logger.info("\n=== Creating Spread Analysis Plots ===")
    
    # Create plots directory
    os.makedirs('data/plots', exist_ok=True)
    
    # Plot price data
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (asset, prices) in enumerate(price_data.items()):
        if i < 4:
            axes[i].plot(prices.index, prices.values)
            axes[i].set_title(f'{asset} Price')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Price')
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('data/plots/price_data.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot spreads for cointegrated pairs
    if strategy.pairs_data:
        fig, axes = plt.subplots(len(strategy.pairs_data), 1, figsize=(15, 5*len(strategy.pairs_data)))
        if len(strategy.pairs_data) == 1:
            axes = [axes]
        
        for i, (pair_key, pair_data) in enumerate(strategy.pairs_data.items()):
            asset1, asset2 = pair_data.asset1, pair_data.asset2
            
            if asset1 in price_data and asset2 in price_data:
                # Align price data
                common_times = price_data[asset1].index.intersection(price_data[asset2].index)
                price1 = price_data[asset1][common_times]
                price2 = price_data[asset2][common_times]
                
                # Calculate spread
                spread = price1 - price2
                z_scores = (spread - pair_data.spread_mean) / pair_data.spread_std
                
                axes[i].plot(spread.index, spread.values, label='Spread', alpha=0.7)
                axes[i].axhline(y=pair_data.spread_mean, color='r', linestyle='--', label='Mean')
                axes[i].axhline(y=pair_data.spread_mean + 2*pair_data.spread_std, color='g', linestyle=':', label='+2σ')
                axes[i].axhline(y=pair_data.spread_mean - 2*pair_data.spread_std, color='g', linestyle=':', label='-2σ')
                axes[i].set_title(f'{asset1}-{asset2} Spread')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel('Spread')
                axes[i].legend()
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('data/plots/spread_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("Plots saved to data/plots/")


def run_complete_demo():
    """Run the complete statistical arbitrage demo."""
    logger.info("Starting Statistical Arbitrage Strategy Demo")
    logger.info("=" * 50)
    
    try:
        # Step 1: Pair Selection
        strategy, price_data = demonstrate_pair_selection()
        
        # Step 2: Signal Generation
        signals = demonstrate_signal_generation(strategy, price_data)
        
        # Step 3: Position Management
        open_positions = demonstrate_position_management(strategy, signals)
        
        # Step 4: Performance Tracking
        demonstrate_performance_tracking(strategy)
        
        # Step 5: Risk Management
        demonstrate_risk_management(strategy, price_data)
        
        # Step 6: Visualization
        plot_spread_analysis(strategy, price_data)
        
        logger.info("\n" + "=" * 50)
        logger.info("Demo completed successfully!")
        
        return strategy, price_data
        
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        raise


if __name__ == "__main__":
    # Run the complete demo
    strategy, price_data = run_complete_demo()
    
    # Additional interactive analysis
    print("\n" + "="*50)
    print("Interactive Analysis Options:")
    print("1. View strategy configuration")
    print("2. View recent signals")
    print("3. View open positions")
    print("4. View performance summary")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                print(f"Z-score threshold: {strategy.z_score_threshold}")
                print(f"Lookback period: {strategy.lookback_period}")
                print(f"Correlation threshold: {strategy.correlation_threshold}")
                print(f"Position size limit: {strategy.position_size_limit}")
                
            elif choice == '2':
                recent_signals = strategy.get_recent_signals(hours=24)
                print(f"Recent signals ({len(recent_signals)}):")
                for signal in recent_signals[-5:]:  # Show last 5
                    print(f"  {signal.timestamp}: {signal.signal_type} for {signal.pair}")
                    
            elif choice == '3':
                positions = strategy.get_open_positions()
                print(f"Open positions ({len(positions)}):")
                for pair, pos in positions.items():
                    print(f"  {pair}: {pos.asset1} {pos.size1:.4f}, {pos.asset2} {pos.size2:.4f}")
                    
            elif choice == '4':
                performance = strategy.get_performance_summary()
                print("Performance Summary:")
                for metric, value in performance.items():
                    print(f"  {metric}: {value}")
                    
            elif choice == '5':
                print("Exiting demo...")
                break
                
            else:
                print("Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\nExiting demo...")
            break
        except Exception as e:
            print(f"Error: {e}") 