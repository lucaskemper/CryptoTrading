#!/usr/bin/env python3
"""
Simple Backtesting Runner

Quick script to run backtests from the command line.
Usage: python run_backtest.py [options]
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backtesting.backtest_engine import BacktestEngine, BacktestConfig
from utils.logger import logger


def create_config_from_args(args):
    """Create backtest configuration from command line arguments."""
    
    # Calculate date range
    if args.days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
    else:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else datetime.now() - timedelta(days=180)
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else datetime.now()
    
    # Strategy configuration
    strategy_config = {
        'statistical_arbitrage': {
            'z_score_threshold': args.z_threshold,
            'lookback_period': args.lookback,
            'correlation_threshold': 0.7,
            'cointegration_pvalue_threshold': 0.05,
            'min_spread_std': 0.001,
            'position_size_limit': args.position_size,
            'stop_loss_pct': args.stop_loss,
            'take_profit_pct': args.take_profit,
            'max_positions': 5,
            'rebalance_frequency': 300,
            'dynamic_threshold_window': 100,
            'spread_model': 'ols',
            'slippage': args.slippage
        },
        'sentiment_analysis': {
            'model': 'gpt-3.5-turbo',
            'confidence_threshold': 0.7,
            'sentiment_weight': 0.3
        },
        'signal_generator': {
            'combination_method': 'weighted',
            'stat_weight': 0.7,
            'sentiment_weight': 0.3,
            'min_confidence': 0.3,
            'enable_risk_checks': True,
            'require_risk_approval': False,
            'use_enhanced_ml': True  # Enable ML-enhanced signal generator by default
        },
        'risk_management': {
            'max_position_size': args.position_size,
            'max_total_exposure': 0.8,
            'max_single_asset_exposure': 0.3,
            'max_open_positions': 10,
            'max_daily_drawdown': 0.05,
            'max_total_drawdown': 0.15,
            'volatility_threshold': 0.1
        },
        'position_management': {
            'enable_position_tracking': True,
            'enable_pnl_calculation': True,
            'enable_risk_checks': True
        }
    }
    
    return BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        symbols=args.symbols.split(',') if args.symbols else ['ETH/USDT', 'SOL/USDT', 'BTC/USDT'],
        exchanges=['binance', 'kraken'],
        strategy_config=strategy_config,
        initial_capital=args.capital,
        max_position_size=args.position_size,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        slippage=args.slippage,
        commission=args.commission,
        data_frequency=args.frequency,
        sentiment_enabled=args.sentiment,
        sentiment_frequency='4h',
        track_metrics=True,
        generate_plots=args.plots,
        save_results=args.save,
        results_dir=args.output_dir
    )


def print_results(results):
    """Print backtest results in a formatted way."""
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Period: {results.start_time.strftime('%Y-%m-%d')} to {results.end_time.strftime('%Y-%m-%d')}")
    print(f"Duration: {results.duration.days} days")
    print()
    print("PERFORMANCE METRICS:")
    print(f"  Total Return: {results.total_return:.2%}")
    print(f"  Annualized Return: {results.annualized_return:.2%}")
    print(f"  Sharpe Ratio: {results.sharpe_ratio:.3f}")
    print(f"  Maximum Drawdown: {results.max_drawdown:.2%}")
    print(f"  Volatility: {results.volatility:.2%}")
    print()
    print("TRADE STATISTICS:")
    print(f"  Total Trades: {results.total_trades}")
    print(f"  Winning Trades: {results.winning_trades}")
    print(f"  Losing Trades: {results.losing_trades}")
    print(f"  Win Rate: {results.win_rate:.2%}")
    print(f"  Profit Factor: {results.profit_factor:.2f}")
    print(f"  Average Win: ${results.avg_win:.2f}")
    print(f"  Average Loss: ${results.avg_loss:.2f}")
    print()
    print("PORTFOLIO METRICS:")
    print(f"  Initial Capital: ${results.config.initial_capital:,.2f}")
    print(f"  Final Portfolio Value: ${results.final_portfolio_value:,.2f}")
    print(f"  Peak Portfolio Value: ${results.peak_portfolio_value:,.2f}")
    print(f"  Strategy Signals: {results.strategy_signals}")
    print(f"  Executed Trades: {results.executed_trades}")
    print(f"  Rejected Trades: {results.rejected_trades}")
    print("="*60)


def main():
    """Main function to run backtest from command line."""
    parser = argparse.ArgumentParser(description='Run crypto trading backtest')
    
    # Date range options
    parser.add_argument('--days', type=int, help='Number of days to backtest (default: 180)')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    
    # Trading parameters
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital (default: 100000)')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols (default: ETH/USDT,SOL/USDT,BTC/USDT)')
    parser.add_argument('--frequency', type=str, default='1h', choices=['1m', '5m', '15m', '1h', '4h', '1d'], help='Data frequency (default: 1h)')
    
    # Strategy parameters
    parser.add_argument('--z-threshold', type=float, default=2.0, help='Z-score threshold (default: 2.0)')
    parser.add_argument('--lookback', type=int, default=100, help='Lookback period (default: 100)')
    parser.add_argument('--position-size', type=float, default=0.1, help='Max position size (default: 0.1)')
    parser.add_argument('--stop-loss', type=float, default=0.05, help='Stop loss percentage (default: 0.05)')
    parser.add_argument('--take-profit', type=float, default=0.1, help='Take profit percentage (default: 0.1)')
    
    # Execution parameters
    parser.add_argument('--slippage', type=float, default=0.001, help='Slippage (default: 0.001)')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission (default: 0.001)')
    
    # Features
    parser.add_argument('--sentiment', action='store_true', help='Enable sentiment analysis')
    parser.add_argument('--plots', action='store_true', help='Generate plots')
    parser.add_argument('--save', action='store_true', help='Save results to files')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='data/backtest_results', help='Output directory (default: data/backtest_results)')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    try:
        # Create configuration
        config = create_config_from_args(args)
        
        # Create and run backtest
        logger.info("Starting backtest...")
        engine = BacktestEngine(config)
        results = engine.run_backtest()
        
        # Print results
        print_results(results)
        
        # Save results if requested
        if args.save:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save summary to file
            summary_file = output_path / f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Backtest Results Summary\n")
                f.write(f"======================\n\n")
                f.write(f"Period: {results.start_time} to {results.end_time}\n")
                f.write(f"Total Return: {results.total_return:.2%}\n")
                f.write(f"Annualized Return: {results.annualized_return:.2%}\n")
                f.write(f"Sharpe Ratio: {results.sharpe_ratio:.3f}\n")
                f.write(f"Max Drawdown: {results.max_drawdown:.2%}\n")
                f.write(f"Win Rate: {results.win_rate:.2%}\n")
                f.write(f"Total Trades: {results.total_trades}\n")
            
            logger.info(f"Results saved to {summary_file}")
        
        logger.info("Backtest completed successfully")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 