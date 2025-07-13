"""
Backtesting Demo

Comprehensive demonstration of the backtesting framework for crypto trading strategies.
Shows how to run backtests, analyze results, and generate reports.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtesting.backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from backtesting.data_simulator import DataSimulator
from backtesting.performance_analyzer import PerformanceAnalyzer
from backtesting.strategy_runner import StrategyRunner
from utils.logger import logger


def create_backtest_config() -> BacktestConfig:
    """Create a comprehensive backtest configuration."""
    
    # Time period (last 6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    # Strategy configuration
    strategy_config = {
        'statistical_arbitrage': {
            'z_score_threshold': 2.0,
            'lookback_period': 100,
            'correlation_threshold': 0.7,
            'cointegration_pvalue_threshold': 0.05,
            'min_spread_std': 0.001,
            'position_size_limit': 0.1,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.1,
            'max_positions': 5,
            'rebalance_frequency': 300,
            'dynamic_threshold_window': 100,
            'spread_model': 'ols',
            'slippage': 0.001
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
            'require_risk_approval': False
        },
        'risk_management': {
            'max_position_size': 0.1,
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
        symbols=['ETH/USDT', 'SOL/USDT', 'BTC/USDT', 'ADA/USDT'],
        exchanges=['binance', 'kraken'],
        strategy_config=strategy_config,
        initial_capital=100000.0,
        max_position_size=0.1,
        stop_loss_pct=0.05,
        take_profit_pct=0.1,
        slippage=0.001,
        commission=0.001,
        data_frequency='1h',
        sentiment_enabled=True,
        sentiment_frequency='4h',
        track_metrics=True,
        generate_plots=True,
        save_results=True,
        results_dir='data/backtest_results'
    )


def run_basic_backtest():
    """Run a basic backtest with default configuration."""
    logger.info("Starting basic backtest...")
    
    # Create configuration
    config = create_backtest_config()
    
    # Create and run backtest
    engine = BacktestEngine(config)
    results = engine.run_backtest()
    
    # Print results summary
    print("\n" + "="*50)
    print("BACKTEST RESULTS SUMMARY")
    print("="*50)
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Annualized Return: {results.annualized_return:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
    print(f"Maximum Drawdown: {results.max_drawdown:.2%}")
    print(f"Win Rate: {results.win_rate:.2%}")
    print(f"Profit Factor: {results.profit_factor:.2f}")
    print(f"Total Trades: {results.total_trades}")
    print(f"Winning Trades: {results.winning_trades}")
    print(f"Losing Trades: {results.losing_trades}")
    print(f"Final Portfolio Value: ${results.final_portfolio_value:,.2f}")
    print(f"Peak Portfolio Value: ${results.peak_portfolio_value:,.2f}")
    print("="*50)
    
    return results


def run_parameter_optimization():
    """Run parameter optimization for statistical arbitrage."""
    logger.info("Starting parameter optimization...")
    
    # Define parameter ranges
    z_score_thresholds = [1.5, 2.0, 2.5, 3.0]
    lookback_periods = [50, 100, 150, 200]
    
    results_summary = []
    
    for z_threshold in z_score_thresholds:
        for lookback in lookback_periods:
            logger.info(f"Testing z_threshold={z_threshold}, lookback={lookback}")
            
            # Create configuration with current parameters
            config = create_backtest_config()
            config.strategy_config['statistical_arbitrage']['z_score_threshold'] = z_threshold
            config.strategy_config['statistical_arbitrage']['lookback_period'] = lookback
            
            # Run backtest
            engine = BacktestEngine(config)
            results = engine.run_backtest()
            
            # Store results
            results_summary.append({
                'z_threshold': z_threshold,
                'lookback': lookback,
                'total_return': results.total_return,
                'sharpe_ratio': results.sharpe_ratio,
                'max_drawdown': results.max_drawdown,
                'win_rate': results.win_rate,
                'profit_factor': results.profit_factor
            })
    
    # Find best parameters
    results_df = pd.DataFrame(results_summary)
    best_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
    
    print("\n" + "="*50)
    print("PARAMETER OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Best Sharpe Ratio: {best_sharpe['sharpe_ratio']:.3f}")
    print(f"Best Z-Score Threshold: {best_sharpe['z_threshold']}")
    print(f"Best Lookback Period: {best_sharpe['lookback']}")
    print(f"Best Total Return: {best_sharpe['total_return']:.2%}")
    print(f"Best Max Drawdown: {best_sharpe['max_drawdown']:.2%}")
    print("="*50)
    
    return results_df


def run_strategy_comparison():
    """Compare different strategy configurations."""
    logger.info("Starting strategy comparison...")
    
    # Define different strategy configurations
    strategies = {
        'Statistical Arbitrage Only': {
            'statistical_arbitrage': {
                'z_score_threshold': 2.0,
                'lookback_period': 100
            },
            'sentiment_analysis': None,
            'signal_generator': {
                'combination_method': 'stat_only'
            }
        },
        'Sentiment Only': {
            'statistical_arbitrage': None,
            'sentiment_analysis': {
                'confidence_threshold': 0.7
            },
            'signal_generator': {
                'combination_method': 'sentiment_only'
            }
        },
        'Combined Strategy': {
            'statistical_arbitrage': {
                'z_score_threshold': 2.0,
                'lookback_period': 100
            },
            'sentiment_analysis': {
                'confidence_threshold': 0.7
            },
            'signal_generator': {
                'combination_method': 'weighted',
                'stat_weight': 0.7,
                'sentiment_weight': 0.3
            }
        }
    }
    
    comparison_results = []
    
    for strategy_name, strategy_config in strategies.items():
        logger.info(f"Testing strategy: {strategy_name}")
        
        # Create configuration
        config = create_backtest_config()
        config.strategy_config = strategy_config
        
        # Run backtest
        engine = BacktestEngine(config)
        results = engine.run_backtest()
        
        # Store results
        comparison_results.append({
            'strategy': strategy_name,
            'total_return': results.total_return,
            'sharpe_ratio': results.sharpe_ratio,
            'max_drawdown': results.max_drawdown,
            'win_rate': results.win_rate,
            'profit_factor': results.profit_factor,
            'total_trades': results.total_trades
        })
    
    # Create comparison table
    comparison_df = pd.DataFrame(comparison_results)
    
    print("\n" + "="*60)
    print("STRATEGY COMPARISON RESULTS")
    print("="*60)
    print(comparison_df.to_string(index=False, float_format='%.3f'))
    print("="*60)
    
    return comparison_df


def run_risk_analysis():
    """Analyze risk characteristics of the strategy."""
    logger.info("Starting risk analysis...")
    
    # Create configuration
    config = create_backtest_config()
    
    # Run backtest
    engine = BacktestEngine(config)
    results = engine.run_backtest()
    
    # Calculate additional risk metrics
    returns = results.equity_curve.pct_change().dropna()
    
    risk_metrics = {
        'volatility': returns.std() * np.sqrt(252),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),
        'var_95': np.percentile(returns, 5),
        'cvar_95': returns[returns <= np.percentile(returns, 5)].mean(),
        'max_drawdown': results.max_drawdown,
        'calmar_ratio': results.annualized_return / abs(results.max_drawdown) if results.max_drawdown != 0 else 0,
        'sortino_ratio': results.annualized_return / (returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0
    }
    
    print("\n" + "="*50)
    print("RISK ANALYSIS")
    print("="*50)
    print(f"Volatility: {risk_metrics['volatility']:.2%}")
    print(f"Skewness: {risk_metrics['skewness']:.3f}")
    print(f"Kurtosis: {risk_metrics['kurtosis']:.3f}")
    print(f"Value at Risk (95%): {risk_metrics['var_95']:.2%}")
    print(f"Conditional VaR (95%): {risk_metrics['cvar_95']:.2%}")
    print(f"Calmar Ratio: {risk_metrics['calmar_ratio']:.3f}")
    print(f"Sortino Ratio: {risk_metrics['sortino_ratio']:.3f}")
    print("="*50)
    
    return risk_metrics


def generate_comprehensive_report():
    """Generate a comprehensive backtesting report."""
    logger.info("Generating comprehensive report...")
    
    # Create configuration
    config = create_backtest_config()
    
    # Run backtest
    engine = BacktestEngine(config)
    results = engine.run_backtest()
    
    # Create performance analyzer
    analyzer = PerformanceAnalyzer()
    
    # Analyze performance
    analysis_results = analyzer.analyze_performance(
        results.equity_curve,
        results.trade_history,
        results.position_history,
        results.signal_history
    )
    
    # Generate report
    report_path = analyzer.generate_report(analysis_results, config.results_dir)
    
    # Generate plots
    plot_paths = analyzer.generate_plots(
        results.equity_curve,
        results.trade_history,
        config.results_dir
    )
    
    print(f"\nComprehensive report generated: {report_path}")
    print(f"Plots generated: {len(plot_paths)} files")
    
    return analysis_results


def main():
    """Main function to run all backtesting demonstrations."""
    print("Crypto Trading Bot - Backtesting Demo")
    print("="*50)
    
    try:
        # Run basic backtest
        print("\n1. Running Basic Backtest...")
        basic_results = run_basic_backtest()
        
        # Run parameter optimization
        print("\n2. Running Parameter Optimization...")
        optimization_results = run_parameter_optimization()
        
        # Run strategy comparison
        print("\n3. Running Strategy Comparison...")
        comparison_results = run_strategy_comparison()
        
        # Run risk analysis
        print("\n4. Running Risk Analysis...")
        risk_metrics = run_risk_analysis()
        
        # Generate comprehensive report
        print("\n5. Generating Comprehensive Report...")
        analysis_results = generate_comprehensive_report()
        
        print("\n" + "="*50)
        print("BACKTESTING DEMO COMPLETED SUCCESSFULLY")
        print("="*50)
        print("Results saved to: data/backtest_results/")
        print("Check the generated files for detailed analysis.")
        
    except Exception as e:
        logger.error(f"Backtesting demo failed: {e}")
        raise


if __name__ == "__main__":
    main() 