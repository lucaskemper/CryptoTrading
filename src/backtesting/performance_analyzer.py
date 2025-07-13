"""
Performance Analyzer for Backtesting

Comprehensive performance analysis and reporting for crypto trading strategies.
Calculates risk metrics, performance ratios, and generates detailed reports.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from src.utils.logger import logger


class PerformanceAnalyzer:
    """
    Analyzes trading performance and generates comprehensive reports.
    
    Features:
    - Risk metrics calculation (Sharpe ratio, Sortino ratio, etc.)
    - Drawdown analysis
    - Trade analysis and statistics
    - Performance attribution
    - Risk-adjusted returns
    """
    
    def __init__(self):
        """Initialize the performance analyzer."""
        self.logger = logger
        
        # Risk-free rate for calculations
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Performance metrics storage
        self.metrics: Dict[str, float] = {}
        self.trade_analysis: Dict[str, Any] = {}
        self.risk_metrics: Dict[str, float] = {}
        
        self.logger.info("PerformanceAnalyzer initialized")
    
    def analyze_performance(self, equity_curve: pd.Series, trade_history: List[Dict], 
                          position_history: List[Dict], signal_history: List[Dict]) -> Dict[str, Any]:
        """
        Perform comprehensive performance analysis.
        
        Args:
            equity_curve: Portfolio value over time
            trade_history: List of executed trades
            position_history: Position tracking over time
            signal_history: Signal generation history
            
        Returns:
            Dictionary containing all performance metrics and analysis
        """
        self.logger.info("Starting performance analysis...")
        
        # Calculate basic metrics
        self._calculate_basic_metrics(equity_curve)
        
        # Calculate risk metrics
        self._calculate_risk_metrics(equity_curve)
        
        # Analyze trades
        self._analyze_trades(trade_history)
        
        # Analyze positions
        self._analyze_positions(position_history)
        
        # Analyze signals
        self._analyze_signals(signal_history)
        
        # Calculate performance attribution
        self._calculate_performance_attribution(equity_curve, trade_history)
        
        # Compile results
        results = {
            'basic_metrics': self.metrics,
            'risk_metrics': self.risk_metrics,
            'trade_analysis': self.trade_analysis,
            'position_analysis': self._analyze_positions(position_history),
            'signal_analysis': self._analyze_signals(signal_history),
            'performance_attribution': self._calculate_performance_attribution(equity_curve, trade_history)
        }
        
        self.logger.info("Performance analysis completed")
        return results
    
    def _calculate_basic_metrics(self, equity_curve: pd.Series):
        """Calculate basic performance metrics."""
        if len(equity_curve) < 2:
            return
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        annualized_return = self._calculate_annualized_return(equity_curve)
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Store metrics
        self.metrics.update({
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'final_value': equity_curve.iloc[-1],
            'initial_value': equity_curve.iloc[0],
            'peak_value': equity_curve.max()
        })
    
    def _calculate_risk_metrics(self, equity_curve: pd.Series):
        """Calculate risk-adjusted performance metrics."""
        if len(equity_curve) < 2:
            return
        
        returns = equity_curve.pct_change().dropna()
        
        # Sharpe ratio
        excess_returns = returns - self.risk_free_rate / 252  # Daily risk-free rate
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio (annualized return / max drawdown)
        annualized_return = self._calculate_annualized_return(equity_curve)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Information ratio (assuming benchmark is risk-free rate)
        information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Store risk metrics
        self.risk_metrics.update({
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'downside_deviation': downside_deviation,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        })
    
    def _analyze_trades(self, trade_history: List[Dict]):
        """Analyze trade performance and statistics."""
        if not trade_history:
            return
        
        # Convert to DataFrame for analysis
        trades_df = pd.DataFrame(trade_history)
        
        # Calculate trade P&L (simplified)
        trades_df['pnl'] = trades_df.apply(self._calculate_trade_pnl, axis=1)
        
        # Basic trade statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L statistics
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        largest_win = trades_df['pnl'].max()
        largest_loss = trades_df['pnl'].min()
        
        # Profit factor
        total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        total_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Average trade duration
        if 'timestamp' in trades_df.columns:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trade_durations = trades_df.groupby('symbol')['timestamp'].apply(
                lambda x: (x.max() - x.min()).total_seconds() / 3600  # Hours
            )
            avg_trade_duration = trade_durations.mean()
        else:
            avg_trade_duration = 0
        
        # Store trade analysis
        self.trade_analysis.update({
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_trade_duration,
            'trades_by_symbol': trades_df.groupby('symbol').size().to_dict(),
            'pnl_by_symbol': trades_df.groupby('symbol')['pnl'].sum().to_dict()
        })
    
    def _calculate_trade_pnl(self, trade: pd.Series) -> float:
        """Calculate P&L for a single trade."""
        # Simplified P&L calculation
        # In a real implementation, this would be more complex
        quantity = trade.get('quantity', 0)
        price = trade.get('price', 0)
        side = trade.get('side', 'buy')
        
        if side == 'buy':
            return -quantity * price  # Simplified: assume we're buying
        else:
            return quantity * price  # Simplified: assume we're selling
    
    def _analyze_positions(self, position_history: List[Dict]) -> Dict[str, Any]:
        """Analyze position performance and characteristics."""
        if not position_history:
            return {}
        
        # Convert to DataFrame
        positions_df = pd.DataFrame(position_history)
        
        # Calculate position metrics
        avg_positions = positions_df['positions'].apply(len).mean()
        max_positions = positions_df['positions'].apply(len).max()
        
        # Position duration analysis
        if 'timestamp' in positions_df.columns:
            positions_df['timestamp'] = pd.to_datetime(positions_df['timestamp'])
            position_durations = []
            
            for symbol in set([pos for pos_list in positions_df['positions'] for pos in pos_list.keys()]):
                symbol_positions = positions_df[positions_df['positions'].apply(lambda x: symbol in x)]
                if len(symbol_positions) > 1:
                    duration = (symbol_positions['timestamp'].max() - symbol_positions['timestamp'].min()).total_seconds() / 3600
                    position_durations.append(duration)
            
            avg_position_duration = np.mean(position_durations) if position_durations else 0
        else:
            avg_position_duration = 0
        
        return {
            'avg_positions': avg_positions,
            'max_positions': max_positions,
            'avg_position_duration': avg_position_duration,
            'position_turnover': len(position_history) / max(avg_positions, 1)
        }
    
    def _analyze_signals(self, signal_history: List[Dict]) -> Dict[str, Any]:
        """Analyze signal generation and quality."""
        if not signal_history:
            return {}
        
        # Convert to DataFrame
        signals_df = pd.DataFrame(signal_history)
        
        # Signal statistics
        total_signals = len(signals_df)
        
        # Signal by strategy
        if 'signal' in signals_df.columns:
            strategies = []
            for signal in signals_df['signal']:
                if isinstance(signal, dict):
                    strategies.append(signal.get('strategy', 'unknown'))
                else:
                    strategies.append('unknown')
            
            signals_df['strategy'] = strategies
            signals_by_strategy = signals_df.groupby('strategy').size().to_dict()
        else:
            signals_by_strategy = {}
        
        # Signal frequency
        if 'timestamp' in signals_df.columns:
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            signal_frequency = len(signals_df) / ((signals_df['timestamp'].max() - signals_df['timestamp'].min()).total_seconds() / 3600)
        else:
            signal_frequency = 0
        
        return {
            'total_signals': total_signals,
            'signals_by_strategy': signals_by_strategy,
            'signal_frequency': signal_frequency
        }
    
    def _calculate_performance_attribution(self, equity_curve: pd.Series, trade_history: List[Dict]) -> Dict[str, Any]:
        """Calculate performance attribution analysis."""
        if not trade_history:
            return {}
        
        # Convert to DataFrame
        trades_df = pd.DataFrame(trade_history)
        
        # Attribution by symbol
        if 'symbol' in trades_df.columns:
            symbol_attribution = trades_df.groupby('symbol')['pnl'].sum().to_dict()
        else:
            symbol_attribution = {}
        
        # Attribution by strategy
        if 'strategy' in trades_df.columns:
            strategy_attribution = trades_df.groupby('strategy')['pnl'].sum().to_dict()
        else:
            strategy_attribution = {}
        
        # Time-based attribution (monthly)
        if 'timestamp' in trades_df.columns:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df['month'] = trades_df['timestamp'].dt.to_period('M')
            monthly_attribution = trades_df.groupby('month')['pnl'].sum().to_dict()
        else:
            monthly_attribution = {}
        
        return {
            'symbol_attribution': symbol_attribution,
            'strategy_attribution': strategy_attribution,
            'monthly_attribution': monthly_attribution
        }
    
    def _calculate_annualized_return(self, equity_curve: pd.Series) -> float:
        """Calculate annualized return."""
        if len(equity_curve) < 2:
            return 0.0
        
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        
        # Calculate time period in years
        if hasattr(equity_curve.index, 'name') and equity_curve.index.name == 'timestamp':
            time_period = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        else:
            time_period = len(equity_curve) / 252  # Assume daily data
        
        if time_period > 0:
            annualized_return = ((1 + total_return) ** (1 / time_period)) - 1
        else:
            annualized_return = 0.0
        
        return annualized_return
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) < 2:
            return 0.0
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        return drawdown.min()
    
    def generate_report(self, results: Dict[str, Any], output_dir: str = None) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            results: Performance analysis results
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        self.logger.info("Generating performance report...")
        
        # Create report content
        report_content = self._create_report_content(results)
        
        # Save report
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            report_path = output_path / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            self.logger.info(f"Performance report saved to {report_path}")
            return str(report_path)
        else:
            return report_content
    
    def _create_report_content(self, results: Dict[str, Any]) -> str:
        """Create HTML report content."""
        basic_metrics = results.get('basic_metrics', {})
        risk_metrics = results.get('risk_metrics', {})
        trade_analysis = results.get('trade_analysis', {})
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Trading Performance Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Basic Performance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Return</td><td class="{'positive' if basic_metrics.get('total_return', 0) > 0 else 'negative'}">{basic_metrics.get('total_return', 0):.2%}</td></tr>
                    <tr><td>Annualized Return</td><td class="{'positive' if basic_metrics.get('annualized_return', 0) > 0 else 'negative'}">{basic_metrics.get('annualized_return', 0):.2%}</td></tr>
                    <tr><td>Volatility</td><td>{basic_metrics.get('volatility', 0):.2%}</td></tr>
                    <tr><td>Maximum Drawdown</td><td class="negative">{basic_metrics.get('max_drawdown', 0):.2%}</td></tr>
                    <tr><td>Final Portfolio Value</td><td>${basic_metrics.get('final_value', 0):,.2f}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Risk Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Sharpe Ratio</td><td>{risk_metrics.get('sharpe_ratio', 0):.3f}</td></tr>
                    <tr><td>Sortino Ratio</td><td>{risk_metrics.get('sortino_ratio', 0):.3f}</td></tr>
                    <tr><td>Calmar Ratio</td><td>{risk_metrics.get('calmar_ratio', 0):.3f}</td></tr>
                    <tr><td>Information Ratio</td><td>{risk_metrics.get('information_ratio', 0):.3f}</td></tr>
                    <tr><td>Value at Risk (95%)</td><td>{risk_metrics.get('var_95', 0):.2%}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Trade Analysis</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Trades</td><td>{trade_analysis.get('total_trades', 0)}</td></tr>
                    <tr><td>Win Rate</td><td>{trade_analysis.get('win_rate', 0):.2%}</td></tr>
                    <tr><td>Profit Factor</td><td>{trade_analysis.get('profit_factor', 0):.2f}</td></tr>
                    <tr><td>Average Win</td><td>${trade_analysis.get('avg_win', 0):.2f}</td></tr>
                    <tr><td>Average Loss</td><td>${trade_analysis.get('avg_loss', 0):.2f}</td></tr>
                    <tr><td>Largest Win</td><td>${trade_analysis.get('largest_win', 0):.2f}</td></tr>
                    <tr><td>Largest Loss</td><td>${trade_analysis.get('largest_loss', 0):.2f}</td></tr>
                </table>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def generate_plots(self, equity_curve: pd.Series, trade_history: List[Dict], 
                      output_dir: str = None) -> List[str]:
        """
        Generate performance visualization plots.
        
        Args:
            equity_curve: Portfolio value over time
            trade_history: List of executed trades
            output_dir: Directory to save plots
            
        Returns:
            List of plot file paths
        """
        plot_paths = []
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Equity Curve
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(equity_curve.index, equity_curve.values)
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        
        # Plot 2: Drawdown
        plt.subplot(2, 2, 2)
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        plt.fill_between(equity_curve.index, drawdown.values, 0, alpha=0.3, color='red')
        plt.title('Portfolio Drawdown')
        plt.xlabel('Time')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        # Plot 3: Trade P&L Distribution
        if trade_history:
            plt.subplot(2, 2, 3)
            trades_df = pd.DataFrame(trade_history)
            if 'pnl' in trades_df.columns:
                plt.hist(trades_df['pnl'], bins=20, alpha=0.7)
                plt.title('Trade P&L Distribution')
                plt.xlabel('P&L ($)')
                plt.ylabel('Frequency')
                plt.grid(True)
        
        # Plot 4: Monthly Returns
        plt.subplot(2, 2, 4)
        returns = equity_curve.pct_change().dropna()
        if len(returns) > 0:
            monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
            if len(monthly_returns) > 0:
                plt.bar(range(len(monthly_returns)), monthly_returns.values)
                plt.title('Monthly Returns')
                plt.xlabel('Month')
                plt.ylabel('Return (%)')
                plt.grid(True)
        
        plt.tight_layout()
        
        if output_dir:
            plot_path = output_path / 'performance_plots.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_paths.append(str(plot_path))
        
        plt.close()
        
        return plot_paths


def create_performance_analyzer() -> PerformanceAnalyzer:
    """Factory function to create a performance analyzer."""
    return PerformanceAnalyzer() 