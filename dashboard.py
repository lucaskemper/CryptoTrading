#!/usr/bin/env python3
"""
Real-time Trading Dashboard
Monitor your crypto trading bot performance and positions
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify
import threading
import queue

app = Flask(__name__)

# Global state for dashboard data
dashboard_data = {
    'portfolio_value': 100000.0,
    'total_pnl': 0.0,
    'open_positions': [],
    'recent_trades': [],
    'performance_metrics': {
        'win_rate': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0
    },
    'signals_generated': 0,
    'last_update': datetime.now().isoformat()
}

def update_dashboard_data():
    """Update dashboard data from bot state"""
    # This would connect to your bot's data sources
    # For now, we'll simulate updates
    pass

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html', data=dashboard_data)

@app.route('/api/status')
def api_status():
    """API endpoint for real-time status"""
    return jsonify(dashboard_data)

@app.route('/api/positions')
def api_positions():
    """API endpoint for current positions"""
    return jsonify({
        'positions': dashboard_data['open_positions'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/trades')
def api_trades():
    """API endpoint for recent trades"""
    return jsonify({
        'trades': dashboard_data['recent_trades'],
        'timestamp': datetime.now().isoformat()
    })

def create_dashboard_templates():
    """Create HTML template for dashboard"""
    template_dir = "templates"
    os.makedirs(template_dir, exist_ok=True)
    
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Crypto Trading Bot Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .positions { background: white; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .trades { background: white; padding: 20px; border-radius: 5px; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Crypto Trading Bot Dashboard</h1>
            <p>Real-time monitoring and performance tracking</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <h3>Portfolio Value</h3>
                <div class="metric-value" id="portfolio-value">${{ "%.2f"|format(data.portfolio_value) }}</div>
            </div>
            <div class="metric-card">
                <h3>Total PnL</h3>
                <div class="metric-value" id="total-pnl">{{ "%.2f"|format(data.total_pnl) }}%</div>
            </div>
            <div class="metric-card">
                <h3>Win Rate</h3>
                <div class="metric-value" id="win-rate">{{ "%.1f"|format(data.performance_metrics.win_rate * 100) }}%</div>
            </div>
            <div class="metric-card">
                <h3>Sharpe Ratio</h3>
                <div class="metric-value" id="sharpe-ratio">{{ "%.3f"|format(data.performance_metrics.sharpe_ratio) }}</div>
            </div>
        </div>
        
        <div class="positions">
            <h2>Open Positions</h2>
            <table id="positions-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Quantity</th>
                        <th>Entry Price</th>
                        <th>Current Price</th>
                        <th>PnL</th>
                    </tr>
                </thead>
                <tbody id="positions-body">
                    {% for position in data.open_positions %}
                    <tr>
                        <td>{{ position.symbol }}</td>
                        <td>{{ position.side }}</td>
                        <td>{{ "%.4f"|format(position.quantity) }}</td>
                        <td>${{ "%.2f"|format(position.entry_price) }}</td>
                        <td>${{ "%.2f"|format(position.current_price) }}</td>
                        <td class="{{ 'positive' if position.pnl > 0 else 'negative' }}">{{ "%.2f"|format(position.pnl) }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <div class="trades">
            <h2>Recent Trades</h2>
            <table id="trades-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Quantity</th>
                        <th>Price</th>
                        <th>PnL</th>
                    </tr>
                </thead>
                <tbody id="trades-body">
                    {% for trade in data.recent_trades %}
                    <tr>
                        <td>{{ trade.time }}</td>
                        <td>{{ trade.symbol }}</td>
                        <td>{{ trade.side }}</td>
                        <td>{{ "%.4f"|format(trade.quantity) }}</td>
                        <td>${{ "%.2f"|format(trade.price) }}</td>
                        <td class="{{ 'positive' if trade.pnl > 0 else 'negative' }}">{{ "%.2f"|format(trade.pnl) }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        // Auto-refresh dashboard data every 30 seconds
        setInterval(function() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('portfolio-value').textContent = '$' + data.portfolio_value.toFixed(2);
                    document.getElementById('total-pnl').textContent = data.total_pnl.toFixed(2) + '%';
                    document.getElementById('win-rate').textContent = (data.performance_metrics.win_rate * 100).toFixed(1) + '%';
                    document.getElementById('sharpe-ratio').textContent = data.performance_metrics.sharpe_ratio.toFixed(3);
                });
        }, 30000);
    </script>
</body>
</html>
"""
    
    with open(f"{template_dir}/dashboard.html", "w") as f:
        f.write(html_template)

if __name__ == '__main__':
    import os
    import sys
    
    # Allow custom port
    port = 5001
    if len(sys.argv) > 1 and sys.argv[1] == '--port':
        port = int(sys.argv[2])
    
    create_dashboard_templates()
    print("🚀 Starting Crypto Trading Bot Dashboard...")
    print(f"📊 Dashboard available at: http://localhost:{port}")
    app.run(debug=True, host='0.0.0.0', port=port) 