#!/bin/bash

# Crypto Trading Bot Development Setup Script

echo "🚀 Setting up Crypto Trading Bot development environment..."

# Activate virtual environment
source myenv/bin/activate

# Install additional dependencies
echo "📦 Installing additional dependencies..."
pip install flask plotly dash prometheus-client

# Set up Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p data/backtest_results
mkdir -p data/plots

# Set up configuration
echo "⚙️ Setting up configuration..."
if [ ! -f config/secrets.env ]; then
    cp config/secrets.env.example config/secrets.env
    echo "📝 Created config/secrets.env - please edit with your API keys"
fi

# Run tests to verify setup
echo "🧪 Running tests to verify setup..."
python3 -m pytest tests/ -v --tb=short

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Edit config/secrets.env with your API keys"
echo "2. Run examples: python3 examples/stat_arb_demo.py"
echo "3. Start dashboard: python3 dashboard.py"
echo "4. Run backtest: python3 run_backtest.py"
echo ""
echo "📚 Available commands:"
echo "- python3 examples/stat_arb_demo.py"
echo "- python3 examples/sentiment_demo.py"
echo "- python3 examples/order_manager_demo.py"
echo "- python3 examples/backtest_demo.py"
echo "- python3 dashboard.py"
echo "- python3 run_bot.py" 