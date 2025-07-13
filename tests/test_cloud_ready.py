"""
Test Cloud-Ready Components

Tests for monitoring, web server, and cloud deployment features.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.monitoring import monitor, TradingBotMonitor, HealthStatus
from src.utils.web_server import TradingBotWebServer


class TestMonitoring:
    """Test monitoring functionality."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        assert monitor is not None
        assert isinstance(monitor, TradingBotMonitor)
        assert monitor.start_time > 0
    
    @pytest.mark.asyncio
    async def test_system_health_check(self):
        """Test system health check."""
        health = await monitor.check_system_health()
        
        assert isinstance(health, dict)
        assert 'cpu_usage' in health
        assert 'memory_usage' in health
        assert 'disk_usage' in health
        assert 'uptime' in health
        
        # Check that values are reasonable
        assert 0 <= health['cpu_usage'] <= 100
        assert 0 <= health['memory_usage'] <= 100
        assert 0 <= health['disk_usage'] <= 100
        assert health['uptime'] > 0
    
    @pytest.mark.asyncio
    async def test_trading_health_check(self):
        """Test trading health check."""
        # Mock trading bot
        mock_bot = Mock()
        mock_bot.running = True
        mock_bot.total_signals = 10
        mock_bot.executed_trades = 5
        mock_bot.total_pnl = 100.0
        mock_bot.data_collector = Mock()
        mock_bot.signal_generator = Mock()
        mock_bot.order_manager = Mock()
        mock_bot.risk_manager = Mock()
        
        health = await monitor.check_trading_health(mock_bot)
        
        assert isinstance(health, dict)
        assert health['bot_running'] is True
        assert health['total_signals'] == 10
        assert health['executed_trades'] == 5
        assert health['total_pnl'] == 100.0
        assert 'components' in health
    
    @pytest.mark.asyncio
    async def test_external_services_check(self):
        """Test external services health check."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock successful responses
            mock_response = Mock()
            mock_response.status = 200
            mock_response.headers.get.return_value = 0.1
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            services = await monitor.check_external_services()
            
            assert isinstance(services, dict)
            assert 'binance' in services
            assert 'kraken' in services
    
    def test_metrics_recording(self):
        """Test metrics recording functions."""
        # Test signal recording
        monitor.record_signal('buy', 'BTC/USDT')
        
        # Test order recording
        monitor.record_order('market', 'ETH/USDT', 'filled')
        
        # Test PnL recording
        monitor.record_pnl('BTC/USDT', 150.0)
        
        # Test error recording
        monitor.record_error('connection_error', 'data_collector')
        
        # These should not raise exceptions
        assert True
    
    def test_metrics_generation(self):
        """Test metrics generation."""
        metrics = monitor.get_metrics()
        assert isinstance(metrics, str)
    
    def test_health_json_generation(self):
        """Test health JSON generation."""
        health_json = monitor.get_health_json()
        assert isinstance(health_json, str)
        
        # Should be valid JSON
        health_data = json.loads(health_json)
        assert 'status' in health_data
        assert 'timestamp' in health_data


class TestWebServer:
    """Test web server functionality."""
    
    @pytest.fixture
    def web_server(self):
        """Create web server instance."""
        return TradingBotWebServer(host='127.0.0.1', port=8081)
    
    def test_web_server_initialization(self, web_server):
        """Test web server initialization."""
        assert web_server.host == '127.0.0.1'
        assert web_server.port == 8081
        assert web_server.app is not None
        assert web_server.trading_bot is None
    
    def test_web_server_set_trading_bot(self, web_server):
        """Test setting trading bot instance."""
        mock_bot = Mock()
        web_server.set_trading_bot(mock_bot)
        assert web_server.trading_bot == mock_bot
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint(self, web_server):
        """Test health check endpoint."""
        # Mock request
        mock_request = Mock()
        
        response = await web_server.health_check(mock_request)
        
        assert response.status in [200, 503, 500]
        assert response.content_type == 'application/json'
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, web_server):
        """Test metrics endpoint."""
        mock_request = Mock()
        
        response = await web_server.metrics(mock_request)
        
        assert response.content_type == 'text/plain; version=0.0.4; charset=utf-8'
    
    @pytest.mark.asyncio
    async def test_status_endpoint(self, web_server):
        """Test status endpoint."""
        mock_request = Mock()
        
        response = await web_server.status(mock_request)
        
        assert response.status in [200, 500]
        assert response.content_type == 'application/json'
    
    @pytest.mark.asyncio
    async def test_trading_info_endpoint(self, web_server):
        """Test trading info endpoint."""
        mock_request = Mock()
        
        # Test without trading bot
        response = await web_server.trading_info(mock_request)
        assert response.status == 503
        
        # Test with trading bot
        mock_bot = Mock()
        mock_bot.running = True
        mock_bot.total_signals = 10
        mock_bot.executed_trades = 5
        mock_bot.total_pnl = 100.0
        mock_bot.start_time = None
        web_server.set_trading_bot(mock_bot)
        
        response = await web_server.trading_info(mock_request)
        assert response.status == 200
        assert response.content_type == 'application/json'
    
    @pytest.mark.asyncio
    async def test_system_info_endpoint(self, web_server):
        """Test system info endpoint."""
        mock_request = Mock()
        
        response = await web_server.system_info(mock_request)
        
        assert response.status in [200, 500]
        assert response.content_type == 'application/json'
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, web_server):
        """Test root endpoint."""
        mock_request = Mock()
        
        response = await web_server.root(mock_request)
        
        assert response.status in [200, 500]
        assert response.content_type == 'application/json'
    
    @pytest.mark.asyncio
    async def test_web_server_lifecycle(self, web_server):
        """Test web server start/stop lifecycle."""
        # Test start
        await web_server.start()
        assert web_server.runner is not None
        assert web_server.site is not None
        
        # Test stop
        await web_server.stop()
        assert web_server.runner is None
        assert web_server.site is None


class TestCloudIntegration:
    """Test cloud integration features."""
    
    def test_dockerfile_exists(self):
        """Test that Dockerfile exists and is valid."""
        dockerfile_path = os.path.join(os.path.dirname(__file__), '..', 'docker', 'Dockerfile')
        assert os.path.exists(dockerfile_path)
        
        with open(dockerfile_path, 'r') as f:
            content = f.read()
            assert 'FROM python:3.11-slim' in content
            assert 'WORKDIR /app' in content
            assert 'EXPOSE 8080' in content
    
    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists and is valid."""
        compose_path = os.path.join(os.path.dirname(__file__), '..', 'docker', 'docker-compose.yml')
        assert os.path.exists(compose_path)
        
        with open(compose_path, 'r') as f:
            content = f.read()
            assert 'version:' in content
            assert 'trading-bot:' in content
            assert 'postgres:' in content
            assert 'redis:' in content
    
    def test_kubernetes_manifests_exist(self):
        """Test that Kubernetes manifests exist."""
        k8s_dir = os.path.join(os.path.dirname(__file__), '..', 'k8s')
        assert os.path.exists(k8s_dir)
        
        required_files = [
            'namespace.yaml',
            'configmap.yaml',
            'secrets.yaml',
            'deployment.yaml',
            'services.yaml',
            'persistent-volumes.yaml'
        ]
        
        for file in required_files:
            file_path = os.path.join(k8s_dir, file)
            assert os.path.exists(file_path), f"Missing Kubernetes manifest: {file}"
    
    def test_deployment_script_exists(self):
        """Test that deployment script exists and is executable."""
        script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'deploy.sh')
        assert os.path.exists(script_path)
        assert os.access(script_path, os.X_OK)
    
    def test_requirements_include_cloud_deps(self):
        """Test that requirements.txt includes cloud dependencies."""
        requirements_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
        assert os.path.exists(requirements_path)
        
        with open(requirements_path, 'r') as f:
            content = f.read()
            assert 'aiohttp' in content
            assert 'prometheus-client' in content
            assert 'psutil' in content
            assert 'redis' in content
            assert 'psycopg2-binary' in content


if __name__ == "__main__":
    pytest.main([__file__]) 