"""
Monitoring and Health Check Module

This module provides monitoring capabilities for the crypto trading bot,
including health checks, metrics collection, and Prometheus integration.
"""

import asyncio
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager
import json

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest
except ImportError:
    # Fallback if prometheus_client is not available
    class MockMetric:
        def __init__(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
    
    Counter = Gauge = Histogram = Summary = MockMetric
    generate_latest = lambda: b""

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health status data class."""
    status: str
    timestamp: datetime
    details: Dict[str, Any]
    checks: Dict[str, bool]


class TradingBotMonitor:
    """
    Monitoring class for the crypto trading bot.
    
    Provides health checks, metrics collection, and monitoring capabilities.
    """
    
    def __init__(self):
        """Initialize the monitoring system."""
        self.logger = logger
        self.start_time = time.time()
        self.health_checks = {}
        self.metrics = {}
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        
        # Health check thresholds
        self.thresholds = {
            'cpu_usage': 80.0,  # 80% CPU usage
            'memory_usage': 85.0,  # 85% memory usage
            'disk_usage': 90.0,  # 90% disk usage
            'response_time': 5.0,  # 5 seconds
            'error_rate': 0.1,  # 10% error rate
        }
        
        self.logger.info("TradingBotMonitor initialized")
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        try:
            # Trading metrics
            self.trading_signals_total = Counter(
                'trading_signals_total',
                'Total number of trading signals generated',
                ['signal_type', 'symbol']
            )
            
            self.trading_orders_total = Counter(
                'trading_orders_total',
                'Total number of orders executed',
                ['order_type', 'symbol', 'status']
            )
            
            self.trading_pnl = Gauge(
                'trading_pnl',
                'Current portfolio P&L',
                ['symbol']
            )
            
            self.trading_position_size = Gauge(
                'trading_position_size',
                'Current position size',
                ['symbol', 'side']
            )
            
            # System metrics
            self.system_cpu_usage = Gauge(
                'system_cpu_usage',
                'System CPU usage percentage'
            )
            
            self.system_memory_usage = Gauge(
                'system_memory_usage',
                'System memory usage percentage'
            )
            
            self.system_disk_usage = Gauge(
                'system_disk_usage',
                'System disk usage percentage'
            )
            
            # Performance metrics
            self.api_response_time = Histogram(
                'api_response_time',
                'API response time in seconds',
                ['endpoint']
            )
            
            self.data_collection_duration = Histogram(
                'data_collection_duration',
                'Data collection duration in seconds',
                ['data_type']
            )
            
            self.signal_generation_duration = Histogram(
                'signal_generation_duration',
                'Signal generation duration in seconds',
                ['strategy']
            )
            
            # Error metrics
            self.errors_total = Counter(
                'errors_total',
                'Total number of errors',
                ['error_type', 'module']
            )
            
            self.logger.info("Prometheus metrics initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Prometheus metrics: {e}")
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Check system health metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.system_memory_usage.set(memory_percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.system_disk_usage.set(disk_percent)
            
            # Network I/O
            network = psutil.net_io_counters()
            
            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'disk_usage': disk_percent,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'uptime': time.time() - self.start_time
            }
            
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
            return {'error': str(e)}
    
    async def check_trading_health(self, trading_bot) -> Dict[str, Any]:
        """Check trading-specific health metrics."""
        try:
            health_data = {
                'bot_running': trading_bot.running if hasattr(trading_bot, 'running') else False,
                'total_signals': getattr(trading_bot, 'total_signals', 0),
                'executed_trades': getattr(trading_bot, 'executed_trades', 0),
                'total_pnl': getattr(trading_bot, 'total_pnl', 0.0),
            }
            
            # Check component health
            components = {}
            for component_name in ['data_collector', 'signal_generator', 'order_manager', 'risk_manager']:
                component = getattr(trading_bot, component_name, None)
                components[component_name] = component is not None
            
            health_data['components'] = components
            
            return health_data
            
        except Exception as e:
            self.logger.error(f"Error checking trading health: {e}")
            return {'error': str(e)}
    
    async def check_database_health(self, db_connection=None) -> Dict[str, Any]:
        """Check database connectivity and health."""
        try:
            if db_connection:
                # Test database connection
                cursor = db_connection.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                
                return {
                    'connected': True,
                    'response_time': 0.001  # Placeholder
                }
            else:
                return {
                    'connected': False,
                    'error': 'No database connection provided'
                }
                
        except Exception as e:
            self.logger.error(f"Error checking database health: {e}")
            return {
                'connected': False,
                'error': str(e)
            }
    
    async def check_external_services(self) -> Dict[str, Any]:
        """Check external service connectivity."""
        try:
            import aiohttp
            
            services = {}
            
            # Check exchange APIs
            exchanges = ['binance', 'kraken']
            for exchange in exchanges:
                try:
                    async with aiohttp.ClientSession() as session:
                        if exchange == 'binance':
                            url = 'https://api.binance.com/api/v3/ping'
                        elif exchange == 'kraken':
                            url = 'https://api.kraken.com/0/public/Time'
                        
                        async with session.get(url, timeout=5) as response:
                            services[exchange] = {
                                'status': 'healthy' if response.status == 200 else 'unhealthy',
                                'response_time': response.headers.get('X-Response-Time', 0)
                            }
                except Exception as e:
                    services[exchange] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            return services
            
        except Exception as e:
            self.logger.error(f"Error checking external services: {e}")
            return {'error': str(e)}
    
    async def get_health_status(self, trading_bot=None) -> HealthStatus:
        """Get comprehensive health status."""
        try:
            checks = {}
            details = {}
            
            # System health
            system_health = await self.check_system_health()
            details['system'] = system_health
            checks['system'] = (
                system_health.get('cpu_usage', 0) < self.thresholds['cpu_usage'] and
                system_health.get('memory_usage', 0) < self.thresholds['memory_usage'] and
                system_health.get('disk_usage', 0) < self.thresholds['disk_usage']
            )
            
            # Trading health
            if trading_bot:
                trading_health = await self.check_trading_health(trading_bot)
                details['trading'] = trading_health
                checks['trading'] = trading_health.get('bot_running', False)
            
            # Database health
            db_health = await self.check_database_health()
            details['database'] = db_health
            checks['database'] = db_health.get('connected', False)
            
            # External services
            external_health = await self.check_external_services()
            details['external_services'] = external_health
            checks['external_services'] = all(
                service.get('status') == 'healthy' 
                for service in external_health.values() 
                if isinstance(service, dict)
            )
            
            # Overall status
            overall_healthy = all(checks.values())
            status = 'healthy' if overall_healthy else 'unhealthy'
            
            return HealthStatus(
                status=status,
                timestamp=datetime.now(),
                details=details,
                checks=checks
            )
            
        except Exception as e:
            self.logger.error(f"Error getting health status: {e}")
            return HealthStatus(
                status='error',
                timestamp=datetime.now(),
                details={'error': str(e)},
                checks={}
            )
    
    def record_signal(self, signal_type: str, symbol: str):
        """Record a trading signal."""
        try:
            self.trading_signals_total.labels(signal_type=signal_type, symbol=symbol).inc()
        except Exception as e:
            self.logger.warning(f"Failed to record signal metric: {e}")
    
    def record_order(self, order_type: str, symbol: str, status: str):
        """Record an order execution."""
        try:
            self.trading_orders_total.labels(order_type=order_type, symbol=symbol, status=status).inc()
        except Exception as e:
            self.logger.warning(f"Failed to record order metric: {e}")
    
    def record_pnl(self, symbol: str, pnl: float):
        """Record P&L for a symbol."""
        try:
            self.trading_pnl.labels(symbol=symbol).set(pnl)
        except Exception as e:
            self.logger.warning(f"Failed to record P&L metric: {e}")
    
    def record_error(self, error_type: str, module: str):
        """Record an error."""
        try:
            self.errors_total.labels(error_type=error_type, module=module).inc()
        except Exception as e:
            self.logger.warning(f"Failed to record error metric: {e}")
    
    @asynccontextmanager
    async def measure_api_call(self, endpoint: str):
        """Context manager to measure API call duration."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            try:
                self.api_response_time.labels(endpoint=endpoint).observe(duration)
            except Exception as e:
                self.logger.warning(f"Failed to record API call metric: {e}")
    
    @asynccontextmanager
    async def measure_data_collection(self, data_type: str):
        """Context manager to measure data collection duration."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            try:
                self.data_collection_duration.labels(data_type=data_type).observe(duration)
            except Exception as e:
                self.logger.warning(f"Failed to record data collection metric: {e}")
    
    @asynccontextmanager
    async def measure_signal_generation(self, strategy: str):
        """Context manager to measure signal generation duration."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            try:
                self.signal_generation_duration.labels(strategy=strategy).observe(duration)
            except Exception as e:
                self.logger.warning(f"Failed to record signal generation metric: {e}")
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics."""
        try:
            return generate_latest().decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error generating metrics: {e}")
            return ""
    
    def get_health_json(self, trading_bot=None) -> str:
        """Get health status as JSON."""
        try:
            health_status = asyncio.run(self.get_health_status(trading_bot))
            return json.dumps({
                'status': health_status.status,
                'timestamp': health_status.timestamp.isoformat(),
                'details': health_status.details,
                'checks': health_status.checks
            }, indent=2)
        except Exception as e:
            self.logger.error(f"Error generating health JSON: {e}")
            return json.dumps({
                'status': 'error',
                'error': str(e)
            })


# Global monitor instance
monitor = TradingBotMonitor() 