"""
Web Server Module for Trading Bot

Provides HTTP endpoints for health checks, metrics, and basic monitoring.
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any
from aiohttp import web, ClientSession
from aiohttp.web import Request, Response
import time

from .monitoring import monitor

logger = logging.getLogger(__name__)


class TradingBotWebServer:
    """
    Web server for the crypto trading bot.
    
    Provides health checks, metrics, and basic monitoring endpoints.
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080, trading_bot=None):
        """Initialize the web server."""
        self.host = host
        self.port = port
        self.trading_bot = trading_bot
        self.app = web.Application()
        self.runner = None
        self.site = None
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"TradingBotWebServer initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Setup HTTP routes."""
        # Health check endpoint
        self.app.router.add_get('/health', self.health_check)
        
        # Metrics endpoint (Prometheus format)
        self.app.router.add_get('/metrics', self.metrics)
        
        # Status endpoint (JSON format)
        self.app.router.add_get('/status', self.status)
        
        # Trading info endpoint
        self.app.router.add_get('/trading/info', self.trading_info)
        
        # System info endpoint
        self.app.router.add_get('/system/info', self.system_info)
        
        # Root endpoint
        self.app.router.add_get('/', self.root)
        
        # Add CORS middleware
        self.app.middlewares.append(self._cors_middleware)
    
    async def _cors_middleware(self, app, handler):
        """CORS middleware."""
        async def middleware(request):
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response
        return middleware
    
    async def health_check(self, request: Request) -> Response:
        """Health check endpoint."""
        try:
            health_status = await monitor.get_health_status(self.trading_bot)
            
            if health_status.status == 'healthy':
                status_code = 200
            elif health_status.status == 'unhealthy':
                status_code = 503
            else:
                status_code = 500
            
            response_data = {
                'status': health_status.status,
                'timestamp': health_status.timestamp.isoformat(),
                'checks': health_status.checks
            }
            
            return web.json_response(response_data, status=status_code)
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return web.json_response({
                'status': 'error',
                'error': str(e)
            }, status=500)
    
    async def metrics(self, request: Request) -> Response:
        """Prometheus metrics endpoint."""
        try:
            metrics_data = monitor.get_metrics()
            return web.Response(
                text=metrics_data,
                content_type='text/plain; version=0.0.4; charset=utf-8'
            )
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return web.Response(
                text="# Error getting metrics\n",
                content_type='text/plain; version=0.0.4; charset=utf-8',
                status=500
            )
    
    async def status(self, request: Request) -> Response:
        """Detailed status endpoint."""
        try:
            health_status = await monitor.get_health_status(self.trading_bot)
            
            status_data = {
                'status': health_status.status,
                'timestamp': health_status.timestamp.isoformat(),
                'details': health_status.details,
                'checks': health_status.checks,
                'uptime': time.time() - monitor.start_time
            }
            
            return web.json_response(status_data)
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return web.json_response({
                'status': 'error',
                'error': str(e)
            }, status=500)
    
    async def trading_info(self, request: Request) -> Response:
        """Trading-specific information endpoint."""
        try:
            if not self.trading_bot:
                return web.json_response({
                    'error': 'Trading bot not available'
                }, status=503)
            
            trading_info = {
                'running': getattr(self.trading_bot, 'running', False),
                'total_signals': getattr(self.trading_bot, 'total_signals', 0),
                'executed_trades': getattr(self.trading_bot, 'executed_trades', 0),
                'total_pnl': getattr(self.trading_bot, 'total_pnl', 0.0),
                'start_time': getattr(self.trading_bot, 'start_time', None),
                'components': {}
            }
            
            # Component status
            for component_name in ['data_collector', 'signal_generator', 'order_manager', 'risk_manager']:
                component = getattr(self.trading_bot, component_name, None)
                trading_info['components'][component_name] = {
                    'available': component is not None,
                    'type': type(component).__name__ if component else None
                }
            
            return web.json_response(trading_info)
            
        except Exception as e:
            logger.error(f"Error getting trading info: {e}")
            return web.json_response({
                'error': str(e)
            }, status=500)
    
    async def system_info(self, request: Request) -> Response:
        """System information endpoint."""
        try:
            system_health = await monitor.check_system_health()
            
            system_info = {
                'system': system_health,
                'thresholds': monitor.thresholds
            }
            
            return web.json_response(system_info)
            
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return web.json_response({
                'error': str(e)
            }, status=500)
    
    async def root(self, request: Request) -> Response:
        """Root endpoint with basic information."""
        try:
            root_info = {
                'service': 'Crypto Trading Bot',
                'version': '1.0.0',
                'endpoints': {
                    '/health': 'Health check endpoint',
                    '/metrics': 'Prometheus metrics',
                    '/status': 'Detailed status information',
                    '/trading/info': 'Trading-specific information',
                    '/system/info': 'System information'
                },
                'timestamp': time.time()
            }
            
            return web.json_response(root_info)
            
        except Exception as e:
            logger.error(f"Error in root endpoint: {e}")
            return web.json_response({
                'error': str(e)
            }, status=500)
    
    async def start(self):
        """Start the web server."""
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            
            self.site = web.TCPSite(
                self.runner, 
                self.host, 
                self.port
            )
            
            await self.site.start()
            logger.info(f"Web server started on http://{self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Error starting web server: {e}")
            raise
    
    async def stop(self):
        """Stop the web server."""
        try:
            if self.site:
                await self.site.stop()
            if self.runner:
                await self.runner.cleanup()
            logger.info("Web server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping web server: {e}")
    
    def set_trading_bot(self, trading_bot):
        """Set the trading bot instance."""
        self.trading_bot = trading_bot
        logger.info("Trading bot instance set")


# Global web server instance
web_server = None


async def start_web_server(host: str = '0.0.0.0', port: int = 8080, trading_bot=None):
    """Start the web server."""
    global web_server
    
    try:
        web_server = TradingBotWebServer(host, port, trading_bot)
        await web_server.start()
        return web_server
        
    except Exception as e:
        logger.error(f"Failed to start web server: {e}")
        raise


async def stop_web_server():
    """Stop the web server."""
    global web_server
    
    if web_server:
        await web_server.stop()
        web_server = None 