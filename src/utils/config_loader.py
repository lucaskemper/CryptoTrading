import os
import yaml
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from .logger import logger


class ConfigLoader:
    """Configuration loader for environment variables and config files."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load secrets from config/secrets.env
        secrets_path = "config/secrets.env"
        if os.path.exists(secrets_path):
            load_dotenv(secrets_path)
            logger.info(f"Loaded secrets from {secrets_path}")
        else:
            logger.warning(f"Secrets file {secrets_path} not found")
        
        # Also try to load .env in root directory for backward compatibility
        if os.path.exists(".env"):
            load_dotenv(".env")
            logger.info("Loaded .env file from root directory")
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    config = yaml.safe_load(file)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config or {}
            else:
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                return {}
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value, checking environment variables first."""
        # Check environment variable first
        env_value = os.getenv(key)
        if env_value is not None:
            return env_value
        
        # Check config file
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def get_exchange_config(self, exchange: str) -> Dict[str, str]:
        """Get exchange-specific configuration."""
        api_key = self.get(f"{exchange.upper()}_API_KEY")
        api_secret = self.get(f"{exchange.upper()}_API_SECRET")
        sandbox = self.get(f"{exchange.upper()}_SANDBOX", "false").lower() == "true"
        
        return {
            "apiKey": api_key,
            "secret": api_secret,
            "sandbox": sandbox
        }
    
    def get_sentiment_config(self) -> Dict[str, str]:
        """Get sentiment API configuration."""
        return {
            "news_api_key": self.get("NEWS_API_KEY"),
            "reddit_client_id": self.get("REDDIT_CLIENT_ID"),
            "reddit_client_secret": self.get("REDDIT_CLIENT_SECRET"),
            "reddit_user_agent": self.get("REDDIT_USER_AGENT", "CryptoBot/1.0"),
            "cryptopanic_api_key": self.get("CRYPTOPANIC_API_KEY")
        }
    
    def get_database_config(self) -> Dict[str, str]:
        """Get database configuration."""
        return {
            "sqlite_path": self.get("SQLITE_PATH", "data/trading_bot.db"),
            "postgres_url": self.get("POSTGRES_URL"),
            "csv_dir": self.get("CSV_DIR", "data/")
        }


# Global config instance
config = ConfigLoader()
