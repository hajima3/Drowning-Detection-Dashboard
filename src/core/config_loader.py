"""
Configuration Loader
Loads settings from config.yaml and environment variables
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Central configuration manager"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with environment variables if present
        self._apply_env_overrides(config)
        return config
    
    def _apply_env_overrides(self, config: Dict[str, Any]):
        """Override config values with environment variables"""
        # Model configuration
        if os.getenv('PRIMARY_MODEL_PATH'):
            config['MODEL']['PRIMARY_MODEL'] = os.getenv('PRIMARY_MODEL_PATH')
        
        # Notification configuration
        if os.getenv('SMS_ENABLED'):
            config['NOTIFICATIONS']['SMS_ENABLED'] = os.getenv('SMS_ENABLED').lower() == 'true'
        if os.getenv('SMS_RECIPIENT_NUMBER'):
            config['NOTIFICATIONS']['SMS_RECIPIENT']['NUMBER'] = os.getenv('SMS_RECIPIENT_NUMBER')
        if os.getenv('EMERGENCY_CALL_NUMBER'):
            config['NOTIFICATIONS']['EMERGENCY_CALL']['NUMBER'] = os.getenv('EMERGENCY_CALL_NUMBER')
        
        # Twilio
        if os.getenv('TWILIO_ACCOUNT_SID'):
            config['NOTIFICATIONS']['SMS_PROVIDER']['API_KEY'] = os.getenv('TWILIO_ACCOUNT_SID')
        if os.getenv('TWILIO_AUTH_TOKEN'):
            config['NOTIFICATIONS']['SMS_PROVIDER']['API_SECRET'] = os.getenv('TWILIO_AUTH_TOKEN')
        if os.getenv('TWILIO_PHONE_NUMBER'):
            config['NOTIFICATIONS']['SMS_PROVIDER']['SENDER_ID'] = os.getenv('TWILIO_PHONE_NUMBER')
        
        # Roboflow
        if os.getenv('ROBOFLOW_API_KEY'):
            config['DATASETS']['ROBOFLOW']['API_KEY'] = os.getenv('ROBOFLOW_API_KEY')
        
        # Flask
        if os.getenv('FLASK_HOST'):
            config['SERVER']['HOST'] = os.getenv('FLASK_HOST')
        if os.getenv('FLASK_PORT'):
            config['SERVER']['PORT'] = int(os.getenv('FLASK_PORT'))
        if os.getenv('FLASK_DEBUG'):
            config['SERVER']['DEBUG'] = os.getenv('FLASK_DEBUG').lower() == 'true'
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation
        Example: config.get('MODEL.PRIMARY_MODEL')
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self._config.get('MODEL', {})
    
    def get_alert_config(self) -> Dict[str, Any]:
        """Get alert configuration"""
        return self._config.get('ALERTS', {})
    
    def get_notification_config(self) -> Dict[str, Any]:
        """Get notification configuration"""
        return self._config.get('NOTIFICATIONS', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self._config.get('PERFORMANCE', {})
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration"""
        return self._config.get('DATASETS', {})
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration"""
        return self._config.get('SERVER', {})
    
    @property
    def all(self) -> Dict[str, Any]:
        """Get full configuration"""
        return self._config


# Singleton instance
_config_instance = None

def get_config(config_path: str = "config/config.yaml") -> Config:
    """Get or create configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
