import logging 
import logging.config 
import yaml 
import os

def setup_logger(config_path):
    """Set up and return a logger based on a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Logging configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
        # Ensure log directory exists for file handlers
        for handler in config.get('handlers', {}).values():
            if handler.get('class') == 'logging.FileHandler':
                log_dir = os.path.dirname(handler.get('filename', ''))
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
        
        logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    return logger