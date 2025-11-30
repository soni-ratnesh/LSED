"""LSED Configuration Module."""

import yaml
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: dict, config_path: str):
    """Save configuration to YAML file."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


__all__ = ["load_config", "save_config", "DEFAULT_CONFIG_PATH"]
