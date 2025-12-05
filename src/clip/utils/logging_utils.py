"""
Logging utilities for experiments.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """Setup logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_metrics_to_jsonl(metrics: Dict[str, Any], output_file: str):
    """Append metrics to JSONL file."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(metrics) + '\n')


def save_metrics_to_json(metrics: Dict[str, Any], output_file: str):
    """Save metrics to JSON file."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
