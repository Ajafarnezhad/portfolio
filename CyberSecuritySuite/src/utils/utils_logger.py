# utils/logger.py - Structured logging module
# PEP 8: Imports at top, blank lines between sections

import logging
import json
from datetime import datetime

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler with JSON formatting for ELK compatibility
    handler = logging.FileHandler(
        f"logs/{name}_{datetime.now().strftime('%Y%m%d')}.log"
    )
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger