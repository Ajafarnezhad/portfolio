# config.py - Application configuration
# Follows PEP 8: Uppercase constants, descriptive names

import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "super_secret_key"
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY") or "jwt_secret"
    LOG_LEVEL = logging.INFO
    DEBUG = False
    # OWASP-aligned: Use secure defaults
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour