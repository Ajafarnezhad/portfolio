# app.py - Main Flask application entry point
# Compliant with PEP 8: 4-space indentation, 79-char lines, etc.

import logging
from flask import Flask, jsonify
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from blueprints.auth import auth_bp
from blueprints.scanner import scanner_bp
from blueprints.chat import chat_bp
from utils.logger import setup_logger
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Setup JWT
jwt = JWTManager(app)

# Rate limiting for security
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Logger setup
logger = setup_logger("cybersecurity_suite")
logger.info("Application starting...")

# Register blueprints
app.register_blueprint(auth_bp, url_prefix="/auth")
app.register_blueprint(scanner_bp, url_prefix="/scan")
app.register_blueprint(chat_bp, url_prefix="/chat")

@app.route("/")
def index():
    return jsonify({"message": "Welcome to CyberSecuritySuite"})

@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 error: {error}")
    return jsonify({"error": "Not found"}), 404

if __name__ == "__main__":
    app.run(debug=Config.DEBUG, host="0.0.0.0", port=5000)