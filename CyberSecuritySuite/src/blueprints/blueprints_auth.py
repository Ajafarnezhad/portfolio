# blueprints/auth.py - Authentication blueprint
# OWASP A07: Identification and Authentication Failures

from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token
from utils.logger import setup_logger

auth_bp = Blueprint("auth", __name__)
logger = setup_logger("auth")

# Mock user database (replace with real DB in production)
USERS = {"admin": "secure_password"}  # Hashed in production

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    
    if username in USERS and USERS[username] == password:
        access_token = create_access_token(identity=username)
        logger.info(f"User {username} logged in")
        return jsonify({"access_token": access_token})
    else:
        logger.warning(f"Failed login attempt for {username}")
        return jsonify({"error": "Invalid credentials"}), 401