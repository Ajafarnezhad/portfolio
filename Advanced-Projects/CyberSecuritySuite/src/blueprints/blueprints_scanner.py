# blueprints/scanner.py - Scanning endpoints
# Includes OWASP Top 10 checks (e.g., vulnerable deps, injection simulation)

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
import socket
import subprocess  # For integrating Bandit/Safety (assume installed)
from utils.validators import sanitize_input, validate_ip
from utils.report_generator import generate_pdf_report
from utils.logger import setup_logger

scanner_bp = Blueprint("scanner", __name__)
logger = setup_logger("scanner")

@scanner_bp.route("/port", methods=["POST"])
@jwt_required()
def scan_ports():
    data = request.json
    target = sanitize_input(data.get("target"))
    ports = data.get("ports", "1-1000").split("-")
    
    if not validate_ip(target):
        return jsonify({"error": "Invalid target"}), 400
    
    results = []
    start, end = int(ports[0]), int(ports[1])
    for port in range(start, end + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((target, port))
        if result == 0:
            results.append(f"Port {port} is open")
        sock.close()
    
    report = generate_pdf_report(results, target, "port_scan")
    return jsonify({"results": results, "report": report})

@scanner_bp.route("/dependencies", methods=["POST"])
@jwt_required()
def scan_dependencies():
    # Integrate Safety for vulnerable deps (OWASP A06)
    try:
        result = subprocess.run(["safety", "check", "--json"], capture_output=True, text=True)
        vulnerabilities = json.loads(result.stdout)
        return jsonify({"vulnerabilities": vulnerabilities})
    except Exception as e:
        logger.error(f"Dependency scan error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@scanner_bp.route("/static", methods=["POST"])
@jwt_required()
def static_analysis():
    # Integrate Bandit for code issues (OWASP A03 Injection, etc.)
    file_path = sanitize_input(request.json.get("file_path"))
    try:
        result = subprocess.run(["bandit", "-r", file_path, "-f", "json"], capture_output=True, text=True)
        issues = json.loads(result.stdout)
        return jsonify({"issues": issues})
    except Exception as e:
        logger.error(f"Static analysis error: {str(e)}")
        return jsonify({"error": str(e)}), 500