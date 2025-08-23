from flask import Flask
from flask_cors import CORS
from .config import Config
from .routes.predict import predict_bp
from .models.face_mask_model import load_model

def create_app():
    """Initialize the Flask application."""
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)  # Enable CORS for React frontend

    # Register blueprints
    app.register_blueprint(predict_bp, url_prefix='/api')

    # Load the model at app startup
    app.model = load_model(app.config['MODEL_PATH'])

    return app