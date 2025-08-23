from flask import Flask
from .config import Config
from .routes.predict import predict_bp
from .models.plant_disease_model import load_model

def create_app():
    """Initialize the Flask application."""
    app = Flask(__name__)
    app.config.from_object(Config)

    # Register blueprints
    app.register_blueprint(predict_bp, url_prefix='/')

    # Load the model at app startup
    app.model = load_model(app.config['MODEL_PATH'])

    return app