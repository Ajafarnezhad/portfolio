from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import logging
from ..models.ventilator_model import predict_pressure

predict_bp = Blueprint('predict', __name__)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in predict_bp.app.config['ALLOWED_EXTENSIONS']

@predict_bp.route('/predict', methods=['POST'])
def predict():
    """Handle CSV upload and return predictions."""
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning("No file selected")
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(predict_bp.app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(predict_bp.app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(file_path)

            predictions = predict_pressure(file_path, predict_bp.app.model)
            return jsonify({'predictions': predictions})
        except Exception as e:
            logger.error("Error processing file: %s", str(e))
            return jsonify({'error': 'Error processing file'}), 500
    
    logger.warning("Invalid file format: %s", file.filename)
    return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400