from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import logging
from ..models.face_mask_model import predict_image

predict_bp = Blueprint('predict', __name__)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in predict_bp.app.config['ALLOWED_EXTENSIONS']

@predict_bp.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction."""
    if 'image' not in request.files:
        logger.warning("No file part in request")
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        logger.warning("No file selected")
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(predict_bp.app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(predict_bp.app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(file_path)

            result = predict_image(file_path, predict_bp.app.model)
            return jsonify({'prediction': result})
        except Exception as e:
            logger.error("Error processing image: %s", str(e))
            return jsonify({'error': 'Error processing image'}), 500
    
    logger.warning("Invalid file format: %s", file.filename)
    return jsonify({'error': 'Invalid file format. Please upload a PNG, JPG, or JPEG image.'}), 400