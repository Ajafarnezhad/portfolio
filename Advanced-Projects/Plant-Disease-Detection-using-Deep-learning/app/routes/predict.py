from flask import Blueprint, render_template, request, flash
from werkzeug.utils import secure_filename
import os
import logging
from ..models.plant_disease_model import predict_image

predict_bp = Blueprint('predict', __name__)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in predict_bp.app.config['ALLOWED_EXTENSIONS']

@predict_bp.route('/', methods=['GET', 'POST'])
def index():
    """Render the homepage."""
    return render_template('index.html')

@predict_bp.route('/analyze', methods=['POST'])
def analyze():
    """Handle image upload and prediction."""
    if 'imagefile' not in request.files:
        flash('No file part in the request', 'error')
        return render_template('result.html', result="No file selected. Please try again.")

    file = request.files['imagefile']
    if file.filename == '':
        flash('No file selected', 'error')
        return render_template('result.html', result="No file selected. Please try again.")

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(predict_bp.app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(predict_bp.app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(file_path)

            result = predict_image(file_path, predict_bp.app.model)
            return render_template('result.html', result=result)
        except Exception as e:
            logger.error("Error processing image: %s", str(e))
            flash('Error processing image. Please try again.', 'error')
            return render_template('result.html', result="Error processing image. Please try again.")
    
    flash('Invalid file format. Please upload a PNG, JPG, or JPEG image.', 'error')
    return render_template('result.html', result="Invalid file format. Please try again.")