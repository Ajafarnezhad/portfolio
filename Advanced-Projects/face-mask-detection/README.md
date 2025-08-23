# Face Mask Detection

![Face Mask Detection](https://via.placeholder.com/800x200.png?text=Face+Mask+Detection)  
A professional web application to detect whether a person is wearing a face mask using a CNN model.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Overview
Face Mask Detection is a full-stack web application that uses a convolutional neural network (CNN) to classify face images as "Mask" or "No Mask." The frontend is built with React and Tailwind CSS, featuring a responsive, user-friendly interface with drag-and-drop image upload. The backend is powered by Flask, providing a robust API for predictions.

## Features
- **Accurate Predictions**: Uses a CNN model trained on a dataset of face images with and without masks.
- **Modern UI**: Responsive design with Tailwind CSS and drag-and-drop functionality.
- **Robust Backend**: Flask API with modular routes, error handling, and logging.
- **Secure File Uploads**: Validates file types and sizes for security.
- **Scalable Architecture**: Modular codebase following industry standards.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/face-mask-detection.git
   cd face-mask-detection
   ```

2. **Set Up a Virtual Environment** (Backend):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Backend Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Frontend Dependencies**:
   ```bash
   cd client
   npm install
   ```

5. **Set Up Environment Variables**:
   Create a `.env` file in the project root:
   ```bash
   SECRET_KEY=your-secret-key
   ```

6. **Download the Model**:
   Place the pre-trained `face_mask_cnn.h5` file in the `models/` directory.

7. **Run the Backend**:
   ```bash
   python run.py
   ```

8. **Run the Frontend**:
   In a new terminal, from the `client/` directory:
   ```bash
   npm start
   ```
   Open `http://localhost:3000` in your browser.

## Usage
1. Navigate to the homepage (`http://localhost:3000`).
2. Drag and drop or select an image (PNG, JPG, or JPEG).
3. The application will display whether the face in the image is classified as "Mask" or "No Mask."

## Project Structure
```
face-mask-detection/
├── app/
│   ├── __init__.py
│   ├── routes/
│   │   ├── predict.py
│   ├── models/
│   │   ├── face_mask_model.py
│   ├── config.py
├── client/
│   ├── public/
│   │   ├── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dropzone.js
│   │   │   ├── Result.js
│   │   ├── App.js
│   │   ├── index.js
│   │   ├── styles/
│   │   │   └── tailwind.css
│   ├── package.json
├── models/
│   └── face_mask_cnn.h5
├── notebooks/
│   └── Face-Predictions(With_and_Without_Mask).ipynb
├── requirements.txt
├── run.py
├── README.md
```

## Model Training
The CNN model was trained using TensorFlow/Keras on a dataset of face images with and without masks. The training process is documented in `notebooks/Face-Predictions(With_and_Without_Mask).ipynb`. To retrain the model:
1. Prepare your dataset in the `dataset.face-detection/train_set` and `dataset.face-detection/test_set` directories.
2. Run the Jupyter notebook to train and save the model.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.