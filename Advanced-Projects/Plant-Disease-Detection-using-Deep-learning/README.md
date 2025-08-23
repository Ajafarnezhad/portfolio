# Plant Disease Prediction

![Plant Disease Prediction](https://via.placeholder.com/800x200.png?text=Plant+Disease+Prediction)  
A professional web application to predict plant diseases from images using a pre-trained InceptionV3 model.

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
Plant Disease Prediction is a Flask-based web application that uses a deep learning model (InceptionV3) to classify plant images into three categories: Healthy, Powdery, or Rust. The application features a modern, responsive UI built with Tailwind CSS and supports drag-and-drop image uploads.

## Features
- **Accurate Predictions**: Utilizes a pre-trained InceptionV3 model for high-accuracy plant disease classification.
- **Modern UI**: Responsive design with Tailwind CSS and drag-and-drop functionality.
- **Robust Backend**: Flask application with modular routes, error handling, and logging.
- **Secure File Uploads**: Validates file types and sizes to ensure security.
- **Scalable Architecture**: Organized codebase following industry-standard practices.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/plant-disease-prediction.git
   cd plant-disease-prediction
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   Create a `.env` file in the project root:
   ```bash
   SECRET_KEY=your-secret-key
   ```

5. **Download the Model**:
   Place the pre-trained `model_inception.h5` file in the `models/` directory.

6. **Run the Application**:
   ```bash
   python run.py
   ```
   Open `http://localhost:5000` in your browser.

## Usage
1. Navigate to the homepage.
2. Drag and drop or select an image (PNG, JPG, or JPEG).
3. Click "Analyze Image" to get the prediction result.
4. View the result, which indicates whether the plant is Healthy, Powdery, or Rust.

## Project Structure
```
plant-disease-prediction/
├── app/
│   ├── __init__.py
│   ├── routes/
│   │   ├── predict.py
│   ├── models/
│   │   ├── plant_disease_model.py
│   ├── static/
│   │   ├── css/
│   │   │   └── tailwind.css
│   │   ├── js/
│   │   │   └── main.js
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── result.html
│   ├── uploads/
│   ├── config.py
├── models/
│   └── model_inception.h5
├── notebooks/
│   └── Plant_Disease_Prediction.ipynb
├── requirements.txt
├── README.md
├── run.py
```

## Model Training
The model was trained using the InceptionV3 architecture on a dataset of plant images. The training process is documented in `notebooks/Plant_Disease_Prediction.ipynb`. To retrain the model:
1. Prepare your dataset in the `Dataset2/train` and `Dataset2/test` directories.
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