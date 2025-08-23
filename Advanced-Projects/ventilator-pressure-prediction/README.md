# Ventilator Pressure Prediction

![Ventilator Pressure Prediction](https://via.placeholder.com/800x200.png?text=Ventilator+Pressure+Prediction)  
A professional web application to predict ventilator pressure using an LSTM model on time-series data.

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
Ventilator Pressure Prediction is a full-stack web application that uses a Long Short-Term Memory (LSTM) model to predict ventilator pressure from time-series data. The frontend is built with React and Tailwind CSS, featuring a responsive interface with CSV file upload. The backend is powered by Flask, providing a robust API for predictions.

## Features
- **Accurate Predictions**: Uses an LSTM model trained on ventilator data to predict pressure.
- **Modern UI**: Responsive design with Tailwind CSS and drag-and-drop CSV upload.
- **Robust Backend**: Flask API with modular routes, error handling, and logging.
- **Secure File Uploads**: Validates file types and sizes for security.
- **Scalable Architecture**: Modular codebase following industry standards.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/ventilator-pressure-prediction.git
   cd ventilator-pressure-prediction
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
   Place the pre-trained `ventilator_lstm.h5` file in the `models/` directory.

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
2. Drag and drop or select a CSV file containing ventilator data (`R`, `C`, `time_step`, `u_in`, `u_out`).
3. View the predicted pressures in a table format.

## Project Structure
```
ventilator-pressure-prediction/
├── app/
│   ├── __init__.py
│   ├── routes/
│   │   ├── predict.py
│   ├── models/
│   │   ├── ventilator_model.py
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
│   └── ventilator_lstm.h5
├── notebooks/
│   └── Google_Brain.ipynb
├── data/
│   ├── train.csv
│   ├── test.csv
├── requirements.txt
├── run.py
├── README.md
```

## Model Training
The LSTM model was trained using TensorFlow/Keras on the dataset provided in `data/train.csv`. The training process is documented in `notebooks/Google_Brain.ipynb`. To retrain the model:
1. Ensure `train.csv` and `test.csv` are in the `data/` directory.
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