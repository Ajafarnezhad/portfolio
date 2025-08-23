# 🏯 AI-Powered Landmark Recognition: Identifying Asian Landmarks with Geolocation

## 🌟 Project Vision
Discover the wonders of Asia through the **AI-Powered Landmark Recognition** project, a sophisticated Python application that identifies landmarks in images using a TensorFlow Hub model and retrieves their geolocation data. Built with modularity and scalability in mind, this system processes images, predicts landmarks with high accuracy, and provides detailed geolocation information via the Nominatim API. With robust error handling, a flexible CLI, and JSON output storage, this project is a premier showcase of computer vision and geospatial analysis, ideal for data science portfolios aiming for international standards.

## ✨ Core Features
- **Landmark Classification** 🧠: Utilizes a pre-trained TensorFlow Hub model to identify Asian landmarks with high accuracy.
- **Geolocation Integration** 🌍: Retrieves precise address, latitude, and longitude for predicted landmarks using Nominatim.
- **Robust Image Processing** 📸: Supports multiple image formats (JPEG, PNG) with standardized preprocessing.
- **JSON Output Storage** 📝: Saves prediction and geolocation results in a structured JSON format for easy integration.
- **Interactive CLI Interface** 🖥️: Offers intuitive commands for image processing and result generation.
- **Comprehensive Error Handling & Logging** 🛡️: Ensures reliability with detailed logs and graceful error recovery.
- **Scalable Design** ⚙️: Supports custom models, additional image formats, and extensible geolocation services.

## 🛠️ Getting Started

### Prerequisites
- **Python**: Version 3.8 or higher
- **Dependencies**: A curated suite of libraries:
  - `tensorflow`
  - `tensorflow-hub`
  - `pillow`
  - `pandas`
  - `numpy`
  - `geopy`
- **Dataset**: The `landmarks_classifier_asia_V1_label_map.csv` file, containing label mappings for the TensorFlow Hub model (download from [TensorFlow Hub](https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1)).

Install dependencies with:
```bash
pip install tensorflow tensorflow-hub pillow pandas numpy geopy
```

### Model and Labels
- **Model**: Uses the TensorFlow Hub model for Asian landmark classification (`landmarks_classifier_asia_V1`).
- **Labels**: The `landmarks_classifier_asia_V1_label_map.csv` file maps model output IDs to landmark names.
- **Setup**: Place `landmarks_classifier_asia_V1_label_map.csv` in the project root or specify its path via CLI.

## 🎉 How to Use

### 1. Prepare the Environment
Ensure the labels CSV file (`landmarks_classifier_asia_V1_label_map.csv`) is in the project root or provide its path.

### 2. Run Landmark Recognition
Identify a landmark in an image and retrieve its geolocation:
```bash
python main.py --image taj_mahal.jpg
```

### 3. Customize Parameters
Specify custom model URL, labels path, or output directory:
```bash
python main.py --image taj_mahal.jpg --model_url https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1 --labels landmarks_classifier_asia_V1_label_map.csv --output_dir ./results
```

### CLI Options
- `--image`: Path to the input image (required).
- `--model_url`: TensorFlow Hub model URL (default: Asian landmarks model).
- `--labels`: Path to labels CSV file (default: `landmarks_classifier_asia_V1_label_map.csv`).
- `--output_dir`: Directory for saving JSON results (default: `./results`).

## 📊 Sample Output

### Console Output
```
2025-08-23 20:49:12,345 - INFO - Predicted landmark: Taj Mahal (Confidence: 95.67%)
2025-08-23 20:49:12,567 - INFO - Geolocation: Taj Mahal, Tajganj, Agra, Uttar Pradesh, India
2025-08-23 20:49:12,789 - INFO - Latitude: 27.1751, Longitude: 78.0421
2025-08-23 20:49:12,901 - INFO - Results saved to ./results/result_20250823_204912.json
```

### JSON Output (`results/result_20250823_204912.json`)
```json
{
    "image": "taj_mahal.jpg",
    "landmark": "Taj Mahal",
    "confidence": 0.9567,
    "geolocation": {
        "address": "Taj Mahal, Tajganj, Agra, Uttar Pradesh, India",
        "latitude": 27.1751,
        "longitude": 78.0421
    },
    "timestamp": "20250823_204912"
}
```

## 🌈 Future Enhancements
- **Multi-Model Support** 🚀: Integrate additional TensorFlow Hub models for global landmark recognition.
- **Batch Processing** ⚡: Support processing multiple images in a single run.
- **Web Interface** 🌐: Develop a Streamlit or Flask app for interactive image uploads and results visualization.
- **Visualization Dashboard** 📊: Add Recharts-based visualizations for confidence scores and geolocation maps.
- **Unit Testing** 🛠️: Implement `pytest` for robust validation of the recognition pipeline.

## 📜 License
Proudly licensed under the **MIT License**, fostering open collaboration and innovation in computer vision and geospatial analysis.

---

🌟 **AI-Powered Landmark Recognition**: Unveiling the world's landmarks with AI and geolocation! 🌟