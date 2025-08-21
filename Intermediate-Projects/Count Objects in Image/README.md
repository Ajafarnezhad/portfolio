# 🚗 Count Objects in Image: Computer Vision with Python

## 🌟 Project Vision
Step into the realm of computer vision with the **Count Objects in Image** project, a beginner-friendly yet sophisticated Python-based application that counts specific objects (e.g., cars) in images using the `cvlib` library. By leveraging pre-trained YOLO models, this project detects and counts objects with ease, delivering stunning visualizations and precise results. With a sleek command-line interface (CLI), robust error handling, and vibrant output, it’s a polished showcase of computer vision expertise, crafted to elevate your portfolio to global standards.

## ✨ Core Features
- **Seamless Image Processing** 📸: Loads and validates images with robust checks for compatibility.
- **Object Detection with cvlib** 🧠: Uses pre-trained YOLO models to detect objects like cars in images.
- **Targeted Object Counting** 🔢: Counts specific objects (e.g., cars) with high accuracy.
- **Visual Output** 🖼️: Displays images with bounding boxes around detected objects for intuitive verification.
- **Elegant CLI Interface** 🖥️: Offers intuitive commands for image processing, object counting, and visualization.
- **Robust Error Handling & Logging** 🛡️: Ensures reliability with meticulous checks and detailed logs for transparency.
- **Scalable Design** ⚙️: Supports extensible object detection for various object types and image formats.

## 🛠️ Getting Started

### Prerequisites
- **Python**: Version 3.8 or higher
- **Dependencies**: A curated suite of libraries to power your computer vision tasks:
  - `cvlib`
  - `opencv-python`
  - `numpy`
  - `matplotlib`
  - `tensorflow` (required by `cvlib` for YOLO)

Install them with a single command:
```bash
pip install cvlib opencv-python numpy matplotlib tensorflow
```

### Image Spotlight
The project works with any image containing detectable objects (e.g., vehicles). A sample image with vehicles is recommended for testing:
- **Source**: Use your own image or download a sample from [Pexels](https://www.pexels.com/search/traffic/) or [Unsplash](https://unsplash.com/s/photos/vehicles).
- **Format**: JPEG or PNG.
- **Setup**: Place the image (e.g., `traffic.jpg`) in the project directory or specify its path via the CLI.

## 🎉 How to Use

### 1. Count Objects in an Image
Detect and count cars in an image, with an optional visualization:
```bash
python count_objects.py --mode count --image_path traffic.jpg --object car
```

### 2. Visualize Detected Objects
Generate an image with bounding boxes around detected objects:
```bash
python count_objects.py --mode visualize --image_path traffic.jpg --object car
```

### CLI Options
- `--mode`: Choose `count` (object counting) or `visualize` (image with bounding boxes) (default: `count`).
- `--image_path`: Path to the input image (default: `traffic.jpg`).
- `--object`: Object type to count (e.g., `car`, `person`, `truck`) (default: `car`).
- `--output_dir`: Directory for saving visualizations (default: `./outputs`).

## 📊 Sample Output

### Count Output
```
🌟 Processing image: traffic.jpg
🔍 Detected 12 objects (10 cars, 2 trucks)
✅ Number of cars: 10
```

### Visualization Output
Find the processed image in the `outputs/` folder:
- `traffic_detected.jpg`: Image with bounding boxes and labels around detected cars.

## 🌈 Future Enhancements
- **Multi-Object Counting** 🚀: Extend to count multiple object types simultaneously.
- **Custom Model Integration** 📚: Incorporate custom-trained YOLO models for specialized object detection.
- **Web App Deployment** 🌐: Transform into an interactive dashboard with Streamlit for user-friendly exploration.
- **Video Processing** 🎥: Enable object counting in video streams for real-time applications.
- **Unit Testing** 🛠️: Implement `pytest` for robust validation of detection and counting logic.

## 📜 License
Proudly licensed under the **MIT License**, fostering open collaboration and innovation in computer vision.

---

🌟 **Count Objects in Image**: Where data science brings images to life! 🌟