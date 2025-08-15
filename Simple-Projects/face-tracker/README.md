# Face Tracker: Control Your Cursor with Your Face 🚀👀

Welcome to the **Face Tracker**, a simple yet powerful Python project that uses OpenCV to detect faces and eyes in real-time via your webcam, seamlessly translating face movements into mouse cursor control. With configurable boundaries, smooth cursor motion, and a user-friendly CLI interface, this project is a standout portfolio piece for showcasing your skills in computer vision, hardware interaction, and modular programming.

---

## 🌟 Project Highlights
This project combines real-time face detection with intuitive mouse control, offering a practical demonstration of computer vision and cross-platform automation. Featuring robust error handling, detailed logging, and a clean code structure, it’s perfect for highlighting your expertise in Python and interactive system design.

---

## 🚀 Features
- **Face and Eye Detection**: Leverages Haar Cascade classifiers for accurate and efficient face and eye detection.
- **Mouse Control**: Maps face position to cursor movement within user-defined screen boundaries.
- **Smoothing**: Applies exponential smoothing to ensure fluid, jitter-free cursor motion.
- **CLI Interface**: Customize detection parameters, movement boundaries, and video feed display via command-line arguments.
- **Error Handling**: Robust checks for camera availability and graceful handling of detection failures.
- **Logging**: Detailed logs for debugging and performance monitoring.
- **Cross-Platform**: Compatible with Windows, macOS, and Linux using `pyautogui`.

---

## 🛠️ Requirements
- **Python**: 3.8 or higher
- **Libraries**:
  - `opencv-python`
  - `pyautogui`

Install dependencies with:
```bash
pip install opencv-python pyautogui
```

---

## 🎮 How to Run

### 1. Install Dependencies
Ensure Python 3.8+ is installed and run:
```bash
pip install opencv-python pyautogui
```

### 2. Run the Script
Launch the Face Tracker with default settings:
```bash
python face_tracker.py
```

### 3. Customize Your Workflow
Use optional CLI arguments to tailor the experience:
```bash
python face_tracker.py --sensitivity 0.5 --disable_video --boundary_x 0.2 --boundary_y 0.2
```
- `--sensitivity`: Adjust mouse movement sensitivity (default: `0.5`).
- `--disable_video`: Disable the webcam video feed display.
- `--boundary_x`: Set horizontal boundary for face movement (default: `0.2`).
- `--boundary_y`: Set vertical boundary for face movement (default: `0.2`).

### 4. Interact
- Face the webcam to start tracking.
- Move your face to control the mouse cursor.
- Press `q` to exit the application.

---

## 📈 Example Output
- **Logs**:
  ```
  INFO: Initializing webcam...
  INFO: Face detected at (x: 320, y: 240)
  INFO: Moving cursor to (screen_x: 960, screen_y: 540)
  INFO: Application terminated successfully
  ```
- **Real-Time Interaction**: The cursor moves smoothly based on your face position, with the webcam feed displayed (unless disabled).

---

## 🔮 Future Enhancements
Take this project to the next level with these exciting ideas:
- **Advanced Detection**: Integrate deep learning-based face detection (e.g., MTCNN) for improved accuracy.
- **Gesture Control**: Add blink or head tilt detection for mouse clicks or scrolling.
- **Web App Integration**: Stream webcam feed to a browser-based interface using Flask or Streamlit.
- **Calibration Mode**: Implement a user-friendly calibration step for personalized sensitivity.
- **Unit Testing**: Add `pytest` for robust validation of detection and mouse control logic.

---

## 📜 License
This project is licensed under the **MIT License**—use, modify, and share it freely!

Transform your webcam into a powerful controller with the **Face Tracker** and showcase your computer vision skills! 🚀