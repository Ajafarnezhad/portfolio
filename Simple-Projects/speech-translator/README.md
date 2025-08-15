# Speech Recognizer: Transform Your Voice into Text 🎙️✨

Welcome to the **Speech Recognizer**, a sleek and powerful Python project that converts spoken input into text using the Google Speech Recognition API. Supporting Persian and other languages, this project features a robust CLI interface, JSON-based history tracking, and multi-language support, making it an outstanding portfolio piece for showcasing your skills in speech processing, error handling, and modular design.

---

## 🌟 Project Highlights
This project delivers real-time speech-to-text conversion with a focus on usability and flexibility. With features like ambient noise adjustment, detailed logging, and a clean code structure, it’s perfect for demonstrating expertise in audio processing and interactive application development.

---

## 🚀 Features
- **Speech Recognition**: Converts microphone audio to text in Persian or other languages via the Google Speech Recognition API.
- **History Tracking**: Saves recognized text with timestamps to a JSON file and displays it in a formatted table using `pandas`.
- **Multi-Language Support**: Easily configure and list supported languages for flexible recognition.
- **CLI Interface**: Customize language, timeouts, and thresholds through intuitive command-line arguments.
- **Error Handling**: Robustly manages microphone issues, API errors, and invalid inputs with clear user feedback.
- **Logging**: Detailed logs for debugging and monitoring recognition performance.
- **Ambient Noise Adjustment**: Automatically calibrates for background noise to enhance recognition accuracy.

---

## 🛠️ Requirements
- **Python**: 3.8 or higher
- **Libraries**:
  - `speechrecognition`
  - `pyaudio`
  - `pandas`

Install dependencies with:
```bash
pip install speechrecognition pyaudio pandas
```

**Note**: `pyaudio` may require system dependencies like PortAudio. Install it with:
- **macOS**: `brew install portaudio`
- **Ubuntu**: `apt-get install portaudio19-dev`
- **Windows**: Typically works out of the box, but refer to PortAudio documentation if needed.

---

## 🎮 How to Run

### 1. Install Dependencies
Ensure Python 3.8+ is installed and run:
```bash
pip install speechrecognition pyaudio pandas
```

### 2. Run the Script
Launch the Speech Recognizer with default settings (Persian, 5s timeout, 10s phrase limit):
```bash
python speech_recognizer.py
```

### 3. Customize Your Workflow
Tailor the recognition process with CLI arguments:
```bash
python speech_recognizer.py --language fa-IR --timeout 7 --phrase_limit 15
```
- `--language`: Set the recognition language (e.g., `fa-IR` for Persian, `en-US` for English).
- `--timeout`: Microphone listening timeout in seconds (default: `5`).
- `--phrase_limit`: Maximum phrase duration in seconds (default: `10`).

### 4. Interact
- Speak into your microphone when prompted.
- View the recognized text and check the JSON history file for past transcriptions.
- Press `Ctrl+C` to exit.

---

## 📈 Example Output
- **Console**:
  ```
  INFO: Adjusting for ambient noise...
  INFO: Listening for speech (language: fa-IR)...
  Recognized: سلام، امروز روز خوبی است
  INFO: Transcription saved to history.json
  ```
- **History Table** (displayed via `pandas`):
  ```
  Timestamp                 Text
  2025-08-16 03:32:00      سلام، امروز روز خوبی است
  ```
- **JSON Output** (in `history.json`):
  ```json
  [
    {"timestamp": "2025-08-16 03:32:00", "text": "سلام، امروز روز خوبی است"}
  ]
  ```

---

## 🔮 Future Enhancements
Elevate this project with these exciting ideas:
- **Offline Recognition**: Integrate local models like Whisper for offline speech recognition.
- **Web App Integration**: Build a Flask or Streamlit app for a browser-based interface.
- **Real-Time Translation**: Add real-time text translation for multilingual outputs.
- **Visualization**: Include audio waveform or spectrogram plots for recognized speech.
- **Unit Testing**: Implement `pytest` for robust validation of recognition and history tracking.

---

## 📜 License
This project is licensed under the **MIT License**—use, modify, and share it freely!

Turn your voice into text with the **Speech Recognizer** and showcase your audio processing prowess! 🚀