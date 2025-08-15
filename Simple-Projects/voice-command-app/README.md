# Voice Command App: Control Your Device with Your Voice 🎙️✨

Welcome to the **Voice Command App**, a simple yet powerful Python project that transforms spoken commands in Persian (or other languages) into actions, such as launching Google Chrome. Powered by the Google Speech Recognition API, this project features a robust CLI interface, cross-platform support, and professional error handling, making it an excellent portfolio piece to showcase your skills in speech processing, automation, and modular design.

---

## 🌟 Project Highlights
This project combines real-time speech recognition with actionable command execution, offering a practical demonstration of voice-controlled automation. With ambient noise adjustment, detailed logging, and a clean code structure, it’s perfect for highlighting expertise in audio processing and cross-platform application development.

---

## 🚀 Features
- **Speech Recognition**: Converts spoken commands to text using the Google Speech Recognition API.
- **Command Processing**: Executes actions (e.g., opening Google Chrome) based on detected keywords in the recognized text.
- **Cross-Platform Support**: Compatible with Windows, macOS, and Linux for seamless operation.
- **CLI Interface**: Customize language, timeout, and phrase duration via intuitive command-line arguments.
- **Error Handling**: Robustly manages microphone issues, API errors, and invalid inputs with clear feedback.
- **Logging**: Detailed logs for debugging and monitoring command execution.
- **Ambient Noise Adjustment**: Automatically calibrates for background noise to enhance recognition accuracy.

---

## 🛠️ Requirements
- **Python**: 3.8 or higher
- **Libraries**:
  - `speechrecognition`
  - `pyaudio`

Install dependencies with:
```bash
pip install speechrecognition pyaudio
```

**Note**: `pyaudio` requires system dependencies like PortAudio. Install it with:
- **macOS**: `brew install portaudio`
- **Ubuntu**: `apt-get install portaudio19-dev`
- **Windows**: Typically works out of the box, but refer to PortAudio documentation if needed.

---

## 🎮 How to Run

### 1. Install Dependencies
Ensure Python 3.8+ is installed and run:
```bash
pip install speechrecognition pyaudio
```

### 2. Run the Script
Launch the Voice Command App with default settings (Persian, 5s timeout, 10s phrase limit):
```bash
python voice_command_app.py
```

### 3. Customize Your Workflow
Tailor the command recognition process with CLI arguments:
```bash
python voice_command_app.py --language fa-IR --timeout 7 --phrase_limit 15
```
- `--language`: Set the recognition language (e.g., `fa-IR` for Persian, `en-US` for English).
- `--timeout`: Microphone listening timeout in seconds (default: `5`).
- `--phrase_limit`: Maximum phrase duration in seconds (default: `10`).

### 4. Interact
- Speak a command (e.g., "باز کردن کروم" in Persian or "open Chrome" in English).
- The app detects keywords and executes the corresponding action (e.g., launching Google Chrome).
- Press `Ctrl+C` to exit.

---

## 📈 Example Output
- **Console**:
  ```
  INFO: Adjusting for ambient noise...
  INFO: Listening for command (language: fa-IR)...
  Recognized: باز کردن کروم
  INFO: Executing action: Opening Google Chrome
  INFO: Action completed successfully
  ```
- **Action**: Google Chrome launches if the command includes keywords like "open Chrome" or "باز کردن کروم".

---

## 🔮 Future Enhancements
Take this project to the next level with these exciting ideas:
- **Expanded Commands**: Add support for more actions (e.g., opening other apps, controlling system settings).
- **Offline Recognition**: Integrate local models like Whisper for offline speech processing.
- **Web App Integration**: Build a Flask or Streamlit app for a browser-based interface.
- **Custom Keyword Mapping**: Allow users to define custom voice commands and actions.
- **Unit Testing**: Implement `pytest` for robust validation of command recognition and execution.

---

## 📜 License
This project is licensed under the **MIT License**—use, modify, and share it freely!

Control your device with the power of your voice using the **Voice Command App** and showcase your automation skills! 🚀