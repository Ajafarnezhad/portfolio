# Speech-to-Text-to-Speech Translator: Break Language Barriers with Voice 🌐🎙️

Welcome to the **Speech-to-Text-to-Speech Translator**, a dynamic Python project that transforms spoken input in Persian (or other languages) into translated text and spoken output in real-time. Combining speech recognition, translation, and text-to-speech capabilities, this project features a professional CLI interface, history tracking, and cross-platform support. It’s an outstanding portfolio piece to showcase your skills in speech processing, API integration, and interactive system design.

---

## 🌟 Project Highlights
This project seamlessly integrates speech recognition, translation, and text-to-speech into a cohesive voice translation system. With robust error handling, detailed logging, and a modular design, it’s perfect for demonstrating expertise in audio processing, multilingual applications, and user-friendly interfaces.

---

## 🚀 Features
- **Speech Recognition**: Converts spoken input to text using the Google Speech Recognition API.
- **Text Translation**: Translates text into multiple languages via the `deep-translator` library (Google Translate API).
- **Text-to-Speech**: Converts translated text to spoken output using `pyttsx3` with customizable voice and rate settings.
- **History Tracking**: Saves translations with timestamps to a JSON file and displays them in a formatted table using `pandas`.
- **Multi-Language Support**: Lists supported languages for speech recognition and translation for easy configuration.
- **CLI Interface**: Offers flexible command-line arguments to configure language, timeouts, and text-to-speech options.
- **Error Handling**: Robustly manages microphone issues, API errors, and file operations with clear feedback.
- **Logging**: Detailed logs for debugging and monitoring system performance.
- **Cross-Platform**: Compatible with Windows, macOS, and Linux.

---

## 🛠️ Requirements
- **Python**: 3.8 or higher
- **Libraries**:
  - `speechrecognition`
  - `pyaudio`
  - `deep-translator`
  - `pyttsx3`
  - `pandas`

Install dependencies with:
```bash
pip install speechrecognition pyaudio deep-translator pyttsx3 pandas
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
pip install speechrecognition pyaudio deep-translator pyttsx3 pandas
```

### 2. Run the Script
Launch the translator with default settings (Persian input, English output, 5s timeout):
```bash
python speech_to_text_to_speech.py
```

### 3. Customize Your Workflow
Tailor the translation process with CLI arguments:
```bash
python speech_to_text_to_speech.py --input_language fa-IR --output_language en-US --timeout 7 --speak
```
- `--input_language`: Speech recognition language (e.g., `fa-IR` for Persian, `en-US` for English).
- `--output_language`: Translation target language (e.g., `en-US`, `fr-FR`).
- `--timeout`: Microphone listening timeout in seconds (default: `5`).
- `--speak`: Enable text-to-speech for the translated text.

### 4. Interact
- Speak into your microphone when prompted.
- View the recognized text, its translation, and hear the translated output (if `--speak` is enabled).
- Check the JSON history file for past translations.
- Press `Ctrl+C` to exit.

---

## 📈 Example Output
- **Console**:
  ```
  INFO: Adjusting for ambient noise...
  INFO: Listening for speech (language: fa-IR)...
  Recognized: سلام، چطور می‌توانم به شما کمک کنم؟
  Translated (to en-US): Hello, how can I help you?
  INFO: Speaking translation...
  INFO: Translation saved to history.json
  ```
- **History Table** (displayed via `pandas`):
  ```
  Timestamp                 Original Text                     Translated Text
  2025-08-16 03:32:00      سلام، چطور می‌توانم به شما کمک کنم؟    Hello, how can I help you?
  ```
- **JSON Output** (in `history.json`):
  ```json
  [
    {
      "timestamp": "2025-08-16 03:32:00",
      "original_text": "سلام، چطور می‌توانم به شما کمک کنم؟",
      "translated_text": "Hello, how can I help you?",
      "input_language": "fa-IR",
      "output_language": "en-US"
    }
  ]
  ```

---

## 🔮 Future Enhancements
Take this project to the next level with these exciting ideas:
- **Offline Speech Recognition**: Integrate local models like Whisper for offline capabilities.
- **Real-Time Streaming**: Enable continuous speech recognition and translation for live conversations.
- **Web App Integration**: Build a Flask or Streamlit app for a browser-based interface.
- **Voice Customization**: Add options for different voices or accents in `pyttsx3`.
- **Unit Testing**: Implement `pytest` for robust validation of recognition, translation, and speech synthesis.

---

## 📜 License
This project is licensed under the **MIT License**—use, modify, and share it freely!

Break language barriers with the **Speech-to-Text-to-Speech Translator** and showcase your skills in multilingual voice applications! 🚀