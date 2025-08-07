\# Speech-to-Text-to-Speech Translator



\## Overview

This simple Python project combines speech recognition, text translation, and text-to-speech to create an interactive voice translation system. It captures spoken input in Persian (or other languages), translates it using the Google Translate API, and optionally speaks the translated text. The project features a professional CLI interface, history tracking, and cross-platform support, making it an excellent portfolio piece for showcasing speech processing and API integration.



\## Features

\- \*\*Speech Recognition\*\*: Convert spoken input to text using Google Speech Recognition API.

\- \*\*Translation\*\*: Translate text to multiple languages via `deep-translator`.

\- \*\*Text-to-Speech\*\*: Speak translated text using `pyttsx3` with configurable voice and rate.

\- \*\*History Tracking\*\*: Save translations with timestamps to a JSON file and display them in a table.

\- \*\*Supported Languages\*\*: List available languages for speech recognition and translation.

\- \*\*CLI Interface\*\*: Flexible command-line arguments for configuration and control.

\- \*\*Error Handling\*\*: Robust handling of microphone issues, API errors, and file operations.

\- \*\*Logging\*\*: Detailed logs for debugging and monitoring.

\- \*\*Cross-Platform\*\*: Works on Windows, macOS, and Linux.



\## Requirements

\- Python 3.8+

\- Libraries: `speechrecognition`, `pyaudio`, `deep-translator`, `pyttsx3`, `pandas`



Install dependencies:

```bash

pip install speechrecognition pyaudio deep-translator pyttsx3 pandas







Note: PyAudio requires system dependencies like PortAudio:



macOS: brew install portaudio

Ubuntu: sudo apt-get install portaudio19-dev

Windows: Usually included with PyAudio, but ensure pip install pyaudio succeeds.



How to Run



Run with default settings (Persian input, English output, 5s timeout):

bashpython speech\_translator.py --speak



Specify options (e.g., English input, Spanish output, custom timeouts):

bashpython speech\_translator.py --language en-US --target\_lang es --timeout 3 --phrase\_time\_limit 8 --speak



Use text input directly (skip speech recognition):

bashpython speech\_translator.py --text "سلام من مجید هستم" --target\_lang en --speak



List supported languages:

bashpython speech\_translator.py --list\_languages



Show translation history:

bashpython speech\_translator.py --show\_history





Usage



Use --text to provide text directly (skips speech recognition).

Use --language to set the speech recognition language (e.g., fa-IR for Persian).

Use --target\_lang to set the translation target (e.g., en for English).

Use --timeout and --phrase\_time\_limit to control listening duration.

Use --voice\_index and --speech\_rate to configure TTS voice and speed.

Use --speak to enable text-to-speech for the translated text.

Use --list\_languages to see supported languages.

Use --show\_history to view past translations.



Example Output

textINFO: Adjusting for ambient noise...

INFO: Listening for speech...

INFO: Processing audio...

Original (fa-IR): سلام من مجید هستم

Translated (en): Hello, I am Majid

INFO: Spoken text: Hello, I am Majid

History:

textTranslation History:

original           translated       source\_lang  target\_lang  timestamp

سلام من مجید هستم  Hello, I am Majid  fa-IR        en           2025-08-07 13:50:00

Improvements and Future Work



Add support for multiple TTS engines (e.g., gTTS for online voices).

Implement a GUI with Tkinter or Streamlit for interactive use.

Add batch processing for translating multiple phrases.

Support additional speech recognition APIs (e.g., Microsoft Azure).

Add unit tests with pytest for speech recognition and translation functions.



License

MIT License

