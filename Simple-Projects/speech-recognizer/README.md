\# Speech Recognizer



\## Overview

This simple Python project captures spoken input via microphone and converts it to text using the Google Speech Recognition API. It features a robust CLI interface, history tracking in a JSON file, and support for multiple languages, making it an excellent portfolio piece for demonstrating speech processing, error handling, and modular design.



\## Features

\- \*\*Speech Recognition\*\*: Converts audio input to text in Persian (or other languages) using Google Speech Recognition API.

\- \*\*History Tracking\*\*: Saves recognized text with timestamps to a JSON file and displays it in a formatted table.

\- \*\*Multi-Language Support\*\*: Lists supported languages for easy configuration.

\- \*\*CLI Interface\*\*: Configurable via command-line arguments for language, timeouts, and thresholds.

\- \*\*Error Handling\*\*: Robust handling of microphone issues, API errors, and invalid inputs.

\- \*\*Logging\*\*: Detailed logs for debugging and monitoring.

\- \*\*Ambient Noise Adjustment\*\*: Automatically adjusts for background noise to improve recognition accuracy.



\## Requirements

\- Python 3.8+

\- Libraries: `speechrecognition`, `pyaudio`, `pandas`



Install dependencies:

```bash

pip install speechrecognition pyaudio pandas



How to Run



Run with default settings (Persian, 5s timeout, 10s phrase limit):

bashpython speech\_recognizer.py



Specify language or timeouts:

bashpython speech\_recognizer.py --language en-US --timeout 3 --phrase\_time\_limit 8



List supported languages:

bashpython speech\_recognizer.py --list\_languages



Show recognition history:

bashpython speech\_recognizer.py --show\_history





Usage



Use --language to set the recognition language (e.g., fa-IR for Persian, en-US for English).

Use --timeout to set how long to wait for speech input.

Use --phrase\_time\_limit to limit the duration of a single phrase.

Use --energy\_threshold to adjust sensitivity to sound levels.

Use --pause\_threshold to set the pause duration before stopping recording.

Use --list\_languages to see supported languages.

Use --show\_history to view past recognitions.

History is saved in speech\_history.json.



Example Output

textINFO: Adjusting for ambient noise...

INFO: Listening for speech...

INFO: Processing audio...

Recognized (fa-IR): سلام من مجید هستم

History:

textSpeech Recognition History:

text                language  timestamp

سلام من مجید هستم  fa-IR     2025-08-07 18:30:00

Improvements and Future Work



Add support for additional speech recognition APIs (e.g., Microsoft Azure, DeepSpeech).

Implement a GUI with Tkinter or Streamlit for interactive use.

Add audio recording and saving to WAV files.

Support batch processing for multiple audio inputs.

Unit tests with pytest for recognition and history functions.

