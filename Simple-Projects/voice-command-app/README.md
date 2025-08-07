\# Voice Command App



\## Overview

This simple Python project uses speech recognition to process voice commands in Persian (or other languages) and execute actions, such as opening Google Chrome. It leverages the `speech\_recognition` library with Google Speech Recognition API, includes a CLI interface, and supports cross-platform execution. The project is designed as a professional portfolio piece, showcasing robust error handling, logging, and modular design.



\## Features

\- \*\*Speech Recognition\*\*: Convert spoken commands to text using Google Speech Recognition API.

\- \*\*Command Processing\*\*: Execute actions (e.g., open Google Chrome) based on keyword detection.

\- \*\*Cross-Platform Support\*\*: Works on Windows, macOS, and Linux.

\- \*\*CLI Interface\*\*: Configure language, timeout, and phrase duration via command-line arguments.

\- \*\*Error Handling\*\*: Robust handling of microphone issues, API errors, and invalid inputs.

\- \*\*Logging\*\*: Detailed logs for debugging and monitoring.

\- \*\*Ambient Noise Adjustment\*\*: Automatically adjusts for background noise to improve recognition accuracy.



\## Requirements

\- Python 3.8+

\- Libraries: `speech\_recognition`, `PyAudio`



Install dependencies:

```bash

pip install speechrecognition pyaudio





Note: PyAudio requires system dependencies like PortAudio. Install it:



macOS: brew install portaudio

Ubuntu: sudo apt-get install portaudio19-dev

Windows: Usually included with PyAudio, but ensure pip install pyaudio succeeds.



How to Run



Run with default settings (Persian, 5s timeout, 10s phrase limit):

bashpython voice\_command\_app.py



Specify language or timeouts:

bashpython voice\_command\_app.py --language en-US --timeout 3 --phrase\_time\_limit 8



Example command (in Persian):



Say: "باز کن گوگل کروم"

Output: Opens Google Chrome and logs the action.







Usage



Use --language to set the recognition language (e.g., fa-IR for Persian, en-US for English).

Use --timeout to set how long to wait for speech input.

Use --phrase\_time\_limit to limit the duration of a single phrase.

Currently supports opening Google Chrome when keywords "گوگل کروم" and "باز" are detected (case-insensitive).



Example Output

textINFO: Adjusting for ambient noise...

INFO: Listening for command...

INFO: Processing audio...

Recognized: باز کن گوگل کروم

INFO: Google Chrome opened successfully.

Improvements and Future Work



Add more commands (e.g., open other apps, control system settings).

Implement a GUI with Tkinter for visual feedback.

Support additional speech recognition APIs (e.g., Microsoft Azure, DeepSpeech).

Add unit tests with pytest for command processing and recognition.

Store command history in a JSON file for later analysis.



License

MIT License

