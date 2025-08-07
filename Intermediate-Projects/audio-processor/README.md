\# Audio Recorder and Processor



\## Overview

This intermediate Python project records audio from a microphone, processes it (including noise reduction via Butterworth filter, volume adjustment, and FFT analysis), plays it back, and saves it as a WAV file. It uses PyAudio for audio I/O, NumPy and SciPy for processing, and Matplotlib for visualizations. The project follows best practices like threading for real-time operations, error handling, and a CLI interface, making it suitable for portfolio demonstration of audio signal processing.



\## Features

\- \*\*Recording\*\*: Capture audio for a specified duration with configurable rate, channels, and chunk size.

\- \*\*Processing\*\*: Noise reduction using bandpass filter, FFT computation for frequency analysis.

\- \*\*Playback\*\*: Play processed audio in real-time.

\- \*\*Visualization\*\*: Plot signal waveform, spectrogram, and FFT.

\- \*\*Saving\*\*: Export to WAV file.

\- \*\*CLI Options\*\*: Control duration, output file, playback, plotting, and noise reduction via arguments.

\- \*\*Error Handling \& Logging\*\*: Robust try-except blocks and logging for debugging.

\- \*\*Threading\*\*: Non-blocking recording to prevent UI freezes.



\## Requirements

\- Python 3.8+

\- Libraries: `pyaudio`, `numpy`, `scipy`, `matplotlib`, `wave` (standard library).



Install dependencies:

```bash

pip install pyaudio numpy scipy matplotlib







Note: PyAudio may require system dependencies like PortAudio (install via brew install portaudio on macOS, apt-get install portaudio19-dev on Ubuntu, or similar).





How to Run



Run the script with default settings (5s recording):

python audio\_processor.py



With options (e.g., 10s recording, apply noise reduction, plot, and play):

python audio\_processor.py --record\_duration 10 --noise\_reduction --plot --play --output\_file my\_audio.wav





The script will record, process, save to WAV, and optionally plot/play.





Example Output



Logs metrics like "Recording started..." and "Noise reduction applied."

Generates plots for signal, spectrogram, and FFT if --plot is used.

Saves a clean WAV file with reduced noise if --noise\_reduction is enabled.



Improvements and Future Work



Add real-time callback mode for continuous streaming.

Integrate advanced noise reduction (e.g., using noisereduce library if allowed).

Add GUI with Tkinter or Streamlit for interactive control.

Support for multiple audio formats (e.g., MP3 via pydub).



License



MIT License

