# Audio Recorder & Processor: Capture and Transform Sound 🎙️✨

Welcome to the **Audio Recorder and Processor**, an intermediate Python project that empowers you to record, process, and analyze audio with ease. Built for audio enthusiasts and developers, this tool leverages PyAudio, NumPy, SciPy, and Matplotlib to record microphone input, apply advanced signal processing (like noise reduction and FFT analysis), and visualize results. With a sleek CLI interface, threading for smooth operation, and robust error handling, it’s a perfect addition to your portfolio for showcasing audio signal processing expertise.

---

## 🌟 Project Highlights
This project combines real-time audio recording, sophisticated processing, and insightful visualizations in a modular and user-friendly package. It’s designed to demonstrate your skills in audio processing, threading, and CLI development while following best practices.

---

## 🚀 Features
- **Audio Recording**: Capture high-quality audio from your microphone with customizable duration, sample rate, and channels.
- **Signal Processing**: Apply noise reduction with a Butterworth bandpass filter and perform FFT for frequency analysis.
- **Real-Time Playback**: Listen to processed audio instantly.
- **Visualizations**: Generate stunning plots, including signal waveforms, spectrograms, and FFT graphs.
- **WAV Export**: Save recordings as WAV files for further use.
- **CLI Interface**: Control recording duration, output file, playback, plotting, and noise reduction via intuitive command-line options.
- **Threading**: Non-blocking recording ensures a smooth, responsive experience.
- **Error Handling & Logging**: Robust try-except blocks and detailed logs for seamless debugging.

---

## 🛠️ Requirements
- **Python**: 3.8 or higher
- **Libraries**:
  - `pyaudio`
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `wave` (included in Python standard library)

Install dependencies with:
```bash
pip install pyaudio numpy scipy matplotlib
```

**Note**: PyAudio requires system dependencies like PortAudio. Install it with:
- **macOS**: `brew install portaudio`
- **Ubuntu**: `apt-get install portaudio19-dev`
- **Other systems**: Check PortAudio documentation for setup instructions.

---

## 🎙️ How to Run

### 1. Record with Default Settings
Capture a 5-second audio clip with default parameters:
```bash
python audio_processor.py
```

### 2. Customize Your Recording
Record for 10 seconds, apply noise reduction, plot visualizations, play the audio, and save it:
```bash
python audio_processor.py --record_duration 10 --noise_reduction --plot --play --output_file my_audio.wav
```

### CLI Options
- `--record_duration`: Set recording length in seconds (e.g., `10`).
- `--noise_reduction`: Enable Butterworth bandpass filter for noise reduction.
- `--plot`: Generate waveform, spectrogram, and FFT plots.
- `--play`: Play the processed audio in real-time.
- `--output_file`: Specify the output WAV file (e.g., `my_audio.wav`).

---

## 📈 Example Output
- **Recording**:
  ```
  INFO: Recording 10 seconds of audio...
  INFO: Audio processing complete. Saved to my_audio.wav
  ```
- **Playback**: Hear the processed audio in real-time (if `--play` is enabled).
- **Visualizations**: Plots saved in the working directory:
  - `waveform.png`: Time-domain signal visualization.
  - `spectrogram.png`: Frequency content over time.
  - `fft.png`: Frequency spectrum via FFT analysis.

---

## 🔮 Future Enhancements
Elevate this project with these exciting ideas:
- **Advanced Filters**: Add low-pass, high-pass, or adaptive filters for enhanced noise reduction.
- **Real-Time Processing**: Implement live audio processing during recording.
- **Web App Deployment**: Create a Streamlit or Flask app for an interactive audio interface.
- **Additional Visualizations**: Include phase plots or real-time spectrograms.
- **Unit Testing**: Add `pytest` for robust validation of audio processing pipelines.

---

## 📜 License
This project is licensed under the **MIT License**—use, modify, and share it freely!

Dive into the world of audio with the **Audio Recorder & Processor** and create something extraordinary! 🎵