import argparse
import logging
import threading
import time
import wave
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
from scipy.signal import butter, lfilter

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, chunk=1024, format=pyaudio.paInt16, channels=1, rate=44100):
        self.chunk = chunk
        self.format = format
        self.channels = channels
        self.rate = rate
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.stream = None
        self.is_recording = False

    def start_recording(self, duration=5):
        """Start recording audio for a given duration using a separate thread."""
        try:
            self.stream = self.p.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.chunk)
            logger.info("Recording started...")
            self.is_recording = True
            threading.Thread(target=self._record_thread, args=(duration,)).start()
        except Exception as e:
            logger.error(f"Error starting recording: {e}")

    def _record_thread(self, duration):
        """Thread for recording audio."""
        start_time = time.time()
        while self.is_recording and (time.time() - start_time) < duration:
            try:
                data = self.stream.read(self.chunk)
                self.frames.append(data)
            except Exception as e:
                logger.error(f"Error during recording: {e}")
                break
        self.stop_recording()

    def stop_recording(self):
        """Stop the recording stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.is_recording = False
        logger.info("Recording stopped.")

    def save_to_wav(self, filename="output.wav"):
        """Save recorded frames to a WAV file."""
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            logger.info(f"Audio saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving WAV file: {e}")

    def get_audio_data(self):
        """Convert frames to numpy array."""
        return np.frombuffer(b''.join(self.frames), dtype=np.int16)

    def apply_noise_reduction(self, data, lowcut=100, highcut=3000, order=5):
        """Apply bandpass filter for noise reduction using Butterworth filter."""
        nyq = 0.5 * self.rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def compute_fft(self, data):
        """Compute FFT and frequencies."""
        fft_data = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1 / self.rate)
        return freqs[:len(freqs)//2], np.abs(fft_data)[:len(fft_data)//2]

    def play_audio(self, data):
        """Play the audio data."""
        try:
            stream = self.p.open(format=self.format,
                                 channels=self.channels,
                                 rate=self.rate,
                                 output=True,
                                 frames_per_buffer=self.chunk)
            stream.write(data.tobytes())
            stream.stop_stream()
            stream.close()
            logger.info("Playback finished.")
        except Exception as e:
            logger.error(f"Error during playback: {e}")

    def plot_signal(self, data):
        """Plot the audio signal."""
        plt.figure(figsize=(10, 4))
        plt.plot(data)
        plt.title("Audio Signal")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.show()

    def plot_spectrogram(self, data):
        """Plot the spectrogram."""
        plt.figure(figsize=(10, 4))
        plt.specgram(data, Fs=self.rate)
        plt.title("Spectrogram")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.show()

    def plot_fft(self, freqs, fft_vals):
        """Plot the FFT."""
        plt.figure(figsize=(10, 4))
        plt.plot(freqs, fft_vals)
        plt.title("FFT of Audio Signal")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.show()

    def terminate(self):
        """Terminate PyAudio."""
        self.p.terminate()

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Audio Recorder and Processor")
    parser.add_argument("--record_duration", type=int, default=5, help="Recording duration in seconds.")
    parser.add_argument("--output_file", type=str, default="output.wav", help="Output WAV file name.")
    parser.add_argument("--play", action="store_true", help="Play the processed audio.")
    parser.add_argument("--plot", action="store_true", help="Plot signal, spectrogram, and FFT.")
    parser.add_argument("--noise_reduction", action="store_true", help="Apply noise reduction filter.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    processor = AudioProcessor()

    # Record
    processor.start_recording(args.record_duration)
    time.sleep(args.record_duration + 1)  # Wait for recording to finish

    # Get data
    data = processor.get_audio_data()

    # Apply noise reduction if requested
    if args.noise_reduction:
        data = processor.apply_noise_reduction(data)
        logger.info("Noise reduction applied.")

    # Compute FFT
    freqs, fft_vals = processor.compute_fft(data)

    # Plot if requested
    if args.plot:
        processor.plot_signal(data)
        processor.plot_spectrogram(data)
        processor.plot_fft(freqs, fft_vals)

    # Save to WAV
    processor.save_to_wav(args.output_file)

    # Play if requested
    if args.play:
        processor.play_audio(data)

    # Clean up
    processor.terminate()