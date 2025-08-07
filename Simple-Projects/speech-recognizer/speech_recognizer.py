import argparse
import logging
import sys
import time
from typing import Optional, List, Dict
import speech_recognition as sr
import json
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SpeechRecognizer:
    def __init__(self, language: str = "fa-IR", energy_threshold: int = 300, pause_threshold: float = 1.0):
        """Initialize the SpeechRecognizer with configurable parameters."""
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.language = language
        self.history_file = "speech_history.json"
        self.history = self.load_history()
        self.supported_languages = [
            {"code": "fa-IR", "name": "Persian"},
            {"code": "en-US", "name": "English (US)"},
            {"code": "es-ES", "name": "Spanish"},
            {"code": "fr-FR", "name": "French"},
            {"code": "de-DE", "name": "German"},
            {"code": "zh-CN", "name": "Chinese (Simplified)"},
        ]

    def load_history(self) -> List[Dict]:
        """Load speech recognition history from JSON file."""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            return []

    def save_history(self) -> None:
        """Save speech recognition history to JSON file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            logger.info(f"Speech history saved to {self.history_file}")
        except Exception as e:
            logger.error(f"Error saving history: {e}")

    def recognize_speech(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[str]:
        """Capture and recognize speech from microphone."""
        with sr.Microphone() as source:
            try:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.info("Listening for speech...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                logger.info("Processing audio...")
                text = self.recognizer.recognize_google(audio, language=self.language)
                logger.info(f"Recognized text: {text}")
                self.history.append({
                    "text": text,
                    "language": self.language,
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                self.save_history()
                return text
            except sr.WaitTimeoutError:
                logger.warning("No speech detected within timeout.")
                return None
            except sr.UnknownValueError:
                logger.warning("Could not understand the audio.")
                return None
            except sr.RequestError as e:
                logger.error(f"Speech recognition error: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error during recognition: {e}")
                return None

    def display_history(self) -> None:
        """Display speech recognition history in a formatted table."""
        if not self.history:
            logger.info("No speech recognition history available.")
            print("No speech recognition history available.")
            return
        df = pd.DataFrame(self.history)
        print("\nSpeech Recognition History:")
        print(df.to_string(index=False))

    def list_supported_languages(self) -> None:
        """List supported languages for speech recognition."""
        print("\nSupported Languages:")
        for lang in self.supported_languages:
            print(f"{lang['code']}: {lang['name']}")

    def run(self, args: argparse.Namespace) -> None:
        """Run the speech recognizer based on CLI arguments."""
        try:
            if args.list_languages:
                self.list_supported_languages()
                return
            if args.show_history:
                self.display_history()
                return
            
            text = self.recognize_speech(timeout=args.timeout, phrase_time_limit=args.phrase_time_limit)
            if text:
                print(f"\nRecognized ({self.language}): {text}")
            else:
                print("No speech recognized.")
        except KeyboardInterrupt:
            logger.info("Program terminated by user.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            sys.exit(1)

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Speech Recognizer with history tracking and multi-language support.")
    parser.add_argument("--language", type=str, default="fa-IR", 
                        help="Language for speech recognition (e.g., 'fa-IR' for Persian, 'en-US' for English).")
    parser.add_argument("--timeout", type=int, default=5, 
                        help="Timeout for listening to speech in seconds.")
    parser.add_argument("--phrase_time_limit", type=int, default=10, 
                        help="Maximum duration of a phrase in seconds.")
    parser.add_argument("--energy_threshold", type=int, default=300, 
                        help="Energy threshold for speech detection.")
    parser.add_argument("--pause_threshold", type=float, default=1.0, 
                        help="Pause threshold for speech detection in seconds.")
    parser.add_argument("--list_languages", action="store_true", help="List supported languages.")
    parser.add_argument("--show_history", action="store_true", help="Show speech recognition history.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    recognizer = SpeechRecognizer(
        language=args.language,
        energy_threshold=args.energy_threshold,
        pause_threshold=args.pause_threshold
    )
    recognizer.run(args)