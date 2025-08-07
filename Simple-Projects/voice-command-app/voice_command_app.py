import argparse
import logging
import sys
import os
import platform
import subprocess
from typing import Optional
import speech_recognition as sr

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class VoiceCommandApp:
    def __init__(self, language: str = "fa-IR"):
        """Initialize the VoiceCommandApp with a recognizer and language."""
        self.recognizer = sr.Recognizer()
        self.language = language
        self.commands = {
            "open_chrome": {
                "keywords": ["گوگل کروم", "باز"],
                "action": self.open_chrome
            },
            # Add more commands here as needed
        }

    def open_chrome(self) -> None:
        """Open Google Chrome based on the operating system."""
        try:
            if platform.system() == "Windows":
                chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
                if not os.path.exists(chrome_path):
                    raise FileNotFoundError("Chrome executable not found at the specified path.")
                subprocess.run([chrome_path], check=True)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", "-a", "Google Chrome"], check=True)
            elif platform.system() == "Linux":
                subprocess.run(["google-chrome"], check=True)
            else:
                raise OSError("Unsupported operating system.")
            logger.info("Google Chrome opened successfully.")
        except Exception as e:
            logger.error(f"Error opening Chrome: {e}")
            raise

    def listen(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[str]:
        """Listen for audio input and convert it to text."""
        with sr.Microphone() as source:
            try:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.recognizer.pause_threshold = 1.0
                logger.info("Listening for command...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                logger.info("Processing audio...")
                text = self.recognizer.recognize_google(audio, language=self.language)
                logger.info(f"Recognized text: {text}")
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
                logger.error(f"Unexpected error during listening: {e}")
                return None

    def process_command(self, text: Optional[str]) -> None:
        """Process the recognized text and execute matching commands."""
        if not text:
            logger.info("No valid text to process.")
            return

        text = text.lower()
        for command_name, command_data in self.commands.items():
            keywords = command_data["keywords"]
            if all(keyword.lower() in text for keyword in keywords):
                try:
                    command_data["action"]()
                    logger.info(f"Executed command: {command_name}")
                except Exception as e:
                    logger.error(f"Error executing command {command_name}: {e}")

    def run(self, args: argparse.Namespace) -> None:
        """Run the voice command app based on CLI arguments."""
        try:
            text = self.listen(timeout=args.timeout, phrase_time_limit=args.phrase_time_limit)
            if text:
                print(f"\nRecognized: {text}")
                self.process_command(text)
            else:
                print("No command recognized.")
        except KeyboardInterrupt:
            logger.info("Program terminated by user.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            sys.exit(1)

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Voice Command App for executing actions based on speech input.")
    parser.add_argument("--language", type=str, default="fa-IR", 
                        help="Language for speech recognition (e.g., 'fa-IR' for Persian, 'en-US' for English).")
    parser.add_argument("--timeout", type=int, default=5, 
                        help="Timeout for listening to speech in seconds.")
    parser.add_argument("--phrase_time_limit", type=int, default=10, 
                        help="Maximum duration of a phrase in seconds.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    app = VoiceCommandApp(language=args.language)
    app.run(args)