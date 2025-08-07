import argparse
import logging
import sys
from typing import Optional, List, Dict
import speech_recognition as sr
import pyttsx3
from deep_translator import GoogleTranslator
import json
import pandas as pd
import platform

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SpeechTranslator:
    def __init__(self, language: str = "fa-IR", voice_index: int = 0, speech_rate: int = 150):
        """Initialize the SpeechTranslator with recognizer, TTS engine, and translator."""
        self.recognizer = sr.Recognizer()
        self.language = language
        self.engine = pyttsx3.init()
        self.set_voice(voice_index)
        self.engine.setProperty('rate', speech_rate)
        self.translator = GoogleTranslator(source="auto", target="en")
        self.history_file = "speech_translation_history.json"
        self.history = self.load_history()
        self.supported_languages = [
            {"code": "fa-IR", "name": "Persian"},
            {"code": "en-US", "name": "English (US)"},
            {"code": "es-ES", "name": "Spanish"},
            {"code": "fr-FR", "name": "French"},
            {"code": "de-DE", "name": "German"},
            {"code": "zh-CN", "name": "Chinese (Simplified)"},
        ]

    def set_voice(self, voice_index: int) -> None:
        """Set the TTS voice based on index."""
        try:
            voices = self.engine.getProperty('voices')
            if 0 <= voice_index < len(voices):
                self.engine.setProperty('voice', voices[voice_index].id)
                logger.info(f"Voice set to {voices[voice_index].name}")
            else:
                logger.warning(f"Invalid voice index {voice_index}. Using default voice.")
        except Exception as e:
            logger.error(f"Error setting voice: {e}")

    def load_history(self) -> List[Dict]:
        """Load translation history from JSON file."""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            return []

    def save_history(self) -> None:
        """Save translation history to JSON file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            logger.info(f"Translation history saved to {self.history_file}")
        except Exception as e:
            logger.error(f"Error saving history: {e}")

    def listen(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[str]:
        """Listen for audio input and convert it to text."""
        with sr.Microphone() as source:
            try:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.recognizer.pause_threshold = 1.0
                logger.info("Listening for speech...")
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

    def translate_text(self, text: str, target_lang: str) -> Optional[str]:
        """Translate text to the target language."""
        try:
            self.translator.target = target_lang
            translated = self.translator.translate(text)
            if not translated:
                raise ValueError("Empty translation returned.")
            logger.info(f"Translated text to {target_lang}: {translated}")
            self.history.append({
                "original": text,
                "translated": translated,
                "source_lang": self.language,
                "target_lang": target_lang,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            self.save_history()
            return translated
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return None

    def speak(self, text: str) -> None:
        """Convert text to speech."""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            logger.info(f"Spoken text: {text}")
        except Exception as e:
            logger.error(f"Error during text-to-speech: {e}")

    def display_history(self) -> None:
        """Display translation history in a formatted table."""
        if not self.history:
            logger.info("No translation history available.")
            print("No translation history available.")
            return
        df = pd.DataFrame(self.history)
        print("\nTranslation History:")
        print(df.to_string(index=False))

    def list_supported_languages(self) -> None:
        """List supported languages."""
        print("\nSupported Languages:")
        for lang in self.supported_languages:
            print(f"{lang['code']}: {lang['name']}")

    def run(self, args: argparse.Namespace) -> None:
        """Run the speech translator based on CLI arguments."""
        try:
            if args.list_languages:
                self.list_supported_languages()
                return
            if args.show_history:
                self.display_history()
                return
            if args.text:
                text = args.text
            else:
                text = self.listen(timeout=args.timeout, phrase_time_limit=args.phrase_time_limit)
                if not text:
                    print("No speech recognized.")
                    return
            print(f"\nOriginal ({self.language}): {text}")
            translated = self.translate_text(text, args.target_lang)
            if translated:
                print(f"Translated ({args.target_lang}): {translated}")
                if args.speak:
                    self.speak(translated)
        except KeyboardInterrupt:
            logger.info("Program terminated by user.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            sys.exit(1)

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Speech-to-Text-to-Speech Translator with history tracking.")
    parser.add_argument("--text", type=str, help="Text to translate (skip speech recognition if provided).")
    parser.add_argument("--language", type=str, default="fa-IR", 
                        help="Source language for speech recognition (e.g., 'fa-IR' for Persian).")
    parser.add_argument("--target_lang", type=str, default="en", 
                        help="Target language for translation (e.g., 'en' for English).")
    parser.add_argument("--timeout", type=int, default=5, 
                        help="Timeout for listening to speech in seconds.")
    parser.add_argument("--phrase_time_limit", type=int, default=10, 
                        help="Maximum duration of a phrase in seconds.")
    parser.add_argument("--voice_index", type=int, default=0, 
                        help="Index of the TTS voice to use.")
    parser.add_argument("--speech_rate", type=int, default=150, 
                        help="Speech rate for text-to-speech (default: 150).")
    parser.add_argument("--speak", action="store_true", help="Speak the translated text.")
    parser.add_argument("--list_languages", action="store_true", help="List supported languages.")
    parser.add_argument("--show_history", action="store_true", help="Show translation history.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    app = SpeechTranslator(language=args.language, voice_index=args.voice_index, speech_rate=args.speech_rate)
    app.run(args)