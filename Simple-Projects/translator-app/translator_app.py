import argparse
import logging
from deep_translator import GoogleTranslator
from deep_translator.exceptions import TranslationError
import pandas as pd
import json
from typing import List, Dict
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TranslatorApp:
    def __init__(self):
        """Initialize the TranslatorApp with supported languages."""
        self.translator = GoogleTranslator(source="auto", target="en")
        self.supported_languages = [
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "zh-cn", "name": "Chinese (Simplified)"},
            {"code": "fa", "name": "Persian"},
            {"code": "ar", "name": "Arabic"},
            {"code": "ja", "name": "Japanese"},
        ]
        self.history_file = "translation_history.json"
        self.history = self.load_history()

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

    def save_history(self):
        """Save translation history to JSON file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            logger.info(f"Translation history saved to {self.history_file}")
        except Exception as e:
            logger.error(f"Error saving history: {e}")

    def translate_text(self, text: str, target_lang: str) -> str:
        """Translate text to the target language."""
        try:
            self.translator.target = target_lang
            translated = self.translator.translate(text)
            if not translated:
                raise TranslationError("Empty translation returned.")
            logger.info(f"Translated text to {target_lang}: {translated}")
            self.history.append({
                "original": text,
                "translated": translated,
                "target_lang": target_lang,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            self.save_history()
            return translated
        except TranslationError as e:
            logger.error(f"Translation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during translation: {e}")
            raise

    def display_history(self) -> None:
        """Display translation history in a formatted table."""
        if not self.history:
            logger.info("No translation history available.")
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
        """Run the translator app based on command-line arguments."""
        if args.list_languages:
            self.list_supported_languages()
            return

        if args.show_history:
            self.display_history()
            return

        if not args.text:
            logger.error("No text provided for translation.")
            sys.exit(1)

        translated = self.translate_text(args.text, args.target_lang)
        print(f"\nOriginal: {args.text}")
        print(f"Translated ({args.target_lang}): {translated}")

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Advanced Translator App with history and multi-language support.")
    parser.add_argument("--text", type=str, help="Text to translate.")
    parser.add_argument("--target_lang", type=str, default="en", 
                        help="Target language code (e.g., 'en' for English, 'es' for Spanish).")
    parser.add_argument("--list_languages", action="store_true", help="List supported languages.")
    parser.add_argument("--show_history", action="store_true", help="Show translation history.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    app = TranslatorApp()
    app.run(args)