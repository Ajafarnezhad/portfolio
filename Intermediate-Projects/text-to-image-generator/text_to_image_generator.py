import argparse
import logging
import sys
import os
import time
import requests
from PIL import Image
from deep_translator import GoogleTranslator
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import replicate
from typing import Optional

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class TextToImageGenerator:
    def __init__(self, bot_token: str, model_path: str = "models/text_to_image_config.json"):
        """Initialize the TextToImageGenerator with Telegram bot token and model configuration."""
        self.bot_token = bot_token
        self.model_path = model_path
        self.application = None
        self.default_params = {
            "width": 512,
            "height": 512,
            "refine": "expert_ensemble_refiner",
            "scheduler": "K_EULER",
            "lora_scale": 0.6,
            "num_outputs": 1,
            "guidance_scale": 7.5,
            "apply_watermark": False,
            "high_noise_frac": 0.8,
            "negative_prompt": "",
            "prompt_strength": 0.8,
            "num_inference_steps": 25
        }

    def initialize_bot(self) -> None:
        """Initialize the Telegram bot application."""
        try:
            self.application = ApplicationBuilder().token(self.bot_token).build()
            logger.info("Telegram bot initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}")
            sys.exit(1)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /start command with a welcome message."""
        welcome_message = (
            "Hello! Welcome to the AIvLearn Text-to-Image Bot. 😍\n"
            "Please send me a text in Persian, and I'll generate an image based on it for you!"
        )
        await update.message.reply_text(welcome_message)

    async def make_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Generate an image from the user's text input and send it back."""
        try:
            text = update.message.text
            logger.info(f"Received text: {text}")
            
            # Translate to English
            en_text = GoogleTranslator(source="auto", target="en").translate(text)
            logger.info(f"Translated text: {en_text}")
            
            # Generate image using Replicate
            output = replicate.run(
                "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                input={**self.default_params, "prompt": en_text}
            )
            image_url = output[0]
            logger.info(f"Generated image URL: {image_url}")
            
            # Send the image to the user
            await update.message.reply_photo(photo=image_url, caption="✅ Your generated image is ready!")
            
        except Exception as e:
            logger.error(f"Error during image generation: {e}")
            await update.message.reply_text("Sorry, an error occurred while generating the image. Please try again later.")

    def setup_handlers(self) -> None:
        """Set up command and message handlers for the bot."""
        start_handler = CommandHandler('start', self.start)
        message_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, self.make_image)
        self.application.add_handler(start_handler)
        self.application.add_handler(message_handler)

    def run(self) -> None:
        """Run the bot with error handling."""
        try:
            self.initialize_bot()
            self.setup_handlers()
            logger.info("Bot is running...")
            self.application.run_polling()
        except Exception as e:
            logger.error(f"Bot execution failed: {e}")
            sys.exit(1)

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Text-to-Image Generator Bot using Telegram and Replicate.")
    parser.add_argument("--bot_token", type=str, required=True, help="Telegram bot token from BotFather.")
    parser.add_argument("--model_path", type=str, default="models/text_to_image_config.json", help="Path to save/load model configuration.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    generator = TextToImageGenerator(bot_token=args.bot_token, model_path=args.model_path)
    generator.run()