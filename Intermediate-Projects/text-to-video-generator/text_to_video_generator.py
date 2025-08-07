import argparse
import logging
import sys
import os
import time
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from deep_translator import GoogleTranslator
import replicate
from typing import Optional

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class TextToVideoGenerator:
    def __init__(self, bot_token: str, model_config_path: str = "models/text_to_video_config.json"):
        """Initialize the TextToVideoGenerator with Telegram bot token and model configuration path."""
        self.bot_token = bot_token
        self.model_config_path = model_config_path
        self.application = None
        self.default_params = {
            "save_fps": 10,
            "ddim_steps": 50,
            "unconditional_guidance_scale": 12
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
            "Hello! Welcome to the AIvLearn Text-to-Video Bot. 😍\n"
            "Please send me a text in Persian, and I'll generate a video based on it for you!"
        )
        await update.message.reply_text(welcome_message)

    async def make_video(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Generate a video from the user's text input and send it back."""
        try:
            text = update.message.text
            logger.info(f"Received text: {text}")
            
            # Send initial acknowledgment
            msg = await update.message.reply_text("Your text has been successfully received.")

            # Translate to English
            en_text = GoogleTranslator(source="auto", target="en").translate(text)
            logger.info(f"Translated text: {en_text}")

            # Update status
            await msg.edit_text("Your text is being processed. Generating video... ⌛\nThis process may take a few minutes.")

            # Generate video using Replicate
            output = replicate.run(
                "cjwbw/videocrafter:02edcff3e9d2d11dcc27e530773d988df25462b1ee93ed0257b6f246de4797c8",
                input={**self.default_params, "prompt": en_text}
            )
            video_url = output
            logger.info(f"Generated video URL: {video_url}")

            # Update status and send video
            await msg.edit_text("Your video has been generated. Sending now...")
            await msg.delete()
            await update.message.reply_video(video=video_url, caption="✅ Your generated video is ready!")

        except Exception as e:
            logger.error(f"Error during video generation: {e}")
            await update.message.reply_text("Sorry, an error occurred while generating the video. Please try again later.")

    def setup_handlers(self) -> None:
        """Set up command and message handlers for the bot."""
        start_handler = CommandHandler('start', self.start)
        message_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, self.make_video)
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
    parser = argparse.ArgumentParser(description="Text-to-Video Generator Bot using Telegram and Replicate.")
    parser.add_argument("--bot_token", type=str, required=True, help="Telegram bot token from BotFather.")
    parser.add_argument("--model_config_path", type=str, default="models/text_to_video_config.json", help="Path to save/load model configuration.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    generator = TextToVideoGenerator(bot_token=args.bot_token, model_config_path=args.model_config_path)
    generator.run()