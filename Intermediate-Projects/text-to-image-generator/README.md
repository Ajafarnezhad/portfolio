markdown# Text-to-Image Generator



\## Overview

This intermediate Python project creates a Telegram bot that converts user-provided text (in Persian or any language) into images using the Stable Diffusion XL model via the Replicate API. It features text translation to English, image generation with customizable parameters, real-time interaction, and robust error handling. The project includes a CLI interface for configuration, logging for debugging, and is designed as a professional portfolio piece for AI-driven image synthesis.



\## Features

\- \*\*Text Translation\*\*: Automatically translates input text to English using GoogleTranslator.

\- \*\*Image Generation\*\*: Utilizes Stable Diffusion XL on Replicate for high-quality image creation with configurable parameters (e.g., resolution, guidance scale).

\- \*\*Telegram Integration\*\*: Responds to /start command and processes text messages to generate images.

\- \*\*CLI Interface\*\*: Configurable via command-line arguments for bot token and model settings.

\- \*\*Error Handling \& Logging\*\*: Comprehensive checks and detailed logs for debugging and error management.

\- \*\*Model Persistence\*\*: Supports saving/loading configuration (optional extension).



\## Requirements

\- Python 3.8+

\- Libraries: `telegram`, `deep-translator`, `requests`, `Pillow`, `replicate`



Install dependencies:

```bash

pip install python-telegram-bot deep-translator requests Pillow replicate

Setup



Obtain Telegram Bot Token:



Create a bot via BotFather on Telegram and copy the token.





Replicate API Token:



Sign up at Replicate, get an API token, and set it as an environment variable:

bashexport REPLICATE\_API\_TOKEN=your\_api\_token



Or add it to your system environment variables.







How to Run



Run the bot:

bashpython text\_to\_image\_generator.py --bot\_token your\_telegram\_bot\_token



Interact:



Start the bot with /start in Telegram.

Send a Persian text (e.g., "یک منظره زیبا از کوهستان") to receive a generated image.







Example Output



Telegram Interaction:

textUser: یک منظره زیبا از کوهستان

Bot: ✅ Your generated image is ready! \[Image]



Logs:

text2025-08-07 18:00:00 - \_\_main\_\_ - INFO - Received text: یک منظره زیبا از کوهستان

2025-08-07 18:00:01 - \_\_main\_\_ - INFO - Translated text: A beautiful mountain landscape

2025-08-07 18:00:03 - \_\_main\_\_ - INFO - Generated image URL: https://...





Improvements and Future Work



Add support for multiple output images with batch processing.

Implement advanced parameters (e.g., style transfer, custom resolutions) via CLI.

Deploy on a cloud service (e.g., Heroku) for 24/7 availability.

Add image validation and preprocessing for better quality control.

Unit tests with pytest for translation and image generation functions.



License

MIT License

