\# Text-to-Video Generator



\## Overview

This intermediate Python project develops a Telegram bot that transforms user-provided text (in Persian or any language) into videos using the VideoCrafter model via the Replicate API. It includes text translation to English, real-time video generation with customizable parameters, interactive status updates, and robust error handling. The project features a CLI interface for configuration, logging for debugging, and is designed as a professional portfolio piece for advanced AI-driven video synthesis.



\## Features

\- \*\*Text Translation\*\*: Automatically translates input text to English using GoogleTranslator.

\- \*\*Video Generation\*\*: Utilizes VideoCrafter on Replicate for high-quality video creation with configurable settings (e.g., FPS, steps).

\- \*\*Telegram Integration\*\*: Responds to /start command, processes text messages, and provides status updates during generation.

\- \*\*CLI Interface\*\*: Configurable via command-line arguments for bot token and model settings.

\- \*\*Real-Time Updates\*\*: Informs users about text reception, processing, and video delivery.

\- \*\*Error Handling \& Logging\*\*: Comprehensive checks and detailed logs for debugging and error management.

\- \*\*Model Persistence\*\*: Supports saving/loading configuration (optional extension).



\## Requirements

\- Python 3.8+

\- Libraries: `python-telegram-bot`, `deep-translator`, `requests`, `replicate`



Install dependencies:

```bash

pip install python-telegram-bot deep-translator requests replicate

Setup



Obtain Telegram Bot Token:



Create a bot via BotFather on Telegram and copy the token.





Replicate API Token:



Sign up at Replicate, get an API token, and set it as an environment variable:

bashexport REPLICATE\_API\_TOKEN=your\_api\_token



Or add it to your system environment variables.







How to Run



Run the bot:

bashpython text\_to\_video\_generator.py --bot\_token your\_telegram\_bot\_token



Interact:



Start the bot with /start in Telegram.

Send a Persian text (e.g., "یک انیمیشن از یک سگ در پارک") to receive a generated video.







Example Output



Telegram Interaction:

textUser: یک انیمیشن از یک سگ در پارک

Bot: Your text has been successfully received.

Bot: Your text is being processed. Generating video... ⌛ (This process may take a few minutes.)

Bot: Your video has been generated. Sending now... \[Video] ✅ Your generated video is ready!



Logs:

text2025-08-07 20:00:00 - \_\_main\_\_ - INFO - Received text: یک انیمیشن از یک سگ در پارک

2025-08-07 20:00:01 - \_\_main\_\_ - INFO - Translated text: An animation of a dog in the park

2025-08-07 20:00:05 - \_\_main\_\_ - INFO - Generated video URL: https://...





Improvements and Future Work



Add support for custom video durations and resolutions.

Implement advanced parameters (e.g., style control, motion enhancement) via CLI.

Deploy on a cloud platform (e.g., Heroku, AWS) for scalability.

Add video quality validation and preprocessing for improved output.

Unit tests with pytest for translation, generation, and Telegram interactions.



License

MIT License

