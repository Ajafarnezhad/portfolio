\# Translator App



\## Overview

This simple Python project provides a command-line interface for translating text between multiple languages using the `deep-translator` library. It includes features like translation history, supported language listing, and error handling, making it a polished portfolio piece for demonstrating API integration, file handling, and CLI design.



\## Features

\- \*\*Translation\*\*: Translate text to multiple languages using Google Translate API via `deep-translator`.

\- \*\*History Tracking\*\*: Save translations with timestamps to a JSON file and display them in a table.

\- \*\*Supported Languages\*\*: List available languages with their codes.

\- \*\*CLI Interface\*\*: Flexible command-line arguments for translation, history, and language listing.

\- \*\*Error Handling\*\*: Robust handling of API errors and file operations.

\- \*\*Logging\*\*: Detailed logs for debugging and tracking operations.



\## Requirements

\- Python 3.8+

\- Libraries: `deep-translator`, `pandas`



Install dependencies:

```bash

pip install deep-translator pandas





How to Run



Translate text:

bashpython translator\_app.py --text "سلام من امیر هستم" --target\_lang en

Output:

textOriginal: سلام من امیر هستم

Translated (en): Hello, I am amir



List supported languages:

bashpython translator\_app.py --list\_languages



Show translation history:

bashpython translator\_app.py --show\_history





Usage



Use --text to specify the text to translate.

Use --target\_lang to choose the target language (default: en).

Use --list\_languages to see available languages.

Use --show\_history to view past translations.

Translation history is saved in translation\_history.json.



Example Output



Translation:

textOriginal: سلام من امیر هستم و میخواهم هوش مصنوعی را آموزش ببینم

Translated (en): Hello, I am amir, and I want to learn artificial intelligence



History:

textTranslation History:

original                                translated           target\_lang  timestamp

سلام من امیر هستم                     Hello, I am amir    en           2025-08-07 13:39:00





Improvements and Future Work



Add batch translation for multiple sentences or files.

Implement a GUI using Tkinter or a web app with Flask.

Support additional translation APIs (e.g., Microsoft Translator).

Add unit tests with pytest for translation and history functions.

Cache API results to reduce calls for repeated translations.



License

MIT License

