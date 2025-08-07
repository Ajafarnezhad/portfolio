Love Bot

This is a simple Python script that automates typing and sending love messages using the pyautogui library. Users can send a custom message repeatedly or choose from random romantic messages, with customizable repetition and delay.

Features



Sends a custom message or random romantic messages.

Configurable number of messages and delay between them.

Safety feature: Stop the script by moving the mouse to the top-left corner.

Error handling for invalid inputs and unexpected issues.

User-friendly prompts and feedback.



Installation



Ensure you have Python 3.x installed.

Install the required library:pip install pyautogui





Clone the repository:git clone https://github.com/Ajafarnezhad/portfolio.git





Navigate to the project directory:cd portfolio/Simple-Projects/love-bot







Usage



Run the script:python love\_bot.py





Follow the prompts to choose a mode (custom or random messages), enter a message (if custom), specify the number of messages, and set the delay.

Focus on the target application (e.g., a chat window) within 3 seconds.

Stop the script anytime by moving the mouse to the top-left corner or pressing Ctrl+C.



Example interaction:

Welcome to Love Bot! This script types and sends love messages.

Choose mode (1 for custom message, 2 for random messages): 2

Enter number of messages to send: 5

Enter delay between messages (seconds, e.g., 0.5): 1

Starting in 3 seconds... Move mouse to top-left corner to stop.

Sent message 1/5: I love you to the moon and back!

Sent message 2/5: You're my sunshine on a cloudy day!

...

Done! All messages sent.



How It Works



Uses pyautogui to simulate typing and pressing the Enter key.

Supports custom messages or a predefined list of romantic messages.

Includes a fail-safe mechanism to stop the script by moving the mouse.

Handles invalid inputs and errors gracefully.

Provides feedback for each message sent.



Improvements Ideas



Add a GUI using Tkinter for a more interactive experience.

Include a configuration file to save favorite messages or settings.

Support for sending messages to specific applications (e.g., via API for chat apps).



Notes



Ensure the target application (e.g., chat window) is focused before the script starts.

Use responsibly to avoid spamming or unintended consequences.



This project is part of my portfolio. Check out my other projects on GitHub: Ajafarnezhad

License: MIT (Free to use and modify)

