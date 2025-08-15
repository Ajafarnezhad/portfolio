# Text-to-Video Generator: Bring Stories to Life with AI 🎥✨

Welcome to the **Text-to-Video Generator**, an intermediate Python project that transforms your text prompts (in Persian or any language) into captivating videos using a Telegram bot powered by the VideoCrafter model via the Replicate API. With seamless translation, real-time status updates, and a professional CLI interface, this project is a standout portfolio piece for showcasing your expertise in AI-driven video synthesis, bot development, and real-time applications.

---

## 🌟 Project Highlights
This project combines cutting-edge AI video generation with user-friendly Telegram integration, enabling users to create dynamic videos from text descriptions. Featuring robust error handling, detailed logging, and configurable settings, it’s ideal for demonstrating skills in AI, natural language processing, and interactive systems.

---

## 🚀 Features
- **Text Translation**: Automatically translates input text (e.g., Persian) to English using `GoogleTranslator` for compatibility with VideoCrafter.
- **Video Generation**: Leverages VideoCrafter via the Replicate API to create high-quality videos with customizable parameters like FPS and steps.
- **Telegram Integration**: Responds to `/start`, processes text prompts, and provides real-time status updates during video generation.
- **CLI Interface**: Configure bot token, model settings, and other parameters via intuitive command-line arguments.
- **Real-Time Updates**: Keeps users informed about text reception, processing progress, and video delivery.
- **Error Handling & Logging**: Robust checks and detailed logs ensure reliable operation and easy debugging.
- **Model Persistence**: Optional support for saving/loading configuration settings for streamlined workflows.

---

## 🛠️ Requirements
- **Python**: 3.8 or higher
- **Libraries**:
  - `python-telegram-bot`
  - `deep-translator`
  - `requests`
  - `replicate`

Install dependencies with:
```bash
pip install python-telegram-bot deep-translator requests replicate
```

---

## 🎮 How to Run

### 1. Set Up the Bot
1. Obtain a Telegram bot token from [BotFather](https://t.me/BotFather).
2. Set up a Replicate API token at [Replicate](https://replicate.com).
3. Store tokens securely (e.g., as environment variables or in a config file).

### 2. Run the Script
Launch the Telegram bot with default settings:
```bash
python text_to_video_generator.py --bot_token YOUR_TELEGRAM_BOT_TOKEN --replicate_token YOUR_REPLICATE_API_TOKEN
```

### 3. Interact with the Bot
- Send `/start` to the bot on Telegram to begin.
- Input a text prompt (e.g., "A vibrant Persian festival under starry skies") to generate a video.
- Receive real-time updates and the final video delivered directly in Telegram.

### 4. Customize Your Workflow
- `--bot_token`: Your Telegram bot token.
- `--replicate_token`: Your Replicate API token.
- `--fps`: Frames per second for the video (e.g., `30`).
- `--steps`: Number of generation steps (e.g., `50`).

---

## 📈 Example Interaction
- **User Input**:
  ```
  /start
  A serene mountain landscape with a flowing river
  ```
- **Bot Response**:
  ```
  Received your prompt! Translating...
  Processing video with VideoCrafter...
  Video generated! Delivering now...
  [Video delivered: A flowing river through a serene mountain landscape]
  ```
- **Logs**:
  ```
  INFO: Received prompt: A serene mountain landscape with a flowing river
  INFO: Translated to English: A serene mountain landscape with a flowing river
  INFO: Video generated and sent successfully
  ```

---

## 🔮 Future Enhancements
Elevate this project with these exciting ideas:
- **Multi-Model Support**: Integrate additional video generation models like RunwayML or DALL·E Video.
- **Advanced Customization**: Add parameters for video length, resolution, or style variations.
- **Web App Deployment**: Create a Flask or Streamlit interface for non-Telegram users.
- **Multilingual Enhancements**: Improve translation accuracy for complex or idiomatic phrases.
- **Unit Testing**: Implement `pytest` for robust validation of translation and video generation pipelines.

---

## 📜 License
This project is licensed under the **MIT License**—use, modify, and share it freely!

Turn your words into mesmerizing videos with the **Text-to-Video Generator** and unleash your creativity with AI! 🚀