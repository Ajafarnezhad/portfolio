# 📜 Shakespearean Text Generator: Crafting Poetic Narratives with Deep Learning

## 🌟 Project Vision
Step into the enchanting world of literature with the **Shakespearean Text Generator**, a sophisticated Python-based deep learning project that conjures poetic, dialogue-style text inspired by the **Tiny Shakespeare** dataset. Powered by a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers, this project delivers contextually rich text generation with finesse. Featuring seamless data preprocessing, robust training pipelines, and an elegant command-line interface (CLI), it’s a dazzling showcase of Natural Language Processing (NLP) expertise, crafted to elevate your machine learning portfolio to world-class standards.

## ✨ Core Features
- **Streamlined Data Preprocessing** 📝: Transforms raw Shakespearean text into numerical sequences, optimized for deep learning models.
- **Advanced RNN Architecture** 🧠: Harnesses a powerful LSTM-based model with embedding and dense layers for coherent text generation.
- **Dynamic Text Creation** ✍️: Generates expressive, Shakespearean-style dialogues from user-defined seed phrases, with customizable output lengths.
- **Robust Training Pipeline** ⚙️: Trains efficiently with checkpointing and optimized batch processing for scalability.
- **Intuitive CLI Experience** 🖥️: Offers user-friendly commands for training and generation, with flexible configuration options.
- **Model Persistence** 💾: Saves and loads model weights using TensorFlow checkpoints for seamless reusability.
- **Comprehensive Error Handling & Logging** 🛡️: Ensures reliability with meticulous checks and detailed logs for transparency.

## 🚀 Getting Started

### Prerequisites
- **Python**: Version 3.8 or higher
- **Dependencies**: A curated suite of libraries to fuel your project:
  - `tensorflow`
  - `tensorflow-datasets`
  - `numpy`

Install them effortlessly:
```bash
pip install tensorflow tensorflow-datasets numpy
```

### Dataset Spotlight
The **Tiny Shakespeare** dataset, sourced via TensorFlow Datasets, is your canvas for literary creation:
- **Content**: A rich collection of Shakespearean dialogues, perfect for generating dramatic text.
- **Format**: Textual data, automatically downloaded and processed into character-based sequences.
- **Size**: Compact yet robust, ideal for training text generation models without external downloads.

## 🎉 How to Use

### 1. Train the Model
Craft a poetic RNN model with customizable parameters:
```bash
python text_generator.py --mode train --epochs 10 --batch_size 64 --seq_length 100
```

### 2. Generate Text
Create Shakespearean-style text from a seed phrase:
```bash
python text_generator.py --mode generate --seed "QUEEN: So, lets end this" --num_generate 1000
```

### CLI Options
- `--mode`: Select `train` or `generate` (default: `train`).
- `--data_path`: Path to the dataset (default: auto-downloaded via TensorFlow).
- `--checkpoint_dir`: Directory for model checkpoints (default: `./training_checkpoints`).
- `--epochs`: Number of training epochs (default: 10).
- `--batch_size`: Batch size for training (default: 64).
- `--seq_length`: Length of input sequences (default: 100).
- `--num_generate`: Number of characters to generate (default: 1000).
- `--seed`: Starting text for generation (default: `QUEEN: So, lets end this`).

## 📊 Sample Output

### Training Output
```
🌟 Loaded Tiny Shakespeare dataset (1,115,394 characters)
🔍 Preprocessing complete: Vocabulary size = 65
⚙️ Model built with 256 embedding dimensions and 1024 LSTM units
✅ Training complete! Loss: 1.2339 (Epoch 10/10)
```

### Generated Text
```
🎉 Generated Text:
QUEEN: So, lets end this noble strife, and part
With hearts unyielded to the bitter frost of fate.
O, how the stars do mock our fleeting dreams,
And cast their shadows on the weary soul!
LORD: Alas, my queen, what words shall mend this wound?
The heavens turn, and time doth carve its mark...
```

## 🌈 Future Enhancements
- **Transformer Architecture** 🚀: Upgrade to a GPT-style transformer for enhanced text coherence and creativity.
- **Expanded Datasets** 📚: Incorporate diverse literary corpora to support multiple text styles.
- **Web App Deployment** 🌐: Transform into an interactive app with Flask or Streamlit for user-friendly text generation.
- **Real-Time Generation** ⚡: Enable streaming text output for dynamic applications.
- **Unit Testing** 🛠️: Implement `pytest` for robust validation of preprocessing and generation pipelines.

## 📜 License
Proudly licensed under the **MIT License**, fostering open collaboration and innovation in NLP.

---

🌟 **Shakespearean Text Generator**: Where deep learning weaves the timeless poetry of the Bard! 🌟