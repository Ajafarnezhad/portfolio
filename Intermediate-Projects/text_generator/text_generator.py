import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import argparse
import logging

# Configure logging for transparency and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(seq_length=100, buffer_size=10000):
    """
    Load and preprocess the Tiny Shakespeare dataset into sequences for training.
    
    Args:
        seq_length (int): Length of each input sequence.
        buffer_size (int): Buffer size for shuffling the dataset.
    
    Returns:
        dataset: Preprocessed TensorFlow dataset.
        vocab: Sorted list of unique characters.
        char2idx: Dictionary mapping characters to indices.
        idx2char: Array mapping indices to characters.
        text_as_int: Text converted to integer indices.
    """
    try:
        # Load dataset
        dataset, _ = tfds.load('tiny_shakespeare', with_info=True, as_supervised=False)
        text = next(iter(dataset['train']))['text'].numpy().decode('utf-8')
        logging.info(f"Loaded Tiny Shakespeare dataset ({len(text)} characters)")

        # Create vocabulary and mappings
        vocab = sorted(set(text))
        char2idx = {char: idx for idx, char in enumerate(vocab)}
        idx2char = np.array(vocab)
        logging.info(f"Vocabulary size: {len(vocab)}")

        # Convert text to integers
        text_as_int = np.array([char2idx[c] for c in text])

        # Create sequences
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

        def split_input_target(chunk):
            """Split sequence into input and target text."""
            return chunk[:-1], chunk[1:]

        dataset = sequences.map(split_input_target)
        dataset = (
            dataset
            .shuffle(buffer_size)
            .batch(args.batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        return dataset, vocab, char2idx, idx2char, text_as_int
    except Exception as e:
        logging.error(f"Error in data preprocessing: {str(e)}")
        raise

def build_model(vocab_size, embedding_dim=256, rnn_units=1024, batch_size=64):
    """
    Build an LSTM-based model for text generation.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embedding layer.
        rnn_units (int): Number of LSTM units.
        batch_size (int): Batch size for training or inference.
    
    Returns:
        model: Compiled Keras model.
    """
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
            tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])
        logging.info(f"Model built with {embedding_dim} embedding dimensions and {rnn_units} LSTM units")
        return model
    except Exception as e:
        logging.error(f"Error building model: {str(e)}")
        raise

def generate_text(model, start_string, char2idx, idx2char, num_generate=1000):
    """
    Generate text using the trained model.
    
    Args:
        model: Trained Keras model.
        start_string (str): Seed text to start generation.
        char2idx (dict): Mapping of characters to indices.
        idx2char (array): Mapping of indices to characters.
        num_generate (int): Number of characters to generate.
    
    Returns:
        str: Generated text starting with the seed string.
    """
    try:
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []
        model.reset_states()

        for _ in range(num_generate):
            predictions = model(input_eval)
            predictions = tf.squeeze(predictions, 0)
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(idx2char[predicted_id])

        return start_string + ''.join(text_generated)
    except Exception as e:
        logging.error(f"Error generating text: {str(e)}")
        raise

def main(args):
    """
    Main function to handle CLI operations for training or text generation.
    
    Args:
        args: Parsed command-line arguments.
    """
    try:
        # Load and preprocess data
        dataset, vocab, char2idx, idx2char, _ = load_and_preprocess_data(args.seq_length, args.buffer_size)

        if args.mode == 'train':
            # Build and compile model
            model = build_model(len(vocab), args.embedding_dim, args.rnn_units, args.batch_size)
            model.compile(optimizer='adam', loss=lambda labels, logits: tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
            logging.info("Model compiled successfully")

            # Set up checkpointing
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            checkpoint_prefix = os.path.join(args.checkpoint_dir, "ckpt_{epoch}")
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_prefix,
                save_weights_only=True
            )

            # Train model
            model.fit(dataset, epochs=args.epochs, callbacks=[checkpoint_callback])
            logging.info("Training completed successfully")

        elif args.mode == 'generate':
            # Build model for generation (batch size = 1)
            model = build_model(len(vocab), args.embedding_dim, args.rnn_units, batch_size=1)
            model.load_weights(tf.train.latest_checkpoint(args.checkpoint_dir))
            model.build(tf.TensorShape([1, None]))
            logging.info("Model loaded for text generation")

            # Generate and display text
            generated_text = generate_text(model, args.seed, char2idx, idx2char, args.num_generate)
            print(f"🎉 Generated Text:\n{generated_text}")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Shakespearean Text Generator: Craft poetic text with deep learning")
    parser.add_argument('--mode', choices=['train', 'generate'], default='train', help="Mode: train or generate text")
    parser.add_argument('--data_path', default=None, help="Path to dataset (auto-downloaded if None)")
    parser.add_argument('--checkpoint_dir', default='./training_checkpoints', help="Directory for model checkpoints")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--seq_length', type=int, default=100, help="Sequence length for training")
    parser.add_argument('--buffer_size', type=int, default=10000, help="Buffer size for shuffling")
    parser.add_argument('--embedding_dim', type=int, default=256, help="Embedding dimension")
    parser.add_argument('--rnn_units', type=int, default=1024, help="Number of RNN units")
    parser.add_argument('--seed', default="QUEEN: So, lets end this", help="Seed text for generation")
    parser.add_argument('--num_generate', type=int, default=1000, help="Number of characters to generate")
    args = parser.parse_args()

    main(args)