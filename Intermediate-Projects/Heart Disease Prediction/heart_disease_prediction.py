# Heart Disease Prediction
# Author: Amir Jafarnejad (https://github.com/Ajafarnezhad)
# Description: A professional implementation of heart disease prediction using machine learning
# and deep learning models, achieving 83.61% accuracy with a neural network.
# Dataset: heart.csv (303 records, 14 features)
# Dependencies: numpy, pandas, matplotlib, seaborn, scikit-learn, tensorflow

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style for professional visualizations
plt.style.use('seaborn')
sns.set(rc={'figure.figsize': (12, 6)})

def load_and_preprocess_data(file_path='heart.csv'):
    """
    Load and preprocess the heart disease dataset.
    
    Args:
        file_path (str): Path to the dataset CSV file.
        
    Returns:
        tuple: Preprocessed features (X_train, X_test, y_train, y_test) and scaler.
    """
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        
        # Verify data integrity
        if df.isnull().sum().sum() > 0:
            raise ValueError("Dataset contains missing values.")
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Standardize numerical features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        raise
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        raise

def perform_eda(df, output_dir='visualizations'):
    """
    Perform exploratory data analysis and save visualizations.
    
    Args:
        df (pd.DataFrame): Input dataset.
        output_dir (str): Directory to save visualization plots.
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Heart Disease Features')
    plt.savefig(f'{output_dir}/correlation_heatmap.png')
    plt.close()
    
    # Distribution of key features
    key_features = ['age', 'chol', 'thalach', 'oldpeak']
    for feature in key_features:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.savefig(f'{output_dir}/{feature}_distribution.png')
        plt.close()
    
    print(f"EDA visualizations saved in {output_dir}/")

def train_machine_learning_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate traditional machine learning models.
    
    Args:
        X_train, X_test: Training and testing features.
        y_train, y_test: Training and testing labels.
        
    Returns:
        dict: Dictionary of model names and their accuracy scores.
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        # Cross-validation for robustness
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"{name} - Test Accuracy: {accuracy:.4f}, CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return results

def build_and_train_neural_network(X_train, X_test, y_train, y_test, epochs=100, batch_size=32):
    """
    Build and train a neural network model.
    
    Args:
        X_train, X_test: Training and testing features.
        y_train, y_test: Training and testing labels.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        
    Returns:
        float: Test accuracy of the neural network.
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train model
    history = model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size,
        validation_data=(X_test, y_test), verbose=0
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Neural Network - Test Accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Neural Network Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('visualizations/nn_training_history.png')
    plt.close()
    
    return test_accuracy

def visualize_model_performance(results, nn_accuracy, output_dir='visualizations'):
    """
    Visualize model performance comparison.
    
    Args:
        results (dict): Dictionary of model names and their accuracy scores.
        nn_accuracy (float): Neural network test accuracy.
        output_dir (str): Directory to save the plot.
    """
    algorithms = list(results.keys()) + ['Neural Network']
    scores = list(results.values()) + [nn_accuracy]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=algorithms, y=scores, palette='viridis')
    plt.title('Model Performance Comparison', fontsize=14, pad=15)
    plt.xlabel('Algorithms', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.ylim(0, 1)
    for i, score in enumerate(scores):
        plt.text(i, score + 0.01, f'{score:.2%}', ha='center', fontsize=10)
    plt.savefig(f'{output_dir}/model_comparison.png')
    plt.close()
    print(f"Model comparison plot saved in {output_dir}/model_comparison.png")

def main():
    """
    Main function to execute the heart disease prediction pipeline.
    """
    print("Starting Heart Disease Prediction Pipeline...")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    
    # Perform EDA
    perform_eda(pd.read_csv('heart.csv'))
    
    # Train machine learning models
    ml_results = train_machine_learning_models(X_train, X_test, y_train, y_test)
    
    # Train neural network
    nn_accuracy = build_and_train_neural_network(X_train, X_test, y_train, y_test)
    
    # Visualize model performance
    visualize_model_performance(ml_results, nn_accuracy)
    
    print("\nPipeline completed successfully!")
    print(f"Final Neural Network Accuracy: {nn_accuracy:.2%}")

if __name__ == "__main__":
    main()