
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

def preprocess_data(df):
    """Preprocess the diabetes dataset."""
    # Replace invalid zeros with NaN for specific columns
    columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[columns_with_zeros] = df[columns_with_zeros].replace(0, np.nan)
    
    # Fill NaN with median of each column
    for col in columns_with_zeros:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Split features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_model(X_train, y_train):
    """Train a logistic regression model."""
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    
    return y_pred

def save_model(model, scaler, model_filename="diabetes_model.joblib", scaler_filename="scaler.joblib"):
    """Save the trained model and scaler."""
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"Model saved as {model_filename}")
    print(f"Scaler saved as {scaler_filename}")

def load_model(model_filename="diabetes_model.joblib", scaler_filename="scaler.joblib"):
    """Load a previously trained model and scaler."""
    if os.path.exists(model_filename) and os.path.exists(scaler_filename):
        model = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)
        print("Loaded existing model and scaler.")
        return model, scaler
    return None, None

def get_user_input():
    """Get user input for a new patient's data."""
    print("\nEnter patient data for diabetes prediction:")
    try:
        data = {
            'Pregnancies': int(input("Number of pregnancies: ")),
            'Glucose': float(input("Glucose level (mg/dL): ")),
            'BloodPressure': float(input("Blood pressure (mm Hg): ")),
            'SkinThickness': float(input("Skin thickness (mm): ")),
            'Insulin': float(input("Insulin level (mu U/ml): ")),
            'BMI': float(input("BMI: ")),
            'DiabetesPedigreeFunction': float(input("Diabetes pedigree function: ")),
            'Age': int(input("Age: "))
        }
        return pd.DataFrame([data])
    except ValueError:
        print("Error: Please enter valid numerical values.")
        return None

def predict_new_patient(model, scaler, patient_data):
    """Predict diabetes for a new patient."""
    patient_data_scaled = scaler.transform(patient_data)
    prediction = model.predict(patient_data_scaled)
    probability = model.predict_proba(patient_data_scaled)[0]
    
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    print(f"\nPrediction: {result}")
    print(f"Probability of being diabetic: {probability[1]:.2%}")
    
    return prediction, probability

def main():
    """Main function to run the diabetes predictor."""
    print("Welcome to Diabetes Predictor!")
    
    # Check for existing model
    model, scaler = load_model()
    
    if model is None:
        # Load and preprocess data
        try:
            df = pd.read_csv('diabetes.csv')
        except FileNotFoundError:
            print("Error: diabetes.csv not found. Please ensure the dataset is in the same directory.")
            return
            
        X_scaled, y, scaler = preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        evaluate_model(model, X_test, y_test)
        
        # Save model and scaler
        save_model(model, scaler)
    
    # Interactive prediction loop
    while True:
        print("\nOptions:")
        print("1. Predict diabetes for a new patient")
        print("2. Exit")
        
        choice = input("Enter your choice (1-2): ").strip()
        if choice == "2":
            print("Goodbye!")
            break
        if choice != "1":
            print("Invalid choice! Please select 1 or 2.")
            continue
            
        # Get and predict for new patient
        patient_data = get_user_input()
        if patient_data is not None:
            prediction, probability = predict_new_patient(model, scaler, patient_data)
            
            # Save prediction to CSV
            patient_data['Prediction'] = prediction
            patient_data['Probability_Diabetic'] = probability[1]
            output_file = 'predictions.csv'
            patient_data.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
            print(f"Prediction saved to {output_file}")

if __name__ == "__main__":
    main()
