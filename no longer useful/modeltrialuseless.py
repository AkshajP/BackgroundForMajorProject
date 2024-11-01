import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import csv
import os

def load_data_from_folder(folder_path):
    """
    Load and parse all CSV files in the specified folder and its subfolders
    
    Parameters:
    folder_path: path to the folder containing CSV files
    
    Returns:
    X: numpy array of inputs (n_samples, 12)
    y: numpy array of outputs (n_samples, 24)
    """
    X = []
    y = []
    
    # Walk through all files in the folder and subfolders
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                
                try:
                    with open(file_path, 'r') as file:
                        csv_reader = csv.reader(file)
                        for row in csv_reader:
                            # Convert string representations of arrays to actual arrays
                            output_array = eval(row[0])  # First element is output
                            input_array = eval(row[1])   # Second element is input
                            
                            y.append(output_array)
                            X.append(input_array)
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")
                    continue
    
    if not X or not y:
        raise ValueError("No valid data found in the specified folder")
    
    return np.array(X), np.array(y)

def create_ffnn_model():
    """
    Creates a Feed-Forward Neural Network model for predicting 
    24 float values from 12 input values
    """
    model = Sequential([
        # Input layer
        Dense(24, input_shape=(12,), activation='relu'),
        BatchNormalization(),
        
        # Hidden layers
        Dense(48, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(48, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output layer
        Dense(24, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X_train, y_train, epochs=100, batch_size=32):
    """
    Train the model with the provided data
    
    Parameters:
    X_train: numpy array of shape (n_samples, 12)
    y_train: numpy array of shape (n_samples, 24)
    """
    model = create_ffnn_model()
    
    # Add early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, 
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Make some predictions
    predictions = model.predict(X_test[:5])  # First 5 samples
    print("\nSample Predictions vs Actual Values:")
    for i in range(5):
        print(f"\nSample {i+1}:")
        print("Prediction:", predictions[i].round(3))
        print("Actual:", y_test[i])

if __name__ == "__main__":
    # Specify your data folder path
    data_folder = "pcpvectors"  # Replace with your folder path
    
    try:
        # Load and prepare the data
        print("Loading data from folder...")
        X, y = load_data_from_folder(data_folder)
        
        print(f"\nLoaded {len(X)} samples from CSV files")
        print(f"Input shape: {X.shape}")
        print(f"Output shape: {y.shape}")
        
        # Calculate appropriate batch size (rule of thumb: sqrt of dataset size)
        batch_size = min(32, int(np.sqrt(len(X))))
        
        # Train the model
        print("\nTraining model...")
        model, history = train_model(X, y, epochs=100, batch_size=batch_size)
        
        # Evaluate the model
        print("\nEvaluating model...")
        evaluate_model(model, X, y)
        
        # Optionally save the model
        save_path = "trained_model"
        model.save(save_path)
        print(f"\nModel saved to {save_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")