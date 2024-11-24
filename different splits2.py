import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, LSTM, Conv1D, MaxPooling1D, BatchNormalization,
    Dropout, Bidirectional, Input, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def parse_file(content):
    """Parse the content of a data file into features and labels."""
    x_data = []
    y_data = []
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    for line in lines:
        # Remove whitespace and split by '],['
        arrays = line.replace(' ', '').strip('[]').split('],[')
        
        if len(arrays) == 2:
            try:
                # First array is the one-hot encoded chord (24 values)
                y_array = [int(x) for x in arrays[0].split(',')]
                # Second array is the feature vector (45 values)
                x_array = [float(x) for x in arrays[1].split(',')]
                
                if len(y_array) == 24 and len(x_array) == 46:
                    x_data.append(x_array)
                    y_data.append(y_array)
            except (ValueError, IndexError):
                print(f"Skipping malformed line: {line}")
                continue
    
    return x_data, y_data

def load_data_from_folder(folder):
    """Load all data files from the specified folder."""
    all_x = []
    all_y = []
    
    data_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    print(f"Found {len(data_files)} data files")
    
    for data_file in data_files:
        with open(os.path.join(folder, data_file), 'r') as file:
            content = file.read()
            x, y = parse_file(content)
            all_x.extend(x)
            all_y.extend(y)
    
    X = np.array(all_x, dtype=np.float32)
    Y = np.array(all_y, dtype=np.float32)
    
    print(f"Loaded {len(X)} samples")
    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {Y.shape}")
    
    return X, Y

def create_cnn_lstm_model(input_shape):
    """Create the CNN-LSTM model."""
    model = Sequential([
        # Reshape layer to add time steps dimension
        Input(shape=input_shape),
        Reshape((1, input_shape[0])),
        
        # First CNN block
        Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=1),  # Keep time dimension since we only have 1
        Dropout(0.25),
        
        # Second CNN block
        Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=1),
        Dropout(0.25),
        
        # Bidirectional LSTM layers
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(LSTM(64, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        # Output layer
        Dense(24, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

def prepare_callbacks():
    """Prepare training callbacks."""
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

def train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """Train the model with the provided data."""
    model = create_cnn_lstm_model(input_shape=(X_train.shape[1],))
    callbacks = prepare_callbacks()
    
    print("\nModel Summary:")
    model.summary()
    
    print("\nStarting training...")
    history = model.fit(
        X_train, 
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"AUC: {test_auc:.4f}")

def main():
    # Configuration
    data_folder = "extracted_robust_45_annotations"  # Folder containing your data files
    test_size = 0.2
    validation_size = 0.2
    epochs = 100
    batch_size = 32
    
    # Load and preprocess data
    print("Loading data...")
    X, y = load_data_from_folder(data_folder)
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into train+val and test sets
    print("\nSplitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )
    
    # Split remaining data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=validation_size, random_state=42
    )
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Train model
    model, history = train_model(X_train, y_train, X_val, y_val, epochs, batch_size)
    
    # Save model
    print("\nSaving model...")
    model.save("cnn_lstm_chord_model.h5")
    print("Model saved as 'cnn_lstm_chord_model.h5'")   

    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    

if __name__ == "__main__":
    main()