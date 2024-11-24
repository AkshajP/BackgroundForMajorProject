import tensorflow as tf
from sklearn.model_selection import train_test_split
from robust_model_maker import create_ffnn_model, train_model, evaluate_model, load_data_from_folder
import os

def train_and_save_models(X, y, splits=[(0.7, 0.3), (0.9, 0.1)], base_model_path="models"):
    """
    Train and save models with different train-test splits
    
    Parameters:
        X (numpy.ndarray): Input features
        y (numpy.ndarray): Target labels
        splits (list): List of tuples containing (train_size, test_size)
        base_model_path (str): Base directory to save models
    
    Returns:
        dict: Dictionary containing training histories and evaluation metrics for each split
    """
    os.makedirs(base_model_path, exist_ok=True)
    results = {}
    
    for train_size, test_size in splits:
        print(f"\nTraining model with {train_size*100:.0f}-{test_size*100:.0f} split")
        
        # Create train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=42
        )
        
        # Calculate appropriate batch size
        batch_size = min(32, int(len(X_train) ** 0.5))
        
        # Train model
        model, history = train_model(X_train, y_train, epochs=100, batch_size=batch_size)
        
        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # Save model
        model_name = f"robust_model_{int(train_size*100)}_{int(test_size*100)}_split.h5"
        model_path = os.path.join(base_model_path, model_name)
        model.save(model_path)
        
        # Store results
        results[f"{int(train_size*100)}-{int(test_size*100)}"] = {
            'history': history.history,
            'test_loss': loss,
            'test_accuracy': accuracy,
            'model_path': model_path
        }
        
        print(f"Model saved to: {model_path}")
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
    
    return results

if __name__ == "__main__":
    # Load your data
    data_folder = "extracted_robust_45_annotations"
    y, X = load_data_from_folder(data_folder)
    
    # Train and save models with both splits
    results = train_and_save_models(X, y)