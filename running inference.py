import tensorflow as tf
import librosa
import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, InputLayer
from model_maker import create_ffnn_model
from pcp_module import pcp_vectorise_segment
from itertools import groupby
from operator import itemgetter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs except errors
tf.get_logger().setLevel('ERROR') 

def load_trained_model(model_path):
    """
    Load a pre-trained neural network model for chord prediction.

    Attempts to load a complete model first (architecture + weights + optimizer state).
    If that fails, creates a new model and loads just the weights.
    Finally recompiles the model for inference only.

    Args:
        model_path (str): Path to the saved model file (.h5 format)

    Returns:
        tensorflow.keras.Model: Loaded and compiled model ready for inference

    Raises:
        OSError: If the model file cannot be found
        ValueError: If the model architecture is incompatible with saved weights
    """
    try:
        model = load_model(model_path)
    except:
        # If loading the model fails, create new model and load just the weights
        model = create_ffnn_model()
        # Load weights without optimizer state
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
    
    # Recompile the model for inference only (no training needed)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    return model
   

def predict_chord(pcp_vector, model):
    """
    Convert a Pitch Class Profile vector into a predicted chord using the trained model.

    Args:
        pcp_vector (numpy.ndarray): A 12-dimensional PCP vector representing 
            the harmonic content of an audio segment
        model (tensorflow.keras.Model): Trained chord prediction model

    Returns:
        str: Predicted chord label (e.g., 'Cmaj', 'Amin', etc.)

    Note:
        The PCP vector should be normalized and extracted using standard parameters
        to match the training data characteristics.
    """
    chord_list = ['Cmaj', 'Cmin', 'C#maj', 'C#min', 'Dmaj', 'Dmin', 'D#maj', 'D#min', 
                  'Emaj', 'Emin', 'Fmaj', 'Fmin', 'F#maj', 'F#min', 'Gmaj', 'Gmin', 
                  'G#maj', 'G#min', 'Amaj', 'Amin', 'A#maj', 'A#min', 'Bmaj', 'Bmin']
    
    pcp_vector = np.array(pcp_vector).reshape(1, -1)
    prediction = model.predict(pcp_vector, verbose=0)
    chord_index = np.argmax(prediction)
    
    return chord_list[chord_index]

def apply_audio_filters(audio_data, sr):
    """
    Apply preprocessing filters to clean the audio signal for improved chord detection.

    Performs:
    1. Preemphasis filtering to boost high frequencies
    2. Harmonic-Percussive Source Separation (HPSS) to isolate harmonic content

    Args:
        audio_data (numpy.ndarray): Raw audio signal
        sr (int): Sampling rate of the audio in Hz

    Returns:
        numpy.ndarray: Filtered audio signal optimized for chord detection

    Note:
        The filtered signal maintains the same sampling rate and length as the input.
    """
    # Apply a bandpass filter (keeping frequencies between 50Hz and 2000Hz)
    y_filtered = librosa.effects.preemphasis(audio_data)

    y_harmonic, _ = librosa.effects.hpss(y_filtered)
    
    return y_harmonic

def group_consecutive_chords(times, chords):
    """
    Group consecutive segments with the same chord prediction into longer segments.

    Args:
        times (numpy.ndarray): Array of timestamps marking segment boundaries in seconds
        chords (list): List of chord predictions corresponding to the segments 
            between consecutive timestamps

    Returns:
        list: List of tuples (start_time, end_time, chord) where consecutive 
            segments with the same chord have been merged

    Example:
        >>> times = [0, 0.5, 1.0, 1.5]
        >>> chords = ['Cmaj', 'Cmaj', 'Amin']
        >>> group_consecutive_chords(times, chords)
        [(0, 1.0, 'Cmaj'), (1.0, 1.5, 'Amin')]
    """
    grouped_segments = []
    
    # Create pairs of (time, chord)
    chord_segments = list(zip(times[:-1], times[1:], chords))
    
    for chord, group in groupby(chord_segments, key=lambda x: x[2]):
        group_list = list(group)
        start_time = group_list[0][0]
        end_time = group_list[-1][1]
        grouped_segments.append((start_time, end_time, chord))
    
    return grouped_segments

def infer_chords(audio_file, model_weights_path):
    """
    Process an audio file to detect and predict its chord progression.

    Workflow:
    1. Loads and preprocesses the audio file
    2. Detects beats and generates half-beat segments
    3. Applies audio filtering to each segment
    4. Extracts PCP vectors and predicts chords
    5. Groups consecutive identical chord predictions

    Args:
        audio_file (str): Path to the input audio file
        model_weights_path (str): Path to the saved model weights

    Returns:
        list: List of tuples (start_time, end_time, chord) representing 
            the detected chord progression

    Raises:
        FileNotFoundError: If audio_file or model_weights_path doesn't exist
        RuntimeError: If beat detection or chord prediction fails
    """
    print(f"Loading audio file: {audio_file}")
    y, sr = librosa.load(audio_file)
    
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    print(f"Detected tempo: {tempo} BPM")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Generate half-beat times by interpolating between beats
    half_beat_times = []
    for i in range(len(beat_times) - 1):
        start_time = beat_times[i]
        end_time = beat_times[i + 1]
        mid_time = start_time + (end_time - start_time) / 2
        half_beat_times.extend([start_time, mid_time])
    # Add the last beat time
    half_beat_times.append(beat_times[-1])
    times = np.array(half_beat_times)

    model = load_trained_model(model_weights_path)
    
    # Process each segment
    predictions = []
    for i in range(len(times) - 1):
        start_time = times[i]
        end_time = times[i + 1]
        start_idx = int(start_time * sr)
        end_idx = int(end_time * sr)
        
        segment = y[start_idx:end_idx]
        
        filtered_segment = apply_audio_filters(segment, sr)
        
        pcp_vector_str = pcp_vectorise_segment(filtered_segment, sr, f"segment_{start_time}")
        pcp_vector = [float(x) for x in pcp_vector_str.strip('[]').split(',')]
        
        chord = predict_chord(pcp_vector, model)
        predictions.append(chord)
    
    grouped_segments = group_consecutive_chords(times, predictions)
    
    return grouped_segments

def format_time(seconds: float):
    """Format time in seconds to MM:SS.mmm"""
    minutes = int(seconds // 60)
    seconds_remainder = seconds % 60
    return f"{minutes:02d}:{seconds_remainder:06.3f}"

if __name__ == "__main__":
    audio_file = "0001_infer.mp3"
    model_weights_path = "model.h5"
    
    # Run inference
    chord_segments = infer_chords(audio_file, model_weights_path)
    
    # Display results
    print("\nPredicted Chord Progression:")
    print("-----------------------------")
    for start_time, end_time, chord in chord_segments:
        print(f"{format_time(start_time)} - {format_time(end_time)}: {chord}")
