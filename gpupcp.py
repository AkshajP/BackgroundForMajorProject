import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm
import warnings

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"GPU devices available: {len(physical_devices)}")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No GPU devices found. Running on CPU.")

@tf.function
def tf_hpss(stft_matrix):
    """GPU-accelerated Harmonic-Percussive Source Separation"""
    H = tf.identity(stft_matrix)
    P = tf.identity(stft_matrix)
    
    # Convert to magnitude spectrograms
    H_mag = tf.abs(H)
    P_mag = tf.abs(P)
    
    # Median filtering
    for _ in range(3):  # Number of iterations
        H_mag = tf.nn.avg_pool1d(
            tf.expand_dims(H_mag, 0),
            ksize=3,
            strides=1,
            padding='SAME'
        )[0]
        P_mag = tf.nn.avg_pool1d(
            tf.expand_dims(tf.transpose(P_mag), 0),
            ksize=3,
            strides=1,
            padding='SAME'
        )[0]
        P_mag = tf.transpose(P_mag)
    
    # Mask creation
    mask_H = H_mag > P_mag
    mask_P = tf.logical_not(mask_H)
    
    # Apply masks
    H_out = tf.where(mask_H, stft_matrix, tf.zeros_like(stft_matrix))
    P_out = tf.where(mask_P, stft_matrix, tf.zeros_like(stft_matrix))
    
    return H_out, P_out

@tf.function
def tf_pcp_vectorise_segment(segment):
    """
    GPU-accelerated PCP vector calculation using TensorFlow operations
    """
    # Convert to float32 tensor
    segment = tf.cast(segment, tf.float32)
    
    # Compute STFT
    n_fft = 512
    hop_length = n_fft // 4
    window = tf.signal.hann_window(n_fft)
    
    stft = tf.signal.stft(
        segment,
        frame_length=n_fft,
        frame_step=hop_length,
        window_fn=lambda x: window,
        pad_end=True
    )
    
    # Harmonic-percussive separation
    H_stft, _ = tf_hpss(stft)
    
    # Compute magnitude spectrum
    magnitude = tf.abs(H_stft)
    
    # Create frequency bins
    sr = 22050  # Default librosa sampling rate
    frequencies = tf.linspace(0.0, sr/2.0, tf.shape(magnitude)[1])
    
    # Create pitch classes (12 semitones)
    pitch_classes = tf.range(12, dtype=tf.float32)
    
    # Calculate reference frequency (C4 = MIDI note 60)
    ref_freq = 440.0 * tf.pow(2.0, (pitch_classes - 69) / 12.0)
    
    # Create chromagram
    chroma = tf.zeros([12, tf.shape(magnitude)[0]], dtype=tf.float32)
    
    for i in range(12):
        # Calculate frequency range for this pitch class
        center_freq = ref_freq[i]
        freq_mask = tf.logical_and(
            frequencies >= center_freq / tf.sqrt(2.0),
            frequencies < center_freq * tf.sqrt(2.0)
        )
        freq_mask = tf.cast(freq_mask, tf.float32)
        
        # Sum the magnitudes for this pitch class
        chroma = tf.tensor_scatter_nd_add(
            chroma,
            [[i, 0]],
            [tf.reduce_sum(magnitude * tf.expand_dims(freq_mask, 0), axis=1)]
        )
    
    # Normalize
    chroma_sum = tf.reduce_sum(chroma, axis=0, keepdims=True)
    chroma_normalized = tf.where(
        chroma_sum > 0,
        chroma / chroma_sum,
        chroma
    )
    
    # Average across time
    pcp_vector = tf.reduce_mean(chroma_normalized, axis=1)
    
    return pcp_vector

def process_audio_and_save_pcp(audio_file_name, dataset_location, annotations_df, output_dir):
    """GPU-accelerated audio processing and PCP vector calculation"""
    try:
        if not os.path.exists(os.path.join(dataset_location, audio_file_name)):
            raise FileNotFoundError(f"Audio file not found: {audio_file_name}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio file
        y, sr = librosa.load(os.path.join(dataset_location, audio_file_name), sr=None)
        # Convert to tensor
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        
        with open(f'./{output_dir}/{audio_file_name}_pcpvectors.csv', 'w') as file:
            for idx, row in tqdm(annotations_df.iloc[:-1].iterrows(), unit=" segments"):
                chord = row['chord'].strip().replace("'", "").replace('"', "")
                if chord == 'N.C.':
                    continue
                
                # Calculate segment indices
                start_idx = int(tf.floor(row['start_time'] * sr))
                end_idx = int(tf.floor(row['end_time'] * sr))
                end_idx = tf.minimum(end_idx, tf.shape(y)[0])
                
                # Extract segment
                segment = y[start_idx:end_idx]
                
                # Calculate PCP vector using GPU
                with tf.device('/GPU:0'):
                    pcp_vector = tf_pcp_vectorise_segment(segment)
                    pcp_vector = pcp_vector.numpy()
                
                try:
                    one_hot_encoded_list = one_hot_encoder(chord)
                    file.write(f"{one_hot_encoded_list},{list(pcp_vector)}\n")
                except ValueError as e:
                    print(f"Error encoding chord {chord}: {e}")
                    continue

    except Exception as e:
        print(f"Error processing {audio_file_name}: {e}")
        raise

def one_hot_encoder(chord: str) -> list[int]:
    """Kept as is from original code"""
    chord_list = ['Cmaj', 'Cmin', 'C#maj', 'C#min', 'Dmaj', 'Dmin', 'D#maj', 'D#min', 
              'Emaj', 'Emin', 'Fmaj', 'Fmin', 'F#maj', 'F#min', 'Gmaj', 'Gmin', 
              'G#maj', 'G#min', 'Amaj', 'Amin', 'A#maj', 'A#min', 'Bmaj', 'Bmin']
    encoding = [0] * 24
    if chord in chord_list:
        encoding[chord_list.index(chord)] = 1
    else:
        raise ValueError(f"Chord '{chord}' not found in chord_list.")
    
    return encoding