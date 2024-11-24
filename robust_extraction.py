import numpy as np
import librosa
import soundfile as sf
import os
import sys
from tqdm import tqdm
import warnings
from scipy import stats
import pandas as pd

def load_annotations(csv_file):
    """Load and process the annotations CSV file."""
    # Read CSV file with specified column names
    df = pd.read_csv(csv_file, header=None, 
                     names=['start_time', 'bar', 'beat', 'chord'])
    
    # Calculate end times by shifting start times
    df['end_time'] = df['start_time'].shift(-1)
    # For the last segment, we'll need to handle it separately
    df = df.ffill()
    
    return df

def apply_noise_reduction(segment, sr):
    """
    Apply comprehensive noise reduction techniques
    
    Parameters:
        segment: Audio time series
        sr: Sampling rate
    
    Returns:
        cleaned_segment: Noise-reduced audio segment
    """
    # 1. Pre-emphasis
    y_pre = librosa.effects.preemphasis(segment, coef=0.97)
    
    # 2. Spectral Gating
    S = librosa.stft(y_pre, n_fft=2048)
    mag = np.abs(S)
    phase = np.angle(S)
    
    # Estimate noise floor from the lowest 10% of magnitudes
    noise_floor = np.mean(np.sort(mag, axis=1)[:, :int(mag.shape[1]*0.1)], axis=1)
    noise_floor = noise_floor.reshape(-1, 1)
    
    # Apply soft thresholding
    gain = (mag - noise_floor) / mag
    gain = np.maximum(0, gain)
    gain = np.minimum(1, gain)
    
    # Apply gains
    mag_clean = mag * gain
    S_clean = mag_clean * np.exp(1j * phase)
    y_clean = librosa.istft(S_clean)
    
    return y_clean

def extract_robust_features(segment, sr, filename=""):
    """
    Extract noise-robust features from audio segment:
    - 12 PCP (median aggregated)
    - 6 Tonnetz features
    - 6 Spectral contrast features
    - 20 MFCCs
    - 1 Zero crossing rate
    
    Parameters:
        segment: Audio time series
        sr: Sampling rate
        filename: Original filename for reporting
    
    Returns:
        String: Formatted feature vector string (45 dimensions total)
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        # 1. Pre-emphasis to enhance high frequencies
        # y_pre = librosa.effects.preemphasis(segment, coef=0.97)
        #y_pre = segment
        # 2. Harmonic-Percussive Source Separation with enhanced parameters
        y_harmonic, y_percussive = librosa.effects.hpss(segment, n_fft=512)
        
        # 3. Chromagram from harmonic component (12 features)
        chromagram = librosa.feature.chroma_cqt(
            y=y_harmonic,
            sr=sr,
            n_chroma=12,     # Standard 12 pitch classes
            # n_octaves=6,     # 6 octaves range
            # fmin=librosa.note_to_hz('C1'),
            #threshold=0.05,   # Higher threshold to reduce noise
            #norm=2
        )
        # Use median for robustness to outliers
        chroma_features = np.median(chromagram, axis=1)
        
        # 4. Tonnetz features from harmonic component (6 features)
        # Represents harmonic relationships in pitch space
        tonnetz = librosa.feature.tonnetz(
            y=y_harmonic,
            sr=sr,
            chroma=chromagram  # Use previously computed chromagram
        )
        tonnetz_features = np.mean(tonnetz, axis=1)
        
        # 5. Spectral Contrast from harmonic component (6 features)
        # Compute magnitude spectrogram
        S = np.abs(librosa.stft(y_harmonic, n_fft=2048, hop_length=512))
        
        contrast = librosa.feature.spectral_contrast(
            S=S,
            sr=sr,
            n_bands=6,       # 6 frequency bands
            fmin=200.0,      # Focus on fundamental frequency range
            # quantile=0.02,   # More extreme for better contrast
            linear=False
        )
        contrast_features = np.mean(contrast, axis=1)
        
        # 6. MFCC from harmonic component (20 features)
        mfcc = librosa.feature.mfcc(
            y=y_harmonic,
            sr=sr,
            n_mfcc=20,       # Extract 20 MFCCs
            n_fft=2048,
            #hop_length=512,
            n_mels=128,      # Number of mel bands
            fmin=0,          # Minimum frequency
            fmax=sr/2        # Maximum frequency
        )
        mfcc_features = np.mean(mfcc, axis=1)
        
        # 7. Zero Crossing Rate from harmonic component (1 feature)
        zcr = librosa.feature.zero_crossing_rate(
            y=y_harmonic,
            frame_length=2048,
            hop_length=512
        )
        zcr_feature = np.mean(zcr)
        
        # 8. Combine all features
        all_features = np.concatenate([
            chroma_features,      # 12 PCP features
            tonnetz_features,     # 6 tonnetz features
            contrast_features,    # 6 spectral contrast features
            mfcc_features,        # 20 MFCC features
            [zcr_feature]         # 1 ZCR feature
        ])
        
        # Normalize features to handle different scales
        all_features = librosa.util.normalize(all_features)
        
        # Format features as string
        feature_str = ','.join([f"{x:.7f}" for x in all_features])
        return f"[{feature_str}]"
        
    except Exception as e:
        print(f"Error processing segment {filename}: {str(e)}")
        return None

def one_hot_encoder(chord: str) -> list[int]:
    """
    Convert chord label to one-hot encoded vector
    
    Parameters:
        chord: Chord label string
    
    Returns:
        list: One-hot encoded vector
    """
    chord_list = ['Cmaj', 'Cmin', 'C#maj', 'C#min', 'Dmaj', 'Dmin', 'D#maj', 'D#min', 
                  'Emaj', 'Emin', 'Fmaj', 'Fmin', 'F#maj', 'F#min', 'Gmaj', 'Gmin', 
                  'G#maj', 'G#min', 'Amaj', 'Amin', 'A#maj', 'A#min', 'Bmaj', 'Bmin']
    encoding = [0] * 24
    if chord in chord_list:
        encoding[chord_list.index(chord)] = 1
    else:
        raise ValueError(f"Chord '{chord}' not found in chord_list.")
    
    return encoding

def process_audio_and_save_features(audio_file_name, dataset_location, annotations_df, output_dir, logging_level=0):
    """
    Process audio file and save robust features
    
    Parameters:
        audio_file_name: Name of audio file with extension
        dataset_location: Path to dataset directory
        annotations_df: Pandas dataframe of annotations
        output_dir: Output directory for features
        logging_level: Logging detail level (0=minimal, 1=detailed)
    """
    audio_file_path = os.path.join(dataset_location, audio_file_name)
    try:
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_name}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Loading audio file: {audio_file_name}")
        warnings.filterwarnings("ignore", category=UserWarning)
        y, sr = librosa.load(audio_file_path, sr=None)
        print(f"Audio loaded with sampling rate: {sr} Hz")
        
        # Calculate samples per microsecond
        samples_per_microsecond = sr / 1_000_000
        print(f"Samples per microsecond: {samples_per_microsecond}")
        
        with open(f'{output_dir}/{audio_file_name}_features.csv', 'w') as file:
            # Process all rows except the last one
            for idx, row in tqdm(annotations_df.iloc[:-1].iterrows(), unit=" segments"):
                chord = row['chord'].strip().replace("'", "").replace('"', "")
                if chord == 'N.C.':  # Skip corrupted data
                    continue
                
                # Convert times to sample indices
                start_idx = int(np.floor(row['start_time'] * sr))
                end_idx = int(np.floor(row['end_time'] * sr))
                
                # Ensure we don't exceed array bounds
                end_idx = min(end_idx, len(y))
                segment = y[start_idx:end_idx]
                
                # Create filename for logging
                filename = f"{idx+1}_{chord}_{row['start_time']:.6f}_{row['end_time']:.6f}.mp3"
                
                # Extract robust features
                feature_vector = extract_robust_features(segment, sr, filename)
                if feature_vector is None:
                    continue
                
                # Get one-hot encoded chord
                one_hot_encoded_list = one_hot_encoder(chord)
                
                # Write to file
                file.write(f"{str(one_hot_encoded_list)},{feature_vector}\n")
                
                if logging_level == 1:
                    print(f"\nProcessed: {filename}")
                    print(f"Start time: {row['start_time']:.6f} seconds")
                    print(f"End time: {row['end_time']:.6f} seconds")
                    print(f"Samples: {len(segment)}")
        
        print(f"\nSuccessfully processed: {audio_file_name}")
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    # Example usage
    audio_file_name = '0001_mix.mp3'
    dataset_location = "./datasetmini/audio-mixes/"
    output_dir = 'robustfeatures'
    annotations_dir_loc = "./datasetmini/annotations/"
    annotations_file_name = "0001_beatinfo.csv"
    
    annotations_file = os.path.join(annotations_dir_loc, annotations_file_name)
    annotations_df = load_annotations(annotations_file)
    process_audio_and_save_features(audio_file_name, dataset_location, annotations_df, output_dir)