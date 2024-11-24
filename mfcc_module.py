import pandas as pd
import soundfile as sf
import librosa
import numpy as np
import os
import sys
from tqdm import tqdm
import warnings 


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


def one_hot_encoder(chord: str) -> list[int]:
    chord_list = ['Cmaj', 'Cmin', 'C#maj', 'C#min', 'Dmaj', 'Dmin', 'D#maj', 'D#min', 
              'Emaj', 'Emin', 'Fmaj', 'Fmin', 'F#maj', 'F#min', 'Gmaj', 'Gmin', 
              'G#maj', 'G#min', 'Amaj', 'Amin', 'A#maj', 'A#min', 'Bmaj', 'Bmin']
    encoding = [0] * 24
    if chord in chord_list:
        encoding[chord_list.index(chord)] = 1
    else:
        raise ValueError(f"Chord '{chord}' not found in chord_list.")
    
    return encoding

def mfcc_vectorise_segment(segment, sr, filename):
    """
    Process audio segment to extract MFCC features from harmonic content.
    
    Parameters:
        segment: Audio time series
        sr: Sampling rate
        filename: Original filename for reporting
    
    Returns:
        String: Vector string in format "[val1,val2,...]" or None if error
    """
    n_fft = 2048
    hop_length = 512
    warnings.filterwarnings("ignore", category=UserWarning, module="librosa.core.spectrum")
    
    try:
        # Check for valid input
        if len(segment) == 0:
            print(f"Warning: Empty segment for {filename}")
            return None
            
        # Apply HPSS to isolate harmonic content
        y_harmonic, y_percussive = librosa.effects.hpss(
            segment,
            margin=3.0
        )
        
        # Compute MFCCs from harmonic signal
        mfccs = librosa.feature.mfcc(
            y=y_harmonic,
            sr=sr,
            n_mfcc=30,
            n_fft=n_fft,
            hop_length=hop_length,
            window='hann',
            center=True,
            power=2.0
        )
        
        # Handle empty or invalid MFCC computation
        if mfccs.size == 0:
            print(f"Warning: Empty MFCC computation for {filename}")
            return None
            
        # Take mean over time axis and handle single-frame case
        if mfccs.shape[1] == 1:
            mfcc_reduced = mfccs.flatten()
        else:
            mfcc_reduced = np.mean(mfccs, axis=1)
        
        # Ensure we have exactly 30 features
        if len(mfcc_reduced) != 30:
            print(f"Warning: Incorrect MFCC dimensions for {filename}: got {len(mfcc_reduced)}")
            return None
            
        # Convert to list of floats and format
        mfcc_list = [float(x) for x in mfcc_reduced]
        vector_str = ','.join([f"{x:.6f}" for x in mfcc_list])
        
        return f"[{vector_str}]"
        
    except Exception as e:
        print(f"Error processing segment {filename}: {str(e)}")
        return None

def process_audio_and_save_mfcc(audio_file_name, dataset_location, annotations_df, output_dir, logging_level=0):
    """
    Slice the audio file according to the annotations and extract MFCC features

    Parameters:
        audio_file_name: Name of audio file with extension
        dataset_location: Path till dataset directory
        annotations_df: Pandas dataframe of csv file read
        output_dir: Name of directory in which all vectors are to be stored
        logging_level: Default 0
            0 - None, 1 - Info level
    """
    audio_file_path = os.path.join(dataset_location, audio_file_name)
    try:
        if not os.path.exists(os.path.join(dataset_location,audio_file_name)):
            raise FileNotFoundError(f"Audio file not found: {audio_file_name}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Loading audio file: {audio_file_name}")
        warnings.filterwarnings("ignore", category=UserWarning, module="librosa.core.spectrum")
        y, sr = librosa.load(audio_file_path, sr=None)
        print(f"Audio loaded with sampling rate: {sr} Hz")
        
        samples_per_microsecond = sr / 1_000_000
        
        with open(f'./{output_dir}/{audio_file_name}_mfccvectors.csv', 'w') as file:
            for idx, row in tqdm(annotations_df.iloc[:-1].iterrows(), unit=" segments"):
                chord = row['chord'].strip().replace("'", "").replace('"', "") 
                if chord == 'N.C.':
                    continue

                start_idx = int(np.floor(row['start_time'] * sr))
                end_idx = int(np.floor(row['end_time'] * sr))
                
                # Ensure valid indices
                if start_idx >= end_idx or end_idx > len(y):
                    print(f"Warning: Invalid indices for segment in {audio_file_name} at {row['start_time']}")
                    continue
                    
                segment = y[start_idx:end_idx]
                
                # Skip if segment is too short
                if len(segment) < sr * 0.1:  # Skip segments shorter than 100ms
                    print(f"Warning: Segment too short in {audio_file_name} at {row['start_time']}")
                    continue
                
                filename = f"{idx+1}_{chord}_{row['start_time']:.6f}_{row['end_time']:.6f}.mp3"
                
                mfcc_vector = mfcc_vectorise_segment(segment, sr, filename)
                
                # Only write to file if we got a valid vector
                if mfcc_vector is not None:
                    one_hot_encoded_list = one_hot_encoder(chord)
                    file.write(f"{str(one_hot_encoded_list)},{mfcc_vector}\n")
                else:
                    print(f"Warning: Skipping invalid segment in {audio_file_name} at {row['start_time']}")

        print(f"Processed: {audio_file_name}")      
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        raise

def process_annotation_file(annotation_file_path):
        """Process a single annotation file and return the processed DataFrame."""
        with open(annotation_file_path, 'r') as file:
            lines = file.readlines()[7:]    
            arff_content = [line.strip().strip("'").split(",") for line in lines]
        
        annotations_df = pd.DataFrame(arff_content, columns=['start_time', 'bar', 'beat', 'chord'])
        annotations_df['start_time'] = annotations_df['start_time'].astype(float)
        annotations_df['bar'] = annotations_df['bar'].astype(int)
        annotations_df['beat'] = annotations_df['beat'].astype(int)
        annotations_df['chord'] = annotations_df['chord'].str.strip("'")
        annotations_df['end_time'] = annotations_df['start_time'].shift(-1)
        annotations_df = annotations_df.ffill()
        return annotations_df

if __name__ == '__main__':
    audio_file_name = '0001_mix.flac'
    dataset_location = "./dataset/audio-mixes/"
    output_dir = 'mfcc_vectors'
    annotations_dir_loc = "./dataset/annotations/"
    annotations_file_name = "0001_beatinfo.arff"
    

    annotations_file = os.path.join(annotations_dir_loc, annotations_file_name)
    annotations_df = load_annotations(annotations_file)
    process_audio_and_save_mfcc(audio_file_name, dataset_location, annotations_df, output_dir)