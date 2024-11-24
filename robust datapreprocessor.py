import os
import pandas as pd
from robust_extraction import process_audio_and_save_features
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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

def process_file_pair(args):
    """Process a pair of audio and annotation files."""
    audio_file_name, annotations_file_name, dataset_location, annotations_dir_loc, output_dir = args
    try:
        # Process annotation file
        annotation_file_path = os.path.join(annotations_dir_loc, annotations_file_name)
        annotations_df = process_annotation_file(annotation_file_path)
        
        # Process audio and save PCP
        process_audio_and_save_features(audio_file_name, dataset_location, annotations_df, output_dir)
        
        return f"Successfully processed {audio_file_name}"
    except Exception as e:
        return f"Error processing {audio_file_name}: {str(e)}"

def main():
    # Configuration
    dataset_location = "./dataset/audio-mixes/"
    output_dir = 'extracted robust 45 annotations'
    annotations_dir_loc = "./dataset/annotations/"
    
    # Get file lists
    audio_file_names = [file for root, dirs, files in os.walk(dataset_location) for file in files]
    arff_files = [file for root, dirs, files in os.walk(annotations_dir_loc) 
                  for file in files if file.endswith('beatinfo.arff')]
    
    # Prepare arguments for parallel processing
    process_args = [
        (audio_file, arff_file, dataset_location, annotations_dir_loc, output_dir)
        for audio_file, arff_file in zip(audio_file_names, arff_files)
    ]
    
    # Use all available logical processors
    num_workers = os.cpu_count()
    
    # Initialize multiprocessing progress bar
    multiprocessing.freeze_support()  # Needed for Windows
    
    # Process files in parallel with progress bar
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_file_pair, args): args[0]
            for args in process_args
        }
        
        # Create progress bar
        with tqdm(total=len(process_args), unit=' Files') as pbar:
            for future in as_completed(future_to_file):
                file_name = future_to_file[future]
                try:
                    result = future.result()
                    if "Error" in result:
                        print(f"\nWarning: {result}")
                except Exception as e:
                    print(f"\nError processing {file_name}: {str(e)}")
                finally:
                    pbar.update(1)

if __name__ == "__main__":
    main()