import gradio as gr
import tempfile
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from running_inference import infer_chords, format_time
import json

def process_audio(audio_file):
    if audio_file is None:
        return None, None, "Please upload an audio file."
    
    try:
        if isinstance(audio_file, str):
            temp_path = audio_file
        else:
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, "temp_audio.mp3")
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        # Process the audio file
        model_weights_path = "pcpmodel after training on 1000 files.h5"
        chord_segments = infer_chords(temp_path, model_weights_path)
        
        # Format the results for display
        result_text = "Predicted Chord Progression:\n"
        result_text += "-----------------------------\n"
        for start_time, end_time, chord in chord_segments:
            result_text += f"{format_time(start_time)} - {format_time(end_time)}: {chord}\n"
        
        # Create JSON for the interactive visualization
        chord_data = [
            {
                "start": start_time,
                "end": end_time,
                "chord": chord
            }
            for start_time, end_time, chord in chord_segments
        ]
        
        # Clean up temporary files if we created them
        if not isinstance(audio_file, str):
            os.remove(temp_path)
            os.rmdir(temp_dir)
        
        return audio_file, json.dumps(chord_data), result_text
        
    except Exception as e:
        return None, None, f"An error occurred: {str(e)}\nFull error details: {type(e)._name_}"

def get_current_chord(timestamp, chord_data):
    if not chord_data:
        return "No chord data available"
    
    try:
        chords = json.loads(chord_data)
        current_chord = "No chord detected"
        
        for segment in chords:
            if segment["start"] <= timestamp <= segment["end"]:
                current_chord = segment["chord"]
                break
        
        return f"Current Chord: {current_chord}"
    except:
        return "Error processing chord data"

# Create the Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Interactive Chord Detection System")
    gr.Markdown("Upload an audio file to detect and visualize its chord progression.")
    
    # Store chord data for use across components
    chord_data_state = gr.State()
    current_time = gr.State(value=0)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Audio input and player with a custom player
            audio_input = gr.Audio(
                label="Upload Audio File",
                type="filepath",
                sources=["upload", "microphone"],
                interactive=True
            )
            process_btn = gr.Button("Process Audio", variant="primary")
            
            # Slider for manual time control
            time_slider = gr.Slider(
                minimum=0,
                maximum=300,  # 5 minutes max
                value=0,
                step=0.1,
                label="Current Time (seconds)"
            )
        
        with gr.Column(scale=1):
            # Results display
            output_text = gr.Textbox(
                label="Chord Progression", 
                placeholder="Results will appear here...",
                lines=10,
                max_lines=20
            )
    
    # Current chord display below the audio player
    current_chord_display = gr.Markdown("Current Chord: None")
    
    # Process the audio and update displays
    def process_and_update(audio):
        playback_audio, chord_data, results = process_audio(audio)
        return playback_audio, chord_data, results, 0  # Reset time slider
    
    # Update current chord based on slider
    def update_display(time, chord_data):
        return get_current_chord(time, chord_data)
    
    # Connect the processing button
    process_btn.click(
        fn=process_and_update,
        inputs=[audio_input],
        outputs=[
            audio_input,
            chord_data_state,
            output_text,
            time_slider
        ]
    )
    
    # Connect the slider to update the current chord display
    time_slider.change(
        fn=update_display,
        inputs=[time_slider, chord_data_state],
        outputs=[current_chord_display],
    )

# Launch the app
if __name__ == "__main__":
    app.launch(share=False)