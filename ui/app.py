import gradio as gr
import tempfile
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from running_inference import infer_chords, format_time


def process_audio(audio_file):
    if audio_file is None:
        return "Please upload an audio file."
    
    try:
        # In Gradio, audio_file is now a tuple of (sample_rate, audio_data)
        # or a string path depending on how it was uploaded
        if isinstance(audio_file, str):
            temp_path = audio_file  # If it's already a path, use it directly
        else:
            # Create a temporary directory and file
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, "temp_audio.mp3")
            
            # For safety, ensure the path exists
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        # Process the audio file
        model_weights_path = "model.h5"  # Ensure this path is correct
        chord_segments = infer_chords(temp_path, model_weights_path)
        
        # Format the results
        result_text = "Predicted Chord Progression:\n"
        result_text += "-----------------------------\n"
        for start_time, end_time, chord in chord_segments:
            result_text += f"{format_time(start_time)} - {format_time(end_time)}: {chord}\n"
        
        # Clean up temporary files if we created them
        if not isinstance(audio_file, str):
            os.remove(temp_path)
            os.rmdir(temp_dir)
        
        return result_text
        
    except Exception as e:
        return f"An error occurred: {str(e)}\nFull error details: {type(e).__name__}"

# Create the Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Chord Detection System")
    gr.Markdown("Upload an audio file to detect its chord progression.")
    
    with gr.Row():
        # Specify sources to allow both microphone and upload
        audio_input = gr.Audio(
            label="Upload Audio File",
            type="filepath",
            sources=["upload", "microphone"]
        )
    
    with gr.Row():
        process_btn = gr.Button("Process Audio", variant="primary")
    
    with gr.Row():
        output_text = gr.Textbox(
            label="Results", 
            placeholder="Results will appear here...",
            lines=10,
            max_lines=20
        )
    
    # Connect the button to the processing function
    process_btn.click(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[output_text],
        api_name="process_audio"
    )

# Launch the app
if __name__ == "__main__":
    app.launch(share=False)