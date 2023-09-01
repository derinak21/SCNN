import pyaudio
import numpy as np
import torch
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tkinter import TclError
from model.mlp import MLPModule
from datasets.dataset import SourceCountingDataLoader



# Create the audio callback function
def audio_callback(in_data, frame_count, time_info, status):
    audio_np = np.frombuffer(in_data, dtype=np.int16)
    # Preprocess audio data if needed
    preprocessed_data = preprocess_audio(audio_np)
    preprocessed_tensor = torch.tensor(preprocessed_data)
    # Perform inference using your model
    with torch.no_grad():
        model.eval()
        output = model(preprocessed_tensor)
    # Process the output to get the predicted number of speakers
    num_speakers = process_output(output)
    # Print the predicted number of speakers
    print(f"Predicted number of speakers: {num_speakers}")
    return in_data, pyaudio.paContinue

# Main function
def main(cfg):
    # Load your trained model
    model = MLPModule.load_from_checkpoint(cfg.checkpoint_path)

    # Set up PyAudio
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=audio_callback
    )

    # Start the PyAudio stream
    stream.start_stream()

    try:
        while stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        # Close the stream when KeyboardInterrupt occurs
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Stream stopped.")

if __name__ == "__main__":
    # Load your configuration and call the main function
    main(your_config)
