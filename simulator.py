import pyaudio
import wave

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1  # Use 1 channel per microphone
fs = 44100  # Record at 44100 samples per second
seconds = 5

# Define filenames for the two recordings
filename_mic0 = "output_mic0.wav"
filename_mic1 = "output_mic1.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"Device {i}: {info['name']}")
print('Recording')

# Stream for microphone with index 0
stream0 = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True,
                input_device_index=0)

# Stream for microphone with index 1
stream1 = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True,
                input_device_index=1)

frames0 = []  # Initialize array to store frames for mic 0
frames1 = []  # Initialize array to store frames for mic 1

# Store data in chunks for the specified duration
for i in range(0, int(fs / chunk * seconds)):
    data0 = stream0.read(chunk)  # Read from mic 0
    data1 = stream1.read(chunk)  # Read from mic 1
    frames0.append(data0)
    frames1.append(data1)

# Stop and close the streams
stream0.stop_stream()
stream0.close()
stream1.stop_stream()
stream1.close()

# Terminate the PortAudio interface
p.terminate()

print('Finished recording')

# Save the recorded data as WAV files for each microphone
wf0 = wave.open(filename_mic0, 'wb')
wf0.setnchannels(channels)
wf0.setsampwidth(p.get_sample_size(sample_format))
wf0.setframerate(fs)
wf0.writeframes(b''.join(frames0))
wf0.close()

wf1 = wave.open(filename_mic1, 'wb')
wf1.setnchannels(channels)
wf1.setsampwidth(p.get_sample_size(sample_format))
wf1.setframerate(fs)
wf1.writeframes(b''.join(frames1))
wf1.close()
