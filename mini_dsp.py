import pyaudio
import numpy as np

CHUNK = 2048
RATE = 44100
CHANNELS = 8

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
player = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK)

i = 1

# Read signals forever
while True:
    try:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        print(f'Chunk no. {i}: ' + str(data[:7]))
        i += 1
        
    except KeyboardInterrupt:
        break
    
stream.stop_stream()
stream.close()
p.terminate()