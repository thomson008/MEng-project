import pyaudio
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from constants import CHUNK, RATE, CHANNELS

LEN = int(sys.argv[1])

recording_angle = sys.argv[2]

print(f'Recording data for angle: {recording_angle} degrees.')
print(f'Recording length will be {LEN} seconds.')

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

frames = []

for i in range(int(LEN * RATE / CHUNK)):
    data = stream.read(CHUNK)
    frames.append(data)

stream.stop_stream()
stream.close()
p.terminate()

print('Saving data...')
frames = np.frombuffer(b''.join(frames), dtype=np.int16)
frames = np.reshape(frames, (CHANNELS, -1), order='F').T
all_data = frames[:, :CHANNELS-1]

cols = [f'mic_{i}' for i in range(CHANNELS-1)]
df = pd.DataFrame(data=all_data, columns=cols)
df['output_angle'] = recording_angle
df.to_csv(f'./training_data/recording_angle_{recording_angle}.csv')

print('Done.')