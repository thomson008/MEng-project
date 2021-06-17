import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from utils import CHUNK, RATE, CHANNELS, FORMAT

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Create figure for subplots of signals
fig, axs = plt.subplots(7, figsize=(6,12))
fig.suptitle('Microphone array data')
plt.subplots_adjust(hspace=0.8)
lines = []

for i, ax in enumerate(axs):
    ax.set_title(f'Microphone {i}')
    ax.set_ylim(-250, 250)
    ax.set_xlim(0, CHUNK)

    x = np.arange(0, 2 * CHUNK, 2)
    lines += ax.plot(x, np.random.rand(CHUNK))

i = 1
# Read signals forever
while True:
    try:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

        # Update plots with data from the new frame
        for c in range(CHANNELS-1):
            mic_data = data[c::CHANNELS]
            lines[c].set_ydata(mic_data)

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.005)

        # Print out the first reading of each microphone from the new frame
        frmt = "{:>5}" * (CHANNELS-1)
        print(f'Frame ' '{:>3}:    '.format(i) + '[' + frmt.format(*data[:(CHANNELS-1)]) + '   ]')
        i += 1
        
    except KeyboardInterrupt:
        break
    
stream.stop_stream()
stream.close()
p.terminate()