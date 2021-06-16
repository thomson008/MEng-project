import pyaudio
import numpy as np
from utils import *
from itertools import combinations
import math

class Predictor():
    def __init__(self):
        self.current_prediction = None

        def callback(in_data, frame_count, time_info, status):
            data = np.frombuffer(in_data, dtype=np.int16)
            data = np.reshape(data, (-1, CHANNELS))

            # Drop irrelevant channels and reorder remaining channels,
            # in order to match the simulated microphone array
            mic_data = data[:, -2:0:-1]
            self.current_prediction = self.get_prediction_from_model(mic_data)
            
            return (data, pyaudio.paContinue)

        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(
            format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
            frames_per_buffer=CHUNK, stream_callback=callback
        )

        self.stream.start_stream()

    
    def end_stream(self):
        # stop stream
        self.stream.stop_stream()
        stream.close()

        # close PyAudio
        self.p.terminate()


    def get_prediction_from_model(self, mic_data):
        gcc_matrix = np.transpose(compute_gcc_matrix(mic_data))

        # Dummy random probabilities, 
        # will be replaced with model outputs later
        y_pred = np.random.rand(36)
        y_pred /= sum(y_pred)

        prediction = np.argmax(y_pred) * RESOLUTION

        return prediction