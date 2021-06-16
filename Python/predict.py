import pyaudio
import numpy as np
from constants import *
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


    def gcc_phat(self, x_1, x_2):
        """
        Function that will compute the GCC-PHAT
        cross-correlation of two separate audio channels
        
        Returns:
            A 1-D GCC vector
        """
        
        n = len(x_1) + len(x_2) - 1
        n += 1 if n % 2 else 0
        
        # Fourier transforms of the two signals
        X_1 = np.fft.rfft(x_1, n=n)
        X_2 = np.fft.rfft(x_2, n=n)
        
        # Normalize by the magnitude of FFT - because PHAT
        np.divide(X_1, np.abs(X_1), X_1, where=np.abs(X_1) != 0)
        np.divide(X_2, np.abs(X_2), X_2, where=np.abs(X_2) != 0)
        
        # GCC-PHAT = [X_1(f)X_2*(f)] / |X_1(f)X_2*(f)|
        # See http://www.xavieranguera.com/phdthesis/node92.html for reference
        CC = X_1 * np.conj(X_2)
        cc = np.fft.irfft(CC, n=n)
            
        # Maximum delay between a pair of microphones,
        # expressed in a number of samples.
        # 0.09 m is the mic array diameter and 
        # 340 m/s is assumed to be the speed of sound.
        max_len = math.ceil(0.09 / 340 * RATE)
        
        # Trim the cc vector to only include a 
        # small number of samples around the origin
        cc = np.concatenate((cc[-max_len:], cc[:max_len+1]))
        
        # Return the cross correlation
        return cc


    def compute_gcc_matrix(self, observation):
        """
        Creates a GCC matrix, where each row is a vector of GCC 
        between a given pair of microphones.
        """
        
        mic_pairs = combinations(range(6), r=2)

        # Initialize a transformed observation, that will be populated with GCC vectors
        # of the observation
        transformed_observation = []

        # Compute GCC for every pair of microphones
        for mic_1, mic_2 in mic_pairs:
            x_1 = observation[:, mic_1]
            x_2 = observation[:, mic_2]

            gcc = self.gcc_phat(x_1, x_2)

            # Add the GCC vector to the GCC matrix
            transformed_observation.append(gcc)
            
        return transformed_observation


    def get_prediction_from_model(self, mic_data):
        gcc_matrix = np.transpose(self.compute_gcc_matrix(mic_data))

        # Dummy random probabilities, 
        # will be replaced with model outputs later
        y_pred = np.random.rand(36)
        y_pred /= sum(y_pred)

        prediction = np.argmax(y_pred) * RESOLUTION

        return prediction