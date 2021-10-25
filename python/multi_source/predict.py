import os
import pathlib
import platform

import tensorflow as tf

from alsa_suppress import noalsaerr
from utils import *


def init_models():
    print('Loading models...')
    # Load the TFLite model and allocate tensors.
    base_dir = pathlib.Path(__file__).parent.parent.parent.absolute()
    az_model_file = os.path.join(base_dir, 'models', 'best_multi_source_model.tflite')
    az_interpreter = tf.lite.Interpreter(model_path=az_model_file)

    print('Allocating tensors...')
    az_interpreter.allocate_tensors()
    print('Tensors allocated.\n')

    # Get input and output tensors for both models
    az_input_details = az_interpreter.get_input_details()
    az_output_details = az_interpreter.get_output_details()
    az_input_shape = az_input_details[0]['shape']
    az_output_shape = az_output_details[0]['shape']
    print('Azimuth model input tensor: ' + str(az_input_shape))
    print('Azimuth model output tensor: ' + str(az_output_shape))
    print('\nModels ready. Press Start to begin inference.\n')

    return az_interpreter, az_input_details, az_output_details


class Predictor:
    def __init__(self, thresh=50, max_silence_frames=10):
        # Model parameters
        self.is_active = False
        self.az_current_predictions = []
        self.az_confidences = np.zeros(360 // AZIMUTH_RESOLUTION)

        # Thresholds for deciding whether to run or not
        self.thresh = thresh
        self.silent_frames = 0
        self.max_silence_frames = max_silence_frames

        if platform.system() == 'Windows':
            self.p = pyaudio.PyAudio()
        else:
            with noalsaerr():
                self.p = pyaudio.PyAudio()

        self.stream = self.p.open(
            format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
            frames_per_buffer=CHUNK, stream_callback=self.callback
        )

        self.stream.start_stream()
        self.az_interpreter, self.az_input_details, self.az_output_details = init_models()

    def callback(self, in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.int16)
        data = np.reshape(data, (-1, CHANNELS))

        # Drop irrelevant channels and reorder remaining channels,
        # in order to match the simulated microphone array
        mic_data = np.hstack([data[:, 1].reshape(-1, 1), data[:, -2:1:-1]])

        if abs(np.max(mic_data)) > self.thresh and self.is_active:
            self.az_current_predictions = self.get_prediction_from_model(mic_data)
        else:
            if self.silent_frames == self.max_silence_frames:
                self.silent_frames = 0
                self.az_current_predictions = []
            self.silent_frames += 1
        if self.is_active:
            self.output_predictions()

        return data, pyaudio.paContinue

    def end_stream(self):
        # stop stream
        self.stream.stop_stream()
        self.stream.close()

        # close PyAudio
        self.p.terminate()

    def get_prediction_from_model(self, mic_data):
        gcc_matrix = np.transpose(compute_gcc_matrix(mic_data))
        input_data = np.array([gcc_matrix], dtype=np.float32)

        # Set input and run azimuth interpreter
        self.az_interpreter.set_tensor(self.az_input_details[0]['index'], input_data)
        self.az_interpreter.invoke()
        az_output_data = self.az_interpreter.get_tensor(self.az_output_details[0]['index'])
        return az_output_data[0]

    def output_predictions(self):
        if len(self.az_current_predictions):
            print([(angle * AZIMUTH_RESOLUTION, conf)
                   for angle, conf in enumerate(self.az_current_predictions) if conf > 0.5])
        else:
            print('[No prediction]')
