import os
import pathlib
import platform

import tensorflow as tf

from alsa_suppress import noalsaerr
from utils import *


def init_models():
    print('Loading models...')
    # Load the TFLite model and allocate tensors.
    base_dir = pathlib.Path(__file__).parent.parent.absolute()
    az_model_file = os.path.join(base_dir, 'models', 'azimuth_model.tflite')
    el_model_file = os.path.join(base_dir, 'models', 'elevation_model_new.tflite')
    az_interpreter = tf.lite.Interpreter(model_path=az_model_file)
    el_interpreter = tf.lite.Interpreter(model_path=el_model_file)

    print('Allocating tensors...')
    az_interpreter.allocate_tensors()
    el_interpreter.allocate_tensors()
    print('Tensors allocated.\n')

    # Get input and output tensors for both models
    az_input_details = az_interpreter.get_input_details()
    az_output_details = az_interpreter.get_output_details()
    az_input_shape = az_input_details[0]['shape']
    az_output_shape = az_output_details[0]['shape']

    el_input_details = el_interpreter.get_input_details()
    el_output_details = el_interpreter.get_output_details()
    el_input_shape = el_input_details[0]['shape']
    el_output_shape = el_output_details[0]['shape']

    print('Azimuth model input tensor: ' + str(az_input_shape))
    print('Azimuth model output tensor: ' + str(az_output_shape))
    print('Elevation model input tensor: ' + str(el_input_shape))
    print('Elevation model output tensor: ' + str(el_output_shape))
    print('\nModels ready. Starting inference:\n')

    return az_interpreter, az_input_details, az_output_details, el_interpreter, el_input_details, el_output_details


class Predictor:
    def __init__(self, thresh=50, max_silence_frames=10):
        # Model parameters
        self.is_active = False
        self.az_current_prediction = None
        self.el_current_prediction = None
        self.az_confidences = np.zeros(360 // RESOLUTION)

        # Thresholds for deciding whether to run or not
        self.thresh = thresh
        self.silent_frames = 0
        self.max_silence_frames = max_silence_frames

        if platform.system() == 'Windows':
            self.p = pyaudio.PyAudio()
            self.thresh = 300
        else:
            with noalsaerr():
                self.p = pyaudio.PyAudio()

        self.stream = self.p.open(
            format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
            frames_per_buffer=CHUNK, stream_callback=self.callback
        )

        self.stream.start_stream()
        self.az_interpreter, self.az_input_details, self.az_output_details, \
            self.el_interpreter, self.el_input_details, self.el_output_details = init_models()

    def callback(self, in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.int16)
        data = np.reshape(data, (-1, CHANNELS))

        # Drop irrelevant channels and reorder remaining channels,
        # in order to match the simulated microphone array
        mic_data = np.hstack([data[:, 1].reshape(-1, 1), data[:, -2:1:-1]])

        if abs(np.max(mic_data)) > self.thresh and self.is_active:
            self.az_current_prediction, self.el_current_prediction = self.get_prediction_from_models(mic_data)

        else:
            if self.silent_frames == self.max_silence_frames:
                self.silent_frames = 0
                self.az_current_prediction = None
                self.az_confidences = np.zeros(360 // RESOLUTION)
                self.el_current_prediction = None
            self.silent_frames += 1

        self.output_predictions()
        return data, pyaudio.paContinue

    def end_stream(self):
        # stop stream
        self.stream.stop_stream()
        self.stream.close()

        # close PyAudio
        self.p.terminate()

    def get_prediction_from_models(self, mic_data):
        gcc_matrix = np.transpose(compute_gcc_matrix(mic_data))
        input_data = np.array([gcc_matrix], dtype=np.float32)

        # Set input and run azimuth interpreter
        self.az_interpreter.set_tensor(self.az_input_details[0]['index'], input_data)
        self.az_interpreter.invoke()
        az_output_data = self.az_interpreter.get_tensor(self.az_output_details[0]['index'])
        self.az_confidences = az_output_data[0]

        # Get the predicted azimuth as argument of the max probability
        az_prediction, az_confidence = np.argmax(az_output_data[0]) * RESOLUTION, np.max(az_output_data[0])

        # Set input and run elevation interpreter
        self.el_interpreter.set_tensor(self.el_input_details[0]['index'], input_data)
        self.el_interpreter.invoke()
        el_output_data = self.el_interpreter.get_tensor(self.el_output_details[0]['index'])

        # Get the predicted elevation as argument of the max probability
        el_prediction, el_confidence = np.argmax(el_output_data[0]) * RESOLUTION, np.max(el_output_data[0])

        return (az_prediction, az_confidence), (el_prediction, el_confidence)

    def output_predictions(self):
        if self.az_current_prediction is not None:
            az_pred, az_conf = self.az_current_prediction
            el_pred, el_conf = self.el_current_prediction
            az_conf = round(az_conf * 100, 1)
            el_conf = round(el_conf * 100, 1)
            print('Azimuth: {:>3} degrees [{:>5}%]'.format(az_pred, az_conf), end=' | ')
            print('Elevation: {:>3} degrees [{:>5}%]'.format(el_pred, el_conf), end='\r')
        else:
            print('{:<63}'.format('[No prediction]'), end='\r')
