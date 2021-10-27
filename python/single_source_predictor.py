import os
import pathlib

import tensorflow as tf

from predictor import Predictor, get_mic_data, get_input_matrix, get_model_details
from utils import *


def init_models():
    print('Loading models...')
    # Load the TFLite model and allocate tensors.
    base_dir = pathlib.Path(__file__).parent.parent.absolute()
    az_model_file = os.path.join(base_dir, 'models', 'best_super_azimuth_model.tflite')
    el_model_file = os.path.join(base_dir, 'models', 'elevation_model.tflite')
    az_interpreter = tf.lite.Interpreter(model_path=az_model_file)
    el_interpreter = tf.lite.Interpreter(model_path=el_model_file)

    print('Allocating tensors...')
    az_interpreter.allocate_tensors()
    el_interpreter.allocate_tensors()
    print('Tensors allocated.\n')

    # Get input and output tensors for both models
    az_input_details, az_input_shape, az_output_details, az_output_shape = get_model_details(az_interpreter)
    el_input_details, el_input_shape, el_output_details, el_output_shape = get_model_details(el_interpreter)

    print('Azimuth model input tensor: ' + str(az_input_shape))
    print('Azimuth model output tensor: ' + str(az_output_shape))
    print('Elevation model input tensor: ' + str(el_input_shape))
    print('Elevation model output tensor: ' + str(el_output_shape))
    print('\nModels ready. Press Start to begin inference.\n')

    return az_interpreter, az_input_details, az_output_details, el_interpreter, el_input_details, el_output_details


class SingleSourcePredictor(Predictor):
    def __init__(self, thresh=50, max_silence_frames=10):
        super().__init__(thresh, max_silence_frames)
        self.az_current_prediction = None
        self.el_current_prediction = None
        self.az_confidences = np.zeros(360 // AZIMUTH_RESOLUTION)

        self.stream = self.p.open(
            format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
            frames_per_buffer=CHUNK, stream_callback=self.callback
        )

        self.stream.start_stream()
        self.az_interpreter, self.az_input_details, self.az_output_details, \
            self.el_interpreter, self.el_input_details, self.el_output_details = init_models()

    def callback(self, in_data, frame_count, time_info, status):
        data, mic_data = get_mic_data(in_data)

        if abs(np.max(mic_data)) > self.thresh and self.is_active:
            self.az_current_prediction, self.el_current_prediction = self.get_prediction_from_models(mic_data)
        else:
            if self.silent_frames == self.max_silence_frames:
                self.silent_frames = 0
                self.az_current_prediction = None
                self.az_confidences = np.zeros(360 // AZIMUTH_RESOLUTION)
                self.el_current_prediction = None
            self.silent_frames += 1
        if self.is_active:
            self.output_predictions()

        return data, pyaudio.paContinue

    def get_prediction_from_models(self, mic_data):
        input_data = get_input_matrix(mic_data)

        # Set input and run azimuth interpreter
        self.az_interpreter.set_tensor(self.az_input_details[0]['index'], input_data)
        self.az_interpreter.invoke()
        az_output_data = self.az_interpreter.get_tensor(self.az_output_details[0]['index'])
        self.az_confidences = az_output_data[0]

        # Get the predicted azimuth as argument of the max probability
        az_prediction, az_confidence = np.argmax(az_output_data[0]) * AZIMUTH_RESOLUTION, np.max(az_output_data[0])

        # Set input and run elevation interpreter
        self.el_interpreter.set_tensor(self.el_input_details[0]['index'], input_data)
        self.el_interpreter.invoke()
        el_output_data = self.el_interpreter.get_tensor(self.el_output_details[0]['index'])

        # Get the predicted elevation as argument of the max probability
        el_prediction, el_confidence = np.argmax(el_output_data[0]) * ELEVATION_RESOLUTION, np.max(el_output_data[0])

        return (az_prediction, az_confidence), (el_prediction, el_confidence)

    def output_predictions(self):
        if self.az_current_prediction is not None:
            az_pred, az_conf = self.az_current_prediction
            el_pred, el_conf = self.el_current_prediction
            az_conf = round(az_conf * 100, 1)
            el_conf = round(el_conf * 100, 1)
            print('Azimuth: {:>3} degrees [{:>5}%]'.format(az_pred, az_conf), end=' | ')
            print('Elevation: {:>3} degrees [{:>5}%]'.format(el_pred, el_conf))
        else:
            print('{:<63}'.format('[No prediction]'))
