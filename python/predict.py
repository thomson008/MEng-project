import os
import pathlib
import platform

import tensorflow as tf

from alsa_suppress import noalsaerr
from utils import *


def init_model():
    print('Loading model...')
    # Load the TFLite model and allocate tensors.
    base_dir = pathlib.Path(__file__).parent.parent.absolute()
    model_file = os.path.join(base_dir, 'models', 'model.tflite')
    interpreter = tf.lite.Interpreter(model_path=model_file)

    print('Allocating tensors...\n')
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    output_shape = output_details[0]['shape']

    print('Input tensor: ' + str(input_shape))
    print('Output tensor: ' + str(output_shape))

    print('\nModel ready. Starting inference:\n')

    return interpreter, input_details, output_details


class Predictor:
    def __init__(self, thresh=50):
        self.is_active = False
        self.current_prediction = None
        self.confidences = np.zeros(360 // RESOLUTION)
        self.thresh = thresh

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
        self.interpreter, self.input_details, self.output_details = init_model()

    def callback(self, in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.int16)
        data = np.reshape(data, (-1, CHANNELS))

        # Drop irrelevant channels and reorder remaining channels,
        # in order to match the simulated microphone array
        mic_data = np.hstack([data[:, 1].reshape(-1, 1), data[:, -2:1:-1]])

        if abs(np.max(mic_data)) > self.thresh and self.is_active:
            self.current_prediction = self.get_prediction_from_model(mic_data)
        else:
            self.current_prediction = None
            self.confidences = np.zeros(360 // RESOLUTION)

        self.run()

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

        # Set input and run interpreter
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        self.confidences = output_data[0]

        # Get the predicted DOA as argument of the max probability
        prediction, confidence = np.argmax(output_data[0]) * RESOLUTION, np.max(output_data[0])

        return prediction, confidence

    def run(self):
        if self.current_prediction is not None:
            pred, conf = self.current_prediction
            conf = round(conf * 100, 1)
            print('DOA: {:>3} degrees [{:>5}%]'.format(pred, conf), end='\r')
        else:
            print('{:<25}'.format('[No prediction]'), end='\r')
