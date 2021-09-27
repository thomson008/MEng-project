#!/usr/bin/python3

import math
import sys
from tkinter import *

import numpy as np

from predict import Predictor
from utils import RESOLUTION

# Window size
WIDTH = 1000
HEIGHT = 650

# Canvas size
DIM = 650

# Distance of circle from the corner of canvas
DIST = DIM / 12

# Coordinates of the circle
COORD = DIST, DIST, DIM - DIST, DIM - DIST

# Radius of the circle
RADIUS = DIM / 2 - DIST


def create_canvas():
    C = Canvas(circle_frame, bg="white", height=DIM, width=DIM)

    # Create the segmented circle in the middle of the canvas
    for i in range(0, 360, 10):
        C.create_arc(COORD, start=i - RESOLUTION // 2, extent=RESOLUTION, fill='#ebe8e8', outline='#595959')

        text_R = RADIUS + 20
        text_x = DIM / 2 + text_R * math.cos(math.radians(i))
        text_y = DIM / 2 - text_R * math.sin(math.radians(i))

        C.create_text(text_x, text_y, fill="darkblue", font="Arial 11 bold", text=str(i))

    C.pack(fill='both')

    return C


def color_arcs(C, confs, max_idx):
    arcs = []
    for i, conf in enumerate(confs):
        # Color the fraction of arc area proportional to probability
        R = RADIUS * conf ** 0.5

        x_1 = y_1 = DIM / 2 - R

        coord = x_1, y_1, x_1 + 2 * R, y_1 + 2 * R

        if i == max_idx:
            fill = '#78ebb3'
            outline = '#2ca86b'
        else:
            fill = '#ffadad'
            outline = '#db6b6b'

        arc = C.create_arc(coord, start=i * RESOLUTION - 5, extent=RESOLUTION, fill=fill, outline=outline, width=2)
        arcs.append(arc)

    top.update()

    # Delete all arcs to draw them again at next iteration
    for arc in arcs:
        C.delete(arc)


def create_labels():
    x = 40
    x_shift = 140
    value_font_size = 25

    label = Label(data_frame, text="CNN DOA")
    label.config(font=("Arial", 40), fg="#4a4a4a")
    label.pack()

    y_shift = 30
    y_azimuth = 120
    az_conf_label, az_conf_val, azimuth_label, azimuth_val = create_doa_labels(
        'Azimuth', value_font_size, x, x_shift, y_azimuth, y_shift)

    y_elevation = 250
    el_conf_label, el_conf_val, elevation_label, elevation_val = create_doa_labels(
        'Elevation', value_font_size, x, x_shift, y_elevation, y_shift)

    return azimuth_label, azimuth_val, az_conf_label, az_conf_val, \
        elevation_label, elevation_val, el_conf_label, el_conf_val


def create_doa_labels(text, value_font_size, x, x_shift, y, y_shift):
    angle_label = Label(data_frame, text=text)
    angle_label.config(font=("Arial", 14), fg="#4a4a4a")
    angle_label.place(x=x, y=y)

    conf_label = Label(data_frame, text="Confidence")
    conf_label.config(font=("Arial", 14), fg="#4a4a4a")
    conf_label.place(x=x + x_shift, y=y)

    angle_val = Label(data_frame, text="-")
    angle_val.config(font=("Arial", value_font_size))
    angle_val.place(x=x, y=y + y_shift)

    conf_val = Label(data_frame, text="-")
    conf_val.config(font=("Arial", value_font_size))
    conf_val.place(x=x + x_shift, y=y + y_shift)

    return conf_label, conf_val, angle_label, angle_val


def create_frames():
    left_frame = LabelFrame(top, width=WIDTH - DIM - 1, height=HEIGHT)
    left_frame.pack(side=RIGHT)
    left_frame.pack_propagate(False)

    right_frame = LabelFrame(top, width=HEIGHT, height=HEIGHT)
    right_frame.pack(side=LEFT)
    right_frame.pack_propagate(False)

    return left_frame, right_frame


def toggle_prediction():
    global prediction_running, start_button
    prediction_running = not prediction_running

    text = 'Stop' if prediction_running else 'Start'
    start_button.config(text=text, relief=SUNKEN if prediction_running else RAISED)


def start_app():
    C = create_canvas()
    az_label, az_val, az_conf_label, az_conf_val, el_label, el_val, el_conf_label, el_conf_val = create_labels()

    exit_button = Button(data_frame, text="Exit", command=top.destroy,
                         height=2, width=10, font=("Arial", 12), cursor="hand2")
    exit_button.place(relx=0.5, y=520, anchor=CENTER)

    predictor = Predictor()

    while True:
        predictor.is_active = prediction_running

        # Get probabilities from model
        confs = predictor.az_confidences
        max_idx = np.argmax(confs)

        # Color arcs based on model probabilities
        try:
            color_arcs(C, confs, max_idx)
        except TclError:
            print()
            sys.exit()

        az_prediction = predictor.az_current_prediction
        el_prediction = predictor.el_current_prediction

        if az_prediction is not None:
            (pred, conf), (el_pred, el_conf) = az_prediction, el_prediction
            conf = round(conf * 100, 1)
            el_conf = round(el_conf * 100, 1)
            az_val.config(text=f'{pred}\N{DEGREE SIGN}')
            az_conf_val.config(text=f'{conf}%')
            el_val.config(text=f'{el_pred}\N{DEGREE SIGN}')
            el_conf_val.config(text=f'{el_conf}%')
        else:
            az_val.config(text='-')
            az_conf_val.config(text='-')
            el_val.config(text='-')
            el_conf_val.config(text='-')


top = Tk()
top.title('ML DoA recognition')
top.geometry(f'{WIDTH}x{HEIGHT}')
top.resizable(False, False)
data_frame, circle_frame = create_frames()
start_button = Button(data_frame, text="Start", command=toggle_prediction,
                      height=3, width=15, font=("Arial", 12), cursor="hand2")
start_button.place(relx=0.5, y=420, anchor=CENTER)

prediction_running = False
start_app()
