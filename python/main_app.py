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
DIM = 600

# Distance of circle from the corner of canvas
DIST = DIM / 12

# Coordinates of the circle
COORD = DIST, DIST, DIM - DIST, DIM - DIST

# Radius of the circle
RADIUS = DIM / 2 - DIST


def create_canvas():
    C = Canvas(top, bg="white", height=DIM, width=DIM)

    # Create the segmented circle in the middle of the canvas
    for i in range(0, 360, 10):
        C.create_arc(COORD, start=i - RESOLUTION // 2, extent=RESOLUTION, fill='#ebe8e8', outline='#595959')

        text_R = RADIUS + 20
        text_x = DIM / 2 + text_R * math.cos(math.radians(i))
        text_y = DIM / 2 - text_R * math.sin(math.radians(i))

        C.create_text(text_x, text_y, fill="darkblue", font="Arial 11 bold", text=str(i))

    C.place(x=(HEIGHT - DIM) / 2, y=(HEIGHT - DIM) / 2)

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
    x = 700

    label = Label(top, text="CNN DOA")
    label.config(font=("Arial", 40), fg="#4a4a4a")
    label.place(x=x, y=25)

    doa_label = Label(top, text="Angle")
    doa_label.config(font=("Arial", 14), fg="#4a4a4a")
    doa_label.place(x=x, y=120)

    conf_label = Label(top, text="Confidence")
    conf_label.config(font=("Arial", 14), fg="#4a4a4a")
    conf_label.place(x=x, y=320)

    doa_val = Label(top, text="-")
    doa_val.config(font=("Arial", 40))
    doa_val.place(x=x, y=150)

    conf_val = Label(top, text="-")
    conf_val.config(font=("Arial", 40))
    conf_val.place(x=x, y=350)

    return doa_label, doa_val, conf_label, conf_val


def start_app():
    C = create_canvas()
    doa_label, doa_val, conf_label, conf_val = create_labels()
    B = Button(top, text="Exit", command=top.destroy, height=2, width=10, font=("Arial", 12))
    B.place(x=780, y=500)

    predictor = Predictor()

    while True:
        # Get probabilities from model
        confs = predictor.confidences
        max_idx = np.argmax(confs)

        # Color arcs based on model probabilities
        try:
            color_arcs(C, confs, max_idx)
        except TclError:
            sys.exit()

        prediction = predictor.current_prediction

        if prediction is not None:
            pred, conf = prediction
            conf = round(conf * 100, 1)
            doa_val.config(text=f'{pred}\N{DEGREE SIGN}')
            conf_val.config(text=f'{conf}%')
        else:
            doa_val.config(text='-')
            conf_val.config(text='-')


top = Tk()
top.title('ML DoA recognition')
top.geometry(f'{WIDTH}x{HEIGHT}')
top.resizable(False, False)

start_app()
