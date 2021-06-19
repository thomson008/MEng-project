from tkinter import *
import math
import time
import numpy as np
from utils import RESOLUTION
from predict import Predictor

top = Tk()
top.title('ML DoA recognition')

dim = 600
dist = 50
coord = dist, dist, dim - dist, dim - dist
radius = dim / 2 - dist

C = Canvas(top, bg="white", height=dim, width=dim)

for i in range(0, 360, 10):
    arc = C.create_arc(coord, start=i-RESOLUTION//2, extent=RESOLUTION, fill='#ebe8e8', outline='#595959')

    R = radius + 20
    text_x = dim / 2 + R * math.cos(math.radians(i))
    text_y = dim / 2 - R * math.sin(math.radians(i))

    text = C.create_text(text_x, text_y, fill="darkblue", font="Arial 11 bold", text=str(i))

C.pack()

predictor = Predictor()

while True:
    confs = predictor.confidences
    max_idx = np.argmax(confs)

    arcs = []

    for i, conf in enumerate(confs):
        frac_R =  conf ** 0.5
        R = radius * frac_R

        x_1 = y_1 = dim / 2 - R

        coord = x_1, y_1, x_1 + 2 * R, y_1 + 2 * R

        if i == max_idx:
            fill = '#78ebb3'
            outline = '#2ca86b'
        else:
            fill = '#ffadad'
            outline = '#db6b6b'

        arc = C.create_arc(coord, start=i*RESOLUTION-5, extent=RESOLUTION, fill=fill, outline=outline, width=2)
        arcs.append(arc)

    top.update()

    for arc in arcs:
        C.delete(arc)
