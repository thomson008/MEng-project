#!/usr/bin/python3

import sys
from tkinter import *

import numpy as np

from predict import Predictor
from python.doa_app import DoaApp
from utils import UI_RESOLUTION


class SingleSourceApp(DoaApp):
    def color_arcs(self, C, confs, max_idx):
        arcs = []
        for i, conf in enumerate(confs):
            # Color the fraction of arc area proportional to probability
            R = self.RADIUS * conf ** 0.5

            x_1 = y_1 = self.DIM / 2 - R

            coord = x_1, y_1, x_1 + 2 * R, y_1 + 2 * R

            if i == max_idx:
                fill = '#78ebb3'
                outline = '#2ca86b'
            else:
                fill = '#ffadad'
                outline = '#db6b6b'

            arc = C.create_arc(coord, start=i * UI_RESOLUTION - 5,
                               extent=UI_RESOLUTION, fill=fill, outline=outline, width=2)
            arcs.append(arc)

        self.top.update()

        # Delete all arcs to draw them again at next iteration
        for arc in arcs:
            C.delete(arc)

    def create_labels(self):
        x = 40
        x_shift = 140
        value_font_size = 25

        label = Label(self.data_frame, text="CNN DOA")
        label.config(font=("Arial", 40), fg="#4a4a4a")
        label.pack()

        y_shift = 30
        y_azimuth = 120
        az_conf_label, az_conf_val, azimuth_label, azimuth_val = self.create_doa_labels(
            'Azimuth', value_font_size, x, x_shift, y_azimuth, y_shift)

        y_elevation = 250
        el_conf_label, el_conf_val, elevation_label, elevation_val = self.create_doa_labels(
            'Elevation', value_font_size, x, x_shift, y_elevation, y_shift)

        return azimuth_label, azimuth_val, az_conf_label, az_conf_val, \
            elevation_label, elevation_val, el_conf_label, el_conf_val

    def create_doa_labels(self, text, value_font_size, x, x_shift, y, y_shift):
        angle_label = Label(self.data_frame, text=text)
        angle_label.config(font=("Arial", 14), fg="#4a4a4a")
        angle_label.place(x=x, y=y)

        conf_label = Label(self.data_frame, text="Confidence")
        conf_label.config(font=("Arial", 14), fg="#4a4a4a")
        conf_label.place(x=x + x_shift, y=y)

        angle_val = Label(self.data_frame, text="-")
        angle_val.config(font=("Arial", value_font_size))
        angle_val.place(x=x, y=y + y_shift)

        conf_val = Label(self.data_frame, text="-")
        conf_val.config(font=("Arial", value_font_size))
        conf_val.place(x=x + x_shift, y=y + y_shift)

        return conf_label, conf_val, angle_label, angle_val

    def run(self):
        C = self.create_canvas()
        az_label, az_val, az_conf_label, az_conf_val, el_label, el_val, el_conf_label, el_conf_val = self.create_labels()
        predictor = Predictor()

        while True:
            predictor.is_active = self.prediction_running

            # Get probabilities from model
            all_confs = np.roll(predictor.az_confidences, UI_RESOLUTION // 2)
            display_confs = [max(group) for group in np.split(all_confs, 360 // UI_RESOLUTION)]

            # Normalize confidences to sum to 1
            total = sum(display_confs)
            if total:
                display_confs /= total
            max_idx = np.argmax(display_confs)

            # Color arcs based on model probabilities
            try:
                self.color_arcs(C, display_confs, max_idx)
            except TclError:
                return

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
