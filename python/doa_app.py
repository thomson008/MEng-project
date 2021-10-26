import copy
import math
from tkinter import *

from utils import UI_RESOLUTION


class DoaApp:
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

    def __init__(self, top):
        self.prediction_running = False
        self.top = top
        self.data_frame, self.circle_frame = self.create_frames()
        self.start_button = Button(self.data_frame, text="Start", command=self.toggle_prediction,
                                   height=3, width=15, font=("Arial", 12), cursor="hand2")
        self.start_button.place(relx=0.5, y=400, anchor=CENTER)
        self.mode_selection = Button(self.data_frame, text="Change mode...", command=self.select_mode,
                                     height=2, width=15, font=("Arial", 12), cursor="hand2")
        self.mode_selection.place(relx=0.5, y=500, anchor=CENTER)

        self.exit_button = Button(self.data_frame, text="Exit", command=self.top.destroy,
                                  height=2, width=15, font=("Arial", 12), cursor="hand2")
        self.exit_button.place(relx=0.5, y=560, anchor=CENTER)

    def select_mode(self):
        self.data_frame.destroy()
        self.circle_frame.destroy()

    def create_canvas(self):
        C = Canvas(self.circle_frame, bg="white", height=self.DIM, width=self.DIM)

        # Create the segmented circle in the middle of the canvas
        for i in range(0, 360, UI_RESOLUTION):
            C.create_arc(self.COORD, start=i - UI_RESOLUTION // 2, extent=UI_RESOLUTION, fill='#ebe8e8',
                         outline='#595959')

            text_R = self.RADIUS + 20
            text_x = self.DIM / 2 + text_R * math.cos(math.radians(i))
            text_y = self.DIM / 2 - text_R * math.sin(math.radians(i))

            C.create_text(text_x, text_y, fill="darkblue", font="Arial 11 bold", text=str(i))

        C.pack(fill='both')

        return C

    def create_frames(self):
        left_frame = LabelFrame(self.top, width=self.WIDTH - self.DIM - 1, height=self.HEIGHT)
        left_frame.pack(side=RIGHT)
        left_frame.pack_propagate(False)

        right_frame = LabelFrame(self.top, width=self.HEIGHT, height=self.HEIGHT)
        right_frame.pack(side=LEFT)
        right_frame.pack_propagate(False)

        return left_frame, right_frame

    def toggle_prediction(self):
        self.prediction_running = not self.prediction_running

        text = 'Stop' if self.prediction_running else 'Start'
        self.start_button.config(text=text, relief=SUNKEN if self.prediction_running else RAISED)
