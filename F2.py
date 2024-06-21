import os
import numpy as np
import cv2
import pygame

class FilterF2:
    def __init__(self):
        # Initialize pygame mixer
        pygame.mixer.init()

        # Load the laser sound
        laser_sound_path = os.path.join(os.path.dirname(__file__), 's.mp3')
        print(f"Checking for {laser_sound_path} at {os.path.abspath(laser_sound_path)}")
        if not os.path.exists(laser_sound_path):
            raise FileNotFoundError(f"Error: Could not load {laser_sound_path}.")
        self.laser_sound = pygame.mixer.Sound(laser_sound_path)

        # Load the frontal face haarcascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Load the eye haarcascade
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Define the path to the blue eye filter image
        blue_eye_filter_path = os.path.join(os.path.dirname(__file__), 'filters', 'e1.png')
        if not os.path.exists(blue_eye_filter_path):
            raise FileNotFoundError(f"Error: Could not load {blue_eye_filter_path}.")
        self.blue_eye_filter = cv2.imread(blue_eye_filter_path, -1)
        if self.blue_eye_filter is None:
            raise IOError(f"Error: Could not load {blue_eye_filter_path}.")

        # Convert the blue eye filter from BGR to RGB
        self.blue_eye_filter = cv2.cvtColor(self.blue_eye_filter, cv2.COLOR_BGRA2RGBA)

        # Define the factor to increase the width of the filter
        self.width_increase_factor = 1.5

    def apply_filter(self, frame):
        try:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            print(f"Error converting frame to grayscale: {e}")
            return frame

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                new_width = int(ew * self.width_increase_factor)
                blue_eye_filter_resized = cv2.resize(self.blue_eye_filter, (new_width, eh))

                start_x = ex - (new_width - ew) // 2

                for i in range(blue_eye_filter_resized.shape[0]):
                    for j in range(blue_eye_filter_resized.shape[1]):
                        if ey + i < h and start_x + j < w and start_x + j >= 0 and blue_eye_filter_resized[i, j][3] != 0:
                            roi_color[ey + i, start_x + j, :] = blue_eye_filter_resized[i, j, :3]

                self.laser_sound.play()

        return frame
