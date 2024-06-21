import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from F1 import apply_filter_F1
from F2 import FilterF2

class FaceFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Filter App")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

        self.video_frame = ttk.Label(root)
        self.video_frame.grid(row=0, column=0, columnspan=2)

        self.btn_demo_filter = ttk.Button(root, text="Apply santa Filter", command=self.apply_demo_filter)
        self.btn_demo_filter.grid(row=1, column=0, pady=10)

        self.btn_blueeye_filter = ttk.Button(root, text="Apply Red Eye Filter", command=self.apply_blueeye_filter)
        self.btn_blueeye_filter.grid(row=1, column=1, pady=10)

        self.current_filter = None
        self.filter_f2 = FilterF2()
        self.show_frame()

    def show_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.current_filter:
            frame = self.current_filter(frame)

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)

        self.root.after(10, self.show_frame)

    def apply_demo_filter(self):
        self.current_filter = apply_filter_F1

    def apply_blueeye_filter(self):
        self.current_filter = self.filter_f2.apply_filter

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceFilterApp(root)
    root.mainloop()
