import numpy as np
import cv2

def apply_filter_blueeye(frame):
    # Load your trained model and filter image
    # Apply your filter similar to the logic in demo.py
    # This is just a placeholder for demonstration
    # Replace with actual filter logic
    return frame

# Load the frontal face haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the eye haarcascade
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the blue eye filter
blue_eye_filter = cv2.imread('filters/BlueEye.png', -1)

# Check if the blue eye filter is loaded correctly
if blue_eye_filter is None:
    print("Error: Could not load BlueEye.png.")
    exit()

# Get webcam
camera = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam...")
# Run the program infinitely
while True:
    ret, img = camera.read()  # Read data from the webcam

    # Check if the frame was captured properly
    if not ret:
        print("Error: Could not read frame.")
        break

    # Preprocess input from webcam
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert RGB data to Grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Identify faces in the webcam

    # For each detected face using the Haar cascade
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # For each detected eye
        for (ex, ey, ew, eh) in eyes:
            # Resize the blue eye filter to match the dimensions of the eye area
            blue_eye_filter_resized = cv2.resize(blue_eye_filter, (ew, eh))

            # Overlay the blue eye filter onto the eye area
            for i in range(blue_eye_filter_resized.shape[0]):
                for j in range(blue_eye_filter_resized.shape[1]):
                    # Check if the pixel is within the bounds of the eye area and the filter image
                    if ey + i < h and ex + j < w and blue_eye_filter_resized[i, j][3] != 0:
                        img[y + ey + i, x + ex + j, :] = blue_eye_filter_resized[i, j, :3]

    # Display the webcam feed with the blue eye filter applied
    cv2.imshow('Webcam with Blue Eye Filter', img)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
