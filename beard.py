import numpy as np
from training import get_model, load_trained_model, compile_model
import cv2

def load_model():
    print("Loading model...")
    model = get_model()
    compile_model(model)
    load_trained_model(model)
    print("Model loaded and compiled.")
    return model

def load_cascade(file_path):
    face_cascade = cv2.CascadeClassifier(file_path)
    if face_cascade.empty():
        raise IOError("Error: Haarcascade XML file not loaded correctly.")
    return face_cascade

def initialize_camera():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise IOError("Error: Could not open webcam.")
    return camera

def detect_faces(face_cascade, gray_frame):
    return face_cascade.detectMultiScale(gray_frame, 1.3, 5)

def preprocess_face(roi_gray):
    img_gray = cv2.resize(roi_gray, (96, 96))  # Resize to model input size
    img_gray = img_gray / 255.0  # Normalize the image data
    return np.reshape(img_gray, (1, 96, 96, 1))

def denormalize_keypoints(keypoints, width_original, height_original):
    x_coords = keypoints[0::2]  # Read alternate elements starting from index 0
    y_coords = keypoints[1::2]  # Read alternate elements starting from index 1
    x_coords_denormalized = (x_coords + 0.5) * width_original  # Denormalize x-coordinate
    y_coords_denormalized = (y_coords + 0.5) * height_original  # Denormalize y-coordinate
    return x_coords_denormalized, y_coords_denormalized

def apply_filter(image, filter_img, position, size):
    if filter_img is not None:
        filter_img = cv2.resize(filter_img, size)
        fw, fh, fc = filter_img.shape

        for i in range(fw):
            for j in range(fh):
                if filter_img[i, j][3] != 0:
                    if 0 <= position[1] + i < image.shape[0] and 0 <= position[0] + j < image.shape[1]:
                        image[position[1] + i, position[0] + j] = filter_img[i, j]
    else:
        print("Error: Filter image not found.")
    return image

def main():
    model = load_model()
    face_cascade = load_cascade('cascades/haarcascade_frontalface_default.xml')
    camera = initialize_camera()

    print("Starting webcam...")
    while True:
        ret, img = camera.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(face_cascade, gray)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            img_copy = np.copy(img)

            width_original = roi_gray.shape[1]
            height_original = roi_gray.shape[0]
            img_model = preprocess_face(roi_gray)
            keypoints = model.predict(img_model)[0]
            x_coords_denormalized, y_coords_denormalized = denormalize_keypoints(keypoints, width_original, height_original)

            left_lip_coords = (int(x_coords_denormalized[11]), int(y_coords_denormalized[11]))
            right_lip_coords = (int(x_coords_denormalized[12]), int(y_coords_denormalized[12]))
            top_lip_coords = (int(x_coords_denormalized[13]), int(y_coords_denormalized[13]))
            left_eye_coords = (int(x_coords_denormalized[3]), int(y_coords_denormalized[3]))
            right_eye_coords = (int(x_coords_denormalized[5]), int(y_coords_denormalized[5]))
            brow_coords = (int(x_coords_denormalized[6]), int(y_coords_denormalized[6]))

            beard_width = right_lip_coords[0] - left_lip_coords[0]
            glasses_width = right_eye_coords[0] - left_eye_coords[0]

            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2BGRA)

            # Beard filter
            santa_filter = cv2.imread('filters/b1.png', -1)
            img_copy = apply_filter(img_copy, santa_filter, (left_lip_coords[0] + x - 60, top_lip_coords[1] + y - 20), (beard_width * 3, 150))

            # Hat filter
            hat = cv2.imread('filters/hat2.png', -1)
            img_copy = apply_filter(img_copy, hat, (x - left_eye_coords[0] * 1 + 20, y - brow_coords[1] * 2), (w, w))

            # Glasses filter
            glasses = cv2.imread('filters/glass1.png', -1)
            img_copy = apply_filter(img_copy, glasses, (left_eye_coords[0] + x - 60, brow_coords[1] + y - 50), (glasses_width * 2, 150))

            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGRA2BGR)

            cv2.imshow('Output', img_copy)

        if cv2.waitKey(1) & 0xFF == ord("e"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")

if __name__ == "__main__":
    main()
