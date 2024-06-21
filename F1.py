import cv2
import numpy as np
from training import get_model, load_trained_model, compile_model

# Load the model once when the module is imported
model = None

def load_model():
    global model
    if model is None:
        print("Loading model...")
        model = get_model()
        compile_model(model)
        load_trained_model(model)
        print("Model loaded successfully.")

# Load frontal face haar cascade
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise IOError("Error: Haarcascade XML file not loaded correctly.")

def apply_filter_F1(frame):
    # Ensure the model is loaded
    load_model()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return frame  # Return the original frame if no faces are detected

    img_copy = cv2.cvtColor(np.copy(frame), cv2.COLOR_BGR2BGRA)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img_copy[y:y + h, x:x + w]

        width_original = roi_gray.shape[1]
        height_original = roi_gray.shape[0]
        img_gray = cv2.resize(roi_gray, (96, 96))
        img_gray = img_gray / 255.0

        img_model = np.reshape(img_gray, (1, 96, 96, 1))
        keypoints = model.predict(img_model)[0]

        x_coords = keypoints[0::2]
        y_coords = keypoints[1::2]

        x_coords_denormalized = (x_coords + 0.5) * width_original
        y_coords_denormalized = (y_coords + 0.5) * height_original

        for i in range(len(x_coords)):
            cv2.circle(roi_color, (int(x_coords_denormalized[i]), int(y_coords_denormalized[i])), 2, (255, 255, 0), -1)

        left_lip_coords = (int(x_coords_denormalized[11]), int(y_coords_denormalized[11]))
        right_lip_coords = (int(x_coords_denormalized[12]), int(y_coords_denormalized[12]))
        top_lip_coords = (int(x_coords_denormalized[13]), int(y_coords_denormalized[13]))
        bottom_lip_coords = (int(x_coords_denormalized[14]), int(y_coords_denormalized[14]))
        left_eye_coords = (int(x_coords_denormalized[3]), int(y_coords_denormalized[3]))
        right_eye_coords = (int(x_coords_denormalized[5]), int(y_coords_denormalized[5]))
        brow_coords = (int(x_coords_denormalized[6]), int(y_coords_denormalized[6]))

        beard_width = right_lip_coords[0] - left_lip_coords[0]
        glasses_width = right_eye_coords[0] - left_eye_coords[0]

        # Beard filter
        santa_filter = cv2.imread('filters/santa_filter.png', -1)
        if santa_filter is not None:
            santa_filter = cv2.resize(santa_filter, (beard_width * 3, 150))
            sw, sh, sc = santa_filter.shape

            for i in range(sw):
                for j in range(sh):
                    if santa_filter[i, j][3] != 0:
                        if 0 <= top_lip_coords[1] + i + y - 20 < img_copy.shape[0] and 0 <= left_lip_coords[0] + j + x - 60 < img_copy.shape[1]:
                            img_copy[top_lip_coords[1] + i + y - 20, left_lip_coords[0] + j + x - 60] = santa_filter[i, j]
        else:
            print("Error: Santa filter image not found.")

        # Hat filter
        hat = cv2.imread('filters/hat2.png', -1)
        if hat is not None:
            hat = cv2.resize(hat, (w, w))
            hw, hh, hc = hat.shape

            for i in range(hw):
                for j in range(hh):
                    if hat[i, j][3] != 0:
                        if 0 <= i + y - brow_coords[1] * 2 < img_copy.shape[0] and 0 <= j + x - left_eye_coords[0] * 1 + 20 < img_copy.shape[1]:
                            img_copy[i + y - brow_coords[1] * 2, j + x - left_eye_coords[0] * 1 + 20] = hat[i, j]
        else:
            print("Error: Hat filter image not found.")

        # Glasses filter
        glasses = cv2.imread('filters/glasses.png', -1)
        if glasses is not None:
            glasses = cv2.resize(glasses, (glasses_width * 2, 150))
            gw, gh, gc = glasses.shape

            for i in range(gw):
                for j in range(gh):
                    if glasses[i, j][3] != 0:
                        if 0 <= brow_coords[1] + i + y - 50 < img_copy.shape[0] and 0 <= left_eye_coords[0] + j + x - 60 < img_copy.shape[1]:
                            img_copy[brow_coords[1] + i + y - 50, left_eye_coords[0] + j + x - 60] = glasses[i, j]
        else:
            print("Error: Glasses filter image not found.")

    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGRA2BGR)

    return img_copy

# Example usage
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = apply_filter_F1(frame)

        cv2.imshow('Face Filter', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
