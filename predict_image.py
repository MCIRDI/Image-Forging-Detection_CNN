import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ======================
# CONFIG
# ======================
IMG_SIZE = 128
MODEL_PATH = "forgery_detector_casia.h5"

# ======================
# LOAD MODEL
# ======================
model = load_model(MODEL_PATH)
print("[INFO] Model loaded")

# ======================
# PREDICT FUNCTION
# ======================
def predict_image(path):
    img = cv2.imread(path)
    if img is None:
        print("Image not found:", path)
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img.astype("float32"))
    img = np.expand_dims(img, axis=0)

    p = model.predict(img)[0][0]
    prob = float(p)

    if prob > 0.5:
        print(f"TAMPERED (prob={prob:.3f})")
    else:
        print(f"AUTHENTIC (prob={1-prob:.3f})")

# ======================
# RUN TEST
# ======================
predict_image("CASIA2/Au/Au_ani_00002.jpg")

# or:
# predict_image("CASIA2/Tp/CArt00076_10290.tif")
