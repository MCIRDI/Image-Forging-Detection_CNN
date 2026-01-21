import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# ======================
# CONFIG
# ======================
DATA_DIR = "CASIA2"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

# ======================
# LOAD DATA
# ======================
print("[INFO] Loading images...")

X = []
y = []

for label, folder in enumerate(["Au", "Tp"]):
    path = os.path.join(DATA_DIR, folder)
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label)

X = np.array(X, dtype="float32")
y = np.array(y)

print("Total images:", len(X))

# Preprocess for MobileNetV2
X = preprocess_input(X)

# ======================
# SPLIT DATA
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# BUILD MODEL (TRANSFER LEARNING)
# ======================
print("[INFO] Building model...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================
# TRAIN
# ======================
print("[INFO] Training...")

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ======================
# EVALUATE
# ======================
print("[INFO] Evaluating...")

loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

y_pred = (model.predict(X_test) > 0.5).astype("int")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Authentic", "Tampered"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ======================
# SAVE MODEL
# ======================
model.save("forgery_detector_casia.h5")
print("[INFO] Model saved as forgery_detector_casia.h5")

# ======================
# PLOT
# ======================
plt.figure()
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Loss")
plt.legend()
plt.show()

