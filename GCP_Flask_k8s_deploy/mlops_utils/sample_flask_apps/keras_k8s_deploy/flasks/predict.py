import cv2
import numpy as np
from flasks.model import load_model

def predict(image):
    model = load_model()
    image = np.expand_dims(image, axis=0)
    output = model.predict(image)
    pred_idx = np.argmax(output)
    return pred_idx
