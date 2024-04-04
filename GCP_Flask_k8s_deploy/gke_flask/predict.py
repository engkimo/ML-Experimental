from google.cloud import storage
import tempfile
import cv2, os
import numpy as np
from gke_flask.model import load_model

client = storage.Client()
bucket_name = 'output-aiplatform'
folder_name = 'GCS_FOLDER'
file_name= 'sonar_model.h5'

def from_gcs():
    blobs = list(client.list_blobs(bucket_name, prefix=folder_name))
    blob = blobs[0]
    _, _ext = os.path.splitext(blob.name)
    _, temp_local_filename = tempfile.mkstemp(suffix=_ext)
    blob.download_to_filename(temp_local_filename)
    return temp_local_filename



def predict(image):
    model = load_model()
    weights_path = from_gcs()
    model.load_weights(weights_path)
    image = np.expand_dims(image, axis=0)
    output = model.predict(image)
    pred_idx = np.argmax(output)
    return pred_idx
