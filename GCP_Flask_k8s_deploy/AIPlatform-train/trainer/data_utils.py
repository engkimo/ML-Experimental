import datetime
from google.cloud import storage
import tempfile
import os, cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.utils import to_categorical


client = storage.Client()
BUCKET_NAME = "mlops-test-bakura"
FOLDER_NAME = "right"
NUM_CLS = 2+1


def load_label(y, num_classes=3):
    return to_categorical(y, num_classes=num_classes)
    
    
def download(bucket_name, folder_name):
    images = []
    labels = []
    c = 0
    for blob in client.list_blobs(bucket_name, prefix=folder_name):
        _, _ext = os.path.splitext(blob.name)
        _, temp_local_filename = tempfile.mkstemp(suffix=_ext)
        blob.download_to_filename(temp_local_filename)
        img = cv2.imread(temp_local_filename)
        images.append(cv2.resize(img, (224, 224)))
        if len(images)==200:
            c += 1
        elif len(images)==400:
            c += 1
        labels.append(c)
        #print(f"Blob {blob_name} downloaded to {temp_local_filename}.")
    return np.array(images)/255, np.array(labels)
    
    
def load_data(args):
    imgs, labels = download(BUCKET_NAME, FOLDER_NAME)
    labels = load_label(labels, num_classes=NUM_CLS)
    print(imgs.shape, labels.shape)
    train_f, test_f, train_l, test_l = train_test_split(
            imgs, labels, test_size=args.test_split, random_state=args.seed)
    return train_f, test_f, train_l, test_l
    
    
def save_model(model_dir, model_name):
    """Saves the model to Google Cloud Storage"""
    bucket = storage.Client().bucket(model_dir)
    blob = bucket.blob('{}/{}'.format(
        datetime.datetime.now().strftime('sonar_%Y%m%d_%H%M%S'),
        model_name))
    blob.upload_from_filename(model_name)
