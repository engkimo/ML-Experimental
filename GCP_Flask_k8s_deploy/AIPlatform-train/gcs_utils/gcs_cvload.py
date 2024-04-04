import tempfile
import os
import cv2
import numpy as np
from google.cloud import storage
client = storage.Client()
    
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
        if len(images)==400:
            c += 1
        elif len(images)==600:
            c += 1
        labels.append(c)
        #print(f"Blob {blob_name} downloaded to {temp_local_filename}.")
    return np.array(images), np.array(labels)
    
def main():
    bucket_name = "mlops-test-bakura"
    folder_name = "right"
    imgs, labels = download(bucket_name, folder_name)
    print(imgs.shape, labels.shape)
    
    
if __name__=='__main__':
    main()
