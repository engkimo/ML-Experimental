import datetime
from google.cloud import storage
import tempfile
import os
import numpy as np
import tempfile
import model
client = storage.Client()
bucket_name = 'output-aiplatform'
folder_name = 'sonar_20210323_084454'
file_name= 'sonar_model.h5'
blobs = list(client.list_blobs(bucket_name, prefix=folder_name))
blob = blobs[0]
_, _ext = os.path.splitext(blob.name)
_, temp_local_filename = tempfile.mkstemp(suffix=_ext)
blob.download_to_filename(temp_local_filename)
print(temp_local_filename)
sonar_model = model.sonar_model()
sonar_model.load_weights(temp_local_filename)
sonar_model.summary()
