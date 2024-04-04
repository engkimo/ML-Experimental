FROM tensorflow/tensorflow:nightly
WORKDIR /root
ENV DEBIAN_FRONTEND=noninteractive


RUN pip install pandas google-cloud-storage scikit-learn
RUN apt-get install -y python-opencv python3-opencv

# Install curl
RUN apt-get update; apt-get install curl -y
# The data for this sample has been publicly hosted on a GCS bucket.
# Download the data from the public Google Cloud Storage bucket for this sample
RUN curl https://storage.googleapis.com/cloud-samples-data/ml-engine/sonar/sonar.all-data --output ./sonar.all-data

COPY model.py ./model.py
COPY data_utils.py ./data_utils.py
COPY task.py ./task.py
# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "task.py"]
