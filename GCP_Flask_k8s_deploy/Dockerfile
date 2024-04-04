FROM tensorflow/tensorflow:devel-gpu

RUN mkdir /home/app
WORKDIR /home/app/
RUN apt-get update; apt-get install curl -y

COPY requirement.txt /home/app/
RUN pip install --upgrade pip
RUN pip install -r /home/app/requirement.txt

ADD main.py /home/app/
ADD gke_flask /home/app/gke_flask
ADD procfile /home/app/

CMD honcho start -f /home/app/procfile $PROCESSES
#CMD ["python", "main.py"]
