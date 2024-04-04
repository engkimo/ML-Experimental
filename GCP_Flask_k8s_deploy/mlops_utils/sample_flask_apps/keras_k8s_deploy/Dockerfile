FROM tensorflow/tensorflow:devel-gpu

RUN mkdir /home/app
WORKDIR /home/app/

COPY requirement.txt /home/app/
RUN pip install --upgrade pip
RUN pip install -r /home/app/requirement.txt

ADD server.py /home/app/
ADD procfile /home/app/
ADD flasks /home/app/flasks

#CMD ["python", "server.py"]
CMD honcho start -f /home/app/procfile $PROCESSES
