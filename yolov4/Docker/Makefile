IMAGE=yolo
run:
	docker build -t ${IMAGE} .
in:
	nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app --rm -it ${IMAGE}:latest bash

in2:
	docker run -v /home/hagi:/home/app --rm -it ${IMAGE}:latest bash
