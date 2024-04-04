WEIGHT_PATH='Yolov4_epoch116.pth'
TINY_PATH='Yolov4_epoch75.pth'
IMG='test4.jpg'
CLASSES=1
BATCH_SIZE=1
yolotorch2onnx:
	python3 yolov4_pytorch2onnx.py ${WEIGHT_PATH} ${IMG} ${BATCH_SIZE} ${CLASSES} 416 416

tinytorch2onnx:
	python3 tiny_pytorch2onnx.py ${TINY_PATH} ${IMG} ${BATCH_SIZE} ${CLASSES} 416 416

yolodarknet2onnx:
	python3 demo_darknet2onnx.py ../cfg/src-yolov4.cfg ../weights/yolov4.weights dog.jpg ${BATCH_SIZE}

tinydarknet2onnx:
	python3 demo_darknet2onnx.py ../cfg/src-yolov4-tiny.cfg ../weights/yolov4-tiny.weights dog.jpg ${BATCH_SIZE}
