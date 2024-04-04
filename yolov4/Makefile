WEIGHT_PATH='checkpoints/Yolov4_epoch116.pth'
TINY_PATH='tiny_utils/checkpoints/Yolov4_epoch76.pth'
IMG='test2.jpg'
CLASSES=1
yolorun:
	python3 yolo_train.py -l 0.001 -g 0 -pretrained weights/yolov4.conv.137.pth -classes 1 -dir Ds

tinyrun:
	python3 tiny_train.py -l 0.001 -g 0 -pretrained weights/yolov4-tiny.conv.29 -classes 1 -dir Ds

yolopred:
	python3 models.py ${CLASSES} ${WEIGHT_PATH} ${IMG} 416 416

tinypred:
	python3 darknet_prediction.py -cfgfile cfg/yolov4-tiny.cfg -weightfile ${TINY_PATH} -imgfile ${IMG}
