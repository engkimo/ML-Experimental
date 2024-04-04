# Versions
```
- python 3.7.0
- pytorch 1.7.1+cu101
- torchvision 0.8.2+cu101
- opencv 3.4.4
- onnx 1.6.0
- onnxruntime 1.10.0
- cuda 10.1
- NVIDIA Driver Version: 430.64
```

# Absract
Yolov4 and Yolov4-tiny which can get good accuracy from training. 
These model are base-body model for more high performance model with custmized.
It also can be converted as onnx.

<img src="https://user-images.githubusercontent.com/48679574/145718477-f56d9fdc-8ff5-44be-89a8-428de7787b64.png" width="400px">


## download weights
- [yolov4 (Pytorch)：yolov4.conv.137.pth](https://drive.google.com/open?id=1fcbR0bWzYfIEdLJPzOsn4R5mlvR6IQyA)
- [yolov4.pth](https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ)
- [yolov4-tiny (Pytorch)：yolov4-tiny.conv.29](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29)
- [yolov4 (Darknet)：yolov4.weights](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwj74fGktd70AhVWk1YBHe9bDjMQFnoECAQQAQ&url=https%3A%2F%2Fgithub.com%2FAlexeyAB%2Fdarknet%2Freleases%2Fdownload%2Fdarknet_yolo_v3_optimal%2Fyolov4.weights&usg=AOvVaw30if4joxtTaS8DAh12vYQ4)
- [yolov4-tiny (Darknet)：yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)

## data format
you can download sample datset (google open image datasets) by [OIDv4_ToolKit](https://github.com/EscVM/OIDv4_ToolKit)
```
# data format (data/train.txt)
image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
...
```

# Results
After put trained weights into weights folder, you can train yolov4 and yolov4-tiny via Makefile. 
Yolov4 train is ```make yolorun```. Yolov4-tiny train is ```make tinyrun```.
```
num class is 1 (Coin)
train images : 1225
validation images : 54
test images : 152
```

## yolov4 results

<img src="https://user-images.githubusercontent.com/48679574/145761032-264e07fc-a5c5-4048-87ce-41c52dc97a74.png" width="700px">

<img src="https://user-images.githubusercontent.com/48679574/145761408-40725671-25c7-4ec3-843e-7cfd4a4788ec.jpg" width="400px"><img src="https://user-images.githubusercontent.com/48679574/145761240-ffeb4dfc-d03c-41c1-9883-ce2f07c0f075.jpg" width="400px">


## yolov4-tiny results

<img src="https://user-images.githubusercontent.com/48679574/145761025-667ba7e1-9c2a-461f-9ffc-82a3bbcaeca4.png" width="700px">

<img src="https://user-images.githubusercontent.com/48679574/145761408-40725671-25c7-4ec3-843e-7cfd4a4788ec.jpg" width="400px"><img src="https://user-images.githubusercontent.com/48679574/145761413-c5668684-f335-4c65-9735-79fcb10e7b09.jpg" width="400px">


# Note: cfg parameter when original class dataset training
if num_class is 1, convolutional-filters is 18 (=```(5+num_class)*3```)

```cfg
[convolutional]
size=1
stride=1
pad=1
filters=18
activation=linear

[yolo]
mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=1
```

# References
- [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)
- [OIDv4_ToolKit](https://github.com/EscVM/OIDv4_ToolKit)
- [Google Open Images Dataset V6 +](https://storage.googleapis.com/openimages/web/index.html)
