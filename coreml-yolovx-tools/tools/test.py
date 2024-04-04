import os
import math
import cv2
import torch
import coremltools as ct
from tqdm import tqdm
import numpy as np
from utils import *
from PIL import Image

IMAGE_FOLDER = "./val"

    
def make_grid(nx=20, ny=20):
    yv, xv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2))
  
def sigmoid(a):
    return 1 / (1 + np.exp(-a))
    
def load_image(path, imgsz):
    # resize_to: (Width, Height)
    img = Image.open(path).resize((imgsz, imgsz))
    return img.convert("RGB")
    
def main():
    save_dir = "output/"
    model = ct.models.MLModel("./yolov7_ios.mlmodel")
    # model_16 = ct.models.MLModel("output/models/yolov5-iOS.mlmodel")
    # model_8 = ct.models.MLModel("output/models/yolov5-iOS.mlmodel")
    imgsz, stride = 640, 32
    source = "val"
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    for i, (path, img, im0s, vid_cap) in enumerate(dataset):
        print(im0s.shape, img.shape, img.max(), img.min())
        img = load_image(path, imgsz=imgsz)
        out_dict = model.predict({"image": img})
        if len(out_dict["confidence"]) > 0:
            #img_src = original_size(img)
            classes = out_dict["confidence"]
            boxes = out_dict["coordinates"]
            #boxes = boxes0*255#scale_coords([imgsz,imgsz], boxes0, im0s.shape).round()
            result_img = plot_boxes_cv2(im0s, boxes, classes, class_names=CLASSES)
            #print(boxes1)
            output_name = os.path.join(save_dir, "result{}.png".format(i))
            print("save plot results to %s" % output_name)
            cv2.imwrite(output_name, result_img)
       # else:
        #print("confidence")
        #print(out_dict["confidence"])
        #print("Bounding Box")
        #print(out_dict["coordinates"])
    # Compare models
    # ct.models.neural_network.quantization_utils.compare_models(model, model_16, in_dicts)
    # ct.models.neural_network.quantization_utils.compare_models(model, model_8, in_dicts)
if __name__=="__main__":
    main()

