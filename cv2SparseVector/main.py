import argparse
import os
import numpy as np

import cv2
import matplotlib.pyplot as plt
from skimage import color

from utils import take_entropy, opening, dilation
from sparse_quantizer import quantize

def vector2raster(img, plot):
    rgb_raster = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab_raster = color.rgb2lab(rgb_raster) 
    if plot:
        ent = take_entropy(rgb_raster)
        print("entropy is ", ent)
        plt.imshow(rgb_raster) 
        plt.show()
    return lab_raster

def sparse_rasters(lab_raster, smooth=None, plot=None):
    q_raster, meta_vector = quantize(lab_raster, n_colors=3)
    sparse_raster = (color.lab2rgb(q_raster) * 255).astype('uint8')
    if smooth:
        sparse_raster = opening(sparse_raster)
        sparse_raster = dilation(sparse_raster)
    if plot:
        entropy_ = take_entropy(sparse_raster)
        print("entropy is ", entropy_)
        plt.imshow(sparse_raster) 
        plt.show()
    return sparse_raster, meta_vector

def main(opt):
    plot = True if opt.plot else False
    smooth = True if opt.smooth else False
    image = cv2.imread(opt.input_path)
    lab_raster = vector2raster(image, plot)
    sparse_raster, meta_raster = sparse_rasters(lab_raster, smooth=smooth, plot=plot)
    bgr_vector = cv2.cvtColor(sparse_raster, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(opt.output_name, ".png"), bgr_vector)
    np.save(opt.output_name, meta_raster)
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="data/input_lenna.png", help='input image path')
    parser.add_argument('--output_name', type=str, default="data/output_lenna", help='output save name')
    parser.add_argument('--plot', action='store_true', help='plot image and calcurate entropy')
    parser.add_argument('--smooth', action='store_true', help='sparse iage become smmothy')
    opt = parser.parse_args()
    main(opt)
    
