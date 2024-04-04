import numpy as np
import cv2

def take_entropy(img):
    marg = np.histogramdd(np.ravel(img), bins = 256)[0]/img.size
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    entropy = -np.sum(np.multiply(marg, np.log2(marg)))
    return entropy

def opening(img):
    kernel = np.ones((3,3), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def dilation(img):
    kernel = np.ones((3,3), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)
