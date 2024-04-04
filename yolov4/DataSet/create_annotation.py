import sys
import json
from tqdm import tqdm
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob

def npint(cls):
    return int(float(cls))

def open_txtfile(label_path):
    truth = {}
    f = open(label_path, 'r', encoding='utf-8')
    for line in f.readlines():
        data = line.split(" ")
        truth[data[0]] = []
        for i in data[1:]:
            truth[data[0]].append([int(float(j)) for j in i.split(',')])
            print(truth)
        if i==3:
            break
                      

def main(save_filetxt_name, data_type, classes):
    classes_dicts = {key:idx for idx, key in enumerate(classes)}
    with open(save_filetxt_name, 'w') as f:
        for cls in classes:
            IMG_PATH = os.path.join(cls, data_type, cls)
            image_dir_path = os.path.join(cls, data_type, cls, '*.jpg')
            label_path = os.path.join(cls, data_type, cls, 'Label', '*.txt')
            imgid = 0
            for path, imgpath in tqdm(zip(sorted(glob.glob(label_path)), sorted(glob.glob(image_dir_path)))):
                p1 = os.path.splitext(os.path.basename(path))[0]
                p2 = os.path.splitext(os.path.basename(imgpath))[0]
                if 'level' in p2:
                    break
                assert p1==p2, 'must same name'
                name = os.path.join(IMG_PATH, 'level{}_'.format(imgid)+'{:012d}.jpg'.format(imgid))
                p1_ = os.path.join(IMG_PATH, p1+'.jpg')
                if name!=p1_:
                    os.rename(p1_, name) 
                f.write(name)
                annof = open(os.path.join(path), 'r', encoding='utf-8')
                for line in annof.readlines():
                    cls, x_min, y_min, x_max, y_max = line.split(" ")
                    clsid = classes_dicts[cls]
                    x_min, y_min, x_max, y_max = npint(x_min), npint(y_min), npint(x_max), npint(y_max)

                    box_info = " %d,%d,%d,%d,%d" % (
                    x_min, y_min, x_max, y_max, int(clsid))
                    f.write(box_info)
                f.write('\n')
                imgid += 1
                
                
if __name__=='__main__':
    data_type = sys.argv[1]
    classes = ['Coin']
    assert data_type in ['train', 'validation', 'test'], 'corecct word from [train, validation, test]'
    save_filetxt_name = os.path.join('../data', '{}.txt'.format(data_type))
    main(save_filetxt_name, data_type, classes)
    print('try to open')
    #open_txtfile(save_filetxt_name)

