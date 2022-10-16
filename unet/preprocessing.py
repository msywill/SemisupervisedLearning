import os
import cv2 as cv
import numpy as np
from PIL import Image


cat3 = '/Users/mengsiyue/PycharmProjects/train400/masks/200-400/cat3'

def rename(dir):
    i = 200
    l = sorted(os.listdir(dir))
    l.sort(key=lambda x: int(x.split('.')[0]))
    for item in l:  # 进入到文件夹内，对每个文件进行循环遍历
        #print(os.path.join(dir, item))
        os.rename(os.path.join(dir, item), os.path.join(dir, (str(i) + '.png')))
        #print(os.path.join(dir, (str(i) + '.png')))
        i += 1

rename(cat3)

def convert_tif_to_png(dir):
    num = 0
    mask_list = os.listdir(dir)
    mask_list.sort(key=lambda x: int(x.split('.')[0][3:]))
    save_path_pre = '/Users/mengsiyue/PycharmProjects/full_high_masks/data2/cat3/'
    for mask_path in mask_list:
        print("processing image: ", mask_path)
        img = cv.imread(os.path.join(dir,mask_path), 1)
        img_png_path = os.path.join(save_path_pre, (str(num)+'.png'))
        cv.imwrite(img_png_path, img)
        num +=1
