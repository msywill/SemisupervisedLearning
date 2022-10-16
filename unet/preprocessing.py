import os
import cv2 as cv
import imagehash
import numpy as np
from PIL import Image
from PIL import ImageChops
from pathlib import Path


def test(dir):
    num = 0
    save_path = '/Users/mengsiyue/PycharmProjects/'
    image_list = os.listdir(dir)
    for image_path in image_list:
        img = Image.open(dir+'/'+image_path)
        png_name = save_path+str(num)+'.png'
        img.save(png_name)
        num +=1
# test(dir)
#img = Image.open('/Users/mengsiyue/PycharmProjects/10.png')

#img_jpg = Image.open('/Users/mengsiyue/PycharmProjects/remotefiles/HighToLowFullTrain/trainB/0.png')

test_tif = '/Users/mengsiyue/PycharmProjects/Master_Arbeit/data/img1200.tif'
img = cv.imread(test_tif,1)
cv.imwrite('/Users/mengsiyue/PycharmProjects/img1200.png',img)

#test_tif.save('/Users/mengsiyue/PycharmProjects/img.png')

path_one = '/Users/mengsiyue/PycharmProjects/remotefiles/10_sequences_trackformer/train/s1/1199.png'
path_two = '/Users/mengsiyue/PycharmProjects/img1200.png'
def compare(path_one, path_two):
    image_one = Image.open(path_one)
    image_two = Image.open(path_two)
    print (np.any((np.asarray(image_two) - np.asarray(image_one)))) #imageone是3通道的




compare(path_one, path_two)



