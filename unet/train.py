import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from utils.data_loading import BasicDataset


def get_dictionaries_list(dir):
    dict_list = os.listdir(dir)
    dict = []
    for item in dict_list:
        dict.append(os.path.join(dir,item))
    return dict


def create_dict(img_list, mask_list):
    dict = []
    for img, mask in zip(img_list, mask_list):
        dict.append({'image': img, 'mask': mask})
    return dict


img_dir = '/Users/mengsiyue/PycharmProjects/train400/images'
mask_dir_1 = '/Users/mengsiyue/PycharmProjects/train400/masks/cat1'


dataset = BasicDataset(img_dir, mask_dir_1)


