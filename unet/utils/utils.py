import glob
import os
import matplotlib.pyplot as plt

def get_dictionaries_list(dir):
    dict_list = os.listdir(dir)
    dict = []
    for item in dict_list:
        dict.appen(os.path.join(dir,item))
    return dict

def create_dict(img_dir, mask_dir):
    img_list = get_dictionaries_list(img_dir)
    mask_list = get_dictionaries_list(mask_dir)
    dict=[]
    for img, mask in zip(img_list ,mask_list):
        dict.append({'image':img, 'mask': mask})
    return dict


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[i, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()
