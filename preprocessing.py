import os
import shutil
import nibabel as nib
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from pydicom import dcmread
import glob
from pathlib import Path
from scipy.ndimage import rotate
from tqdm import tqdm
import random
import config


def get_filename(dir):
    path_absolute = Path(dir).absolute()
    return path_absolute.name


def get_patient_id_labels(dir):
    filename = get_filename(dir)
    return filename.split('.')[0][-2:]


def labels_preprocessing(parent_dir):
    pancreas_list = []
    src_dir = parent_dir + '/TCIA_pancreas_labels-02-05-2017/*.nii.gz'

    for src in glob.iglob(src_dir):
        patient_id = get_patient_id_labels(src)
        path = ""

        if patient_id == "25" or patient_id == "70" or patient_id == "80":
            os.remove(src)
            print("patient " + patient_id + " removed")
            continue
        elif int(patient_id) in config.test_indices:
            path = parent_dir + "/test/PATIENT_00" + patient_id + "/labels"
            os.makedirs(path)
        elif int(patient_id) in config.val_indices:
            path = parent_dir + "/val/PATIENT_00" + patient_id + "/labels"
            os.makedirs(path)
        elif int(patient_id) in config.train_indices:
            path = parent_dir + "/train/PATIENT_00" + patient_id + "/labels"
            os.makedirs(path)

        # split nifti into slices
        # array of shape (512, 512, #slices)
        nifti = nib.load(src).get_fdata()
        # list of length #slices containing all slices of one patient as arrays of shape (512, 512, 1)
        slices = np.dsplit(nifti, nifti.shape[-1])
        for slice_index, slice in enumerate(tqdm(slices)):
            slice_index += 1
            # list of length #slices containing arrays of shape (512, 512)
            npy_array = np.squeeze(slice)

            if slice_index < 10:
                prefix = "00"
            elif slice_index < 100:
                prefix = "0"
            else:
                prefix = ""

            # abandone labels without pancreas for train and val and create pancreas list
            if int(patient_id) not in config.test_indices:
                if npy_array.max() == 0:
                    continue
                pancreas_list.append(patient_id + "-" + prefix + str(slice_index))

            # the following is performed for all slices in the test set & for all slice with pancreas in the train set and val set
            npy_array_rotated = rotate(npy_array, 90)
            file_name = "L-" + patient_id + "-" + prefix + str(slice_index)
            dst_path = os.path.join(path, file_name)
            np.savez_compressed(dst_path, npy_array_rotated.astype(int))

        # delete original nifti file
        os.remove(src)
        print("patient " + patient_id + " processed")

    # delete old data path
    shutil.rmtree(parent_dir + "/TCIA_pancreas_labels-02-05-2017", ignore_errors=True)

    return pancreas_list


def image_preprocessing(parent_dir, pancreas_list):
    image_dir = parent_dir + '/manifest-1599750808610/Pancreas-CT'
    patient_dirs = [os.path.join(image_dir, o) for o in os.listdir(image_dir) if
                    os.path.isdir(os.path.join(image_dir, o))]

    for patient in patient_dirs:
        patient_id = patient[-2:]

        if patient_id == "25" or patient_id == "70" or patient_id == "80":
            continue
        elif int(patient_id) in config.test_indices:
            path = parent_dir + "/test/PATIENT_00" + patient_id + "/images"
            os.makedirs(path)
        elif int(patient_id) in config.val_indices:
            path = parent_dir + "/val/PATIENT_00" + patient_id + "/images"
            os.makedirs(path)
        elif int(patient_id) in config.train_indices:
            path = parent_dir + "/train/PATIENT_00" + patient_id + "/images"
            os.makedirs(path)

        image_paths = patient + '/**/**/*.dcm'
        for image in tqdm(glob.iglob(image_paths)):
            new_filename = patient_id + image[-8:-4]
            if int(patient_id) not in config.test_indices and new_filename not in pancreas_list:
                os.remove(image)
                continue

            dcm = dcmread(image)
            npy_array = np.array(dcm.pixel_array)
            npy_array_rotated = rotate(npy_array, 180)[..., ::-1]
            np.savez_compressed(path + '/' + new_filename, npy_array_rotated)
            os.remove(image)

    # delete old data path
    shutil.rmtree(parent_dir + '/manifest-1599750808610', ignore_errors=True)


def preprocessing(parent_dir):
    pancreas_list = labels_preprocessing(parent_dir)
    image_preprocessing(parent_dir, pancreas_list)


def add_noise(seg_mask):
    case = random.randint(1, 4)

    if case == 1:
        seq = iaa.Sequential([
            iaa.Affine(scale=(0.98, 1.02)),
            iaa.PiecewiseAffine(scale=(0.009, 0.01)),
            iaa.ShearX((-1.5, 1.5)),
            iaa.ShearY((-1.5, 1.5))
         ])
    elif case == 2:
        seq = iaa.Sequential([
            iaa.Affine(scale=(0.95, 1.05)),
            iaa.PiecewiseAffine(scale=(0.009, 0.01)),
        ])
    elif case == 3:
        seq = iaa.Sequential([
            iaa.PiecewiseAffine(scale=(0.009, 0.02)),
            iaa.ShearX((-1.5, 1.5)),
            iaa.ShearY((-1.5, 1.5))
        ])
    elif case == 4:
        seq = iaa.Sequential([
        iaa.Affine(scale=(0.92, 1.08)),
        iaa.ShearX((-2, 2)),
        iaa.ShearY((-2, 2))
        ])

    noisy_mask = np.array(seq(images=seg_mask)).astype(int)
    delta = np.abs(noisy_mask - seg_mask)
    
    show = False
    if show:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(seg_mask, cmap='gray')
        ax1.set_title('original')
        ax2.imshow(delta, cmap='Reds')
        ax2.set_title('delta')
        ax3.imshow(noisy_mask, cmap='gray')
        ax3.set_title('noisy')
        fig.show()

    return noisy_mask


def add_light_noise(seg_mask):
    case = random.randint(1, 4)

    if case == 1:
        seq = iaa.Sequential([
            iaa.Affine(scale=(0.99, 1.01)), # standard  = 0.98, 1.02
            iaa.PiecewiseAffine(scale=(0.005, 0.005)), # standard  = 0.009, 0.01
            iaa.ShearX((-1.0, 1.0)), # standard  = -1.5, 1.5
            iaa.ShearY((-1.0, 1.0)) # standard  = -1.5, 1.5
         ])
    elif case == 2:
        seq = iaa.Sequential([
            iaa.Affine(scale=(0.98, 1.02)), # standard  = 0.95, 1.05
            iaa.PiecewiseAffine(scale=(0.005, 0.005)), # standard  = 0.009, 0.01
        ])
    elif case == 3:
        seq = iaa.Sequential([
            iaa.PiecewiseAffine(scale=(0.005, 0.005)), # standard  = 0.009, 0.02
            iaa.ShearX((-1.0, 1.0)), # standard  = -1.5, 1.5
            iaa.ShearY((-1.0, 1.0)) # standard  = -1.5, 1.5
        ])
    elif case == 4:
        seq = iaa.Sequential([
        iaa.Affine(scale=(0.98, 1.02)), # standard  = 0.92, 1.08
        iaa.ShearX((-1.5, 1.5)), # standard  = -2, 2
        iaa.ShearY((-1.5, 1.5)) # standard  = -2, 2
        ])

    noisy_mask_light = np.array(seq(images=seg_mask)).astype(int)
    delta = np.abs(noisy_mask_light - seg_mask)
    
    show = False
    if show:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(seg_mask, cmap='gray')
        ax1.set_title('original')
        ax2.imshow(delta, cmap='Reds')
        ax2.set_title('delta')
        ax3.imshow(noisy_mask_light, cmap='gray')
        ax3.set_title('noisy')
        fig.show()

    return noisy_mask_light


def add_heavy_noise(seg_mask):
    case = random.randint(1, 4)

    if case == 1:
        seq = iaa.Sequential([
            iaa.Affine(scale=(0.8, 1.2)), # standard  = 0.98, 1.02
            iaa.PiecewiseAffine(scale=(0.01, 0.07)), # standard  = 0.009, 0.01
            iaa.ShearX((-2.5, 2.5)), # standard  = -1.5, 1.5
            iaa.ShearY((-2.5, 2.5)) # standard  = -1.5, 1.5
         ])
    elif case == 2:
        seq = iaa.Sequential([
            iaa.Affine(scale=(0.7, 1.3)), # standard  = 0.95, 1.05
            iaa.PiecewiseAffine(scale=(0.03, 0.09)), # standard  = 0.009, 0.01
        ])
    elif case == 3:
        seq = iaa.Sequential([
            iaa.PiecewiseAffine(scale=(0.03, 0.09)), # standard  = 0.009, 0.02
            iaa.ShearX((-2.5, 2.5)), # standard  = -1.5, 1.5
            iaa.ShearY((-2.5, 2.5)) # standard  = -1.5, 1.5
        ])
    elif case == 4:
        seq = iaa.Sequential([
        iaa.Affine(scale=(0.7, 1.3)), # standard  = 0.92, 1.08
        iaa.ShearX((-3, 3)), # standard  = -2, 2
        iaa.ShearY((-3, 3)) # standard  = -2, 2
        ])

    noisy_mask_heavy = np.array(seq(images=seg_mask)).astype(int)
    delta = np.abs(noisy_mask_heavy - seg_mask)
    
    show = False
    if show:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(seg_mask, cmap='gray')
        ax1.set_title('original')
        ax2.imshow(delta, cmap='Reds')
        ax2.set_title('delta')
        ax3.imshow(noisy_mask_heavy, cmap='gray')
        ax3.set_title('noisy')
        fig.show()

    return noisy_mask_heavy


def add_semiheavy_noise(seg_mask):
    case = random.randint(1, 4)

    if case == 1:
        seq = iaa.Sequential([
            iaa.Affine(scale=(0.9, 1.1)), # standard  = 0.98, 1.02
            iaa.PiecewiseAffine(scale=(0.01, 0.05)), # standard  = 0.009, 0.01
            iaa.ShearX((-2, 2)), # standard  = -1.5, 1.5
            iaa.ShearY((-2, 2)) # standard  = -1.5, 1.5
         ])
    elif case == 2:
        seq = iaa.Sequential([
            iaa.Affine(scale=(0.88, 1.15)), # standard  = 0.95, 1.05
            iaa.PiecewiseAffine(scale=(0.02, 0.06)), # standard  = 0.009, 0.01
        ])
    elif case == 3:
        seq = iaa.Sequential([
            iaa.PiecewiseAffine(scale=(0.02, 0.06)), # standard  = 0.009, 0.02
            iaa.ShearX((-2, 2)), # standard  = -1.5, 1.5
            iaa.ShearY((-2, 2)) # standard  = -1.5, 1.5
        ])
    elif case == 4:
        seq = iaa.Sequential([
        iaa.Affine(scale=(0.88, 1.15)), # standard  = 0.92, 1.08
        iaa.ShearX((-2.5, 2.5)), # standard  = -2, 2
        iaa.ShearY((-2.5, 2.5)) # standard  = -2, 2
        ])

    noisy_mask_heavy = np.array(seq(images=seg_mask)).astype(int)
    delta = np.abs(noisy_mask_heavy - seg_mask)
    
    show = False
    if show:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(seg_mask, cmap='gray')
        ax1.set_title('original')
        ax2.imshow(delta, cmap='Reds')
        ax2.set_title('delta')
        ax3.imshow(noisy_mask_heavy, cmap='gray')
        ax3.set_title('noisy')
        fig.show()

    return noisy_mask_heavy


def compute_evaluation_metrics(highquality_mask, noisy_mask):
    highquality_mask = np.array(highquality_mask).astype(int)

    count_noisy_pixels = int(np.sum(np.abs(noisy_mask - highquality_mask)))

    ratio_gtarea = count_noisy_pixels / np.sum(highquality_mask)

    dice_score = 2. * np.logical_and(noisy_mask, highquality_mask).sum() / (noisy_mask.sum() + highquality_mask.sum())

    return count_noisy_pixels, ratio_gtarea, dice_score


def create_noisy_labels():
    count = 0
    count_noisy_pixel_abs_list = []         # absolut number of changed pixels
    ratio_noisy_pixel_wrt_gtarea_list = []  # ratio of changed pixels : pancreas area
    dice_score_list = []                    # dice score noisy - HQ labels

    for patient_id in range(1, 83):
        path = "data/train/PATIENT_00" + ('0' if patient_id < 10 else '') + str(patient_id) + "/noisy_labels"
        # remove noisy labels if there already exist some
        if os.path.exists(path):
            shutil.rmtree(path)
        if patient_id in (config.hq_indices + config.lq_indices):
            count += 1
            print("Create noisy labels for patient " + str(patient_id) + " (" + str(count) + "/" + str(len(config.hq_indices + config.lq_indices)) + ")")
            if patient_id < 10:
                prefix = '0'
            else:
                prefix = ''
            # create a new subfolder
            os.mkdir(path)

            # iterate over quality label slices and transform to noisy label slices
            for quality_label_slice_path in glob.iglob("data/train/PATIENT_00" + prefix + str(patient_id) + '/labels/*.npz'):
                quality_label_slice = np.load(quality_label_slice_path)['arr_0']
                noisy_label_slice = add_noise(quality_label_slice)
                count_noisy_pixels, ratio_gtarea, dice_score = compute_evaluation_metrics(quality_label_slice, noisy_label_slice)
                count_noisy_pixel_abs_list.append(count_noisy_pixels)
                ratio_noisy_pixel_wrt_gtarea_list.append(ratio_gtarea)
                dice_score_list.append(dice_score)
                slice_index = quality_label_slice_path.split('.')[0][-3:]
                file_name = "L_noisy-" + prefix + str(patient_id) + "-" + str(slice_index)
                dst_path = os.path.join(path, file_name)
                np.savez_compressed(dst_path, noisy_label_slice)

    print('-'*60)
    print("Number pixels changed:")
    print("Mean (of abs. number):", np.round(np.mean(count_noisy_pixel_abs_list), 3), "; ratio wrt to 512 x 512 pixels:", np.round(np.mean(count_noisy_pixel_abs_list)/(512*512), 5))
    print("Var (of abs. number):", np.round(np.var(count_noisy_pixel_abs_list), 3), ";  std (of abs. num):", np.round(np.std(count_noisy_pixel_abs_list), 3))
    print('-'*10)
    print("Number pixels changed wrt pancreas area:")
    print("Mean:", np.round(np.mean(ratio_noisy_pixel_wrt_gtarea_list), 3))
    print("Variance:", np.round(np.var(ratio_noisy_pixel_wrt_gtarea_list), 3))
    print('-'*10)
    print("Dice score (noisy labels - high quality labels):")
    print("Mean:", np.round(np.mean(dice_score_list), 3))
    print("Variance:", np.round(np.var(dice_score_list), 3))


def create_noisy_labels_light():
    count = 0
    count_noisy_pixel_abs_list = []         # absolut number of changed pixels
    ratio_noisy_pixel_wrt_gtarea_list = []  # ratio of changed pixels : pancreas area
    dice_score_list = []                    # dice score noisy - HQ labels

    for patient_id in range(1, 83):
        path = "data/train/PATIENT_00" + ('0' if patient_id < 10 else '') + str(patient_id) + "/noisy_labels_light"
        # remove noisy labels if there already exist some
        if os.path.exists(path):
            shutil.rmtree(path)
        if patient_id in (config.hq_indices + config.lq_indices):
            count += 1
            print("Create light noisy labels for patient " + str(patient_id) + " (" + str(count) + "/" + str(len(config.hq_indices + config.lq_indices)) + ")")
            if patient_id < 10:
                prefix = '0'
            else:
                prefix = ''
            # create a new subfolder
            os.mkdir(path)

            # iterate over quality label slices and transform to noisy label slices
            for quality_label_slice_path in glob.iglob("data/train/PATIENT_00" + prefix + str(patient_id) + '/labels/*.npz'):
                quality_label_slice = np.load(quality_label_slice_path)['arr_0']
                noisy_label_slice = add_light_noise(quality_label_slice)
                count_noisy_pixels, ratio_gtarea, dice_score = compute_evaluation_metrics(quality_label_slice, noisy_label_slice)
                count_noisy_pixel_abs_list.append(count_noisy_pixels)
                ratio_noisy_pixel_wrt_gtarea_list.append(ratio_gtarea)
                dice_score_list.append(dice_score)
                slice_index = quality_label_slice_path.split('.')[0][-3:]
                file_name = "L_noisy_light-" + prefix + str(patient_id) + "-" + str(slice_index)
                dst_path = os.path.join(path, file_name)
                np.savez_compressed(dst_path, noisy_label_slice)

    print('-'*60)
    print("Number pixels changed:")
    print("Mean (of abs. number):", np.round(np.mean(count_noisy_pixel_abs_list), 3), "; ratio wrt to 512 x 512 pixels:", np.round(np.mean(count_noisy_pixel_abs_list)/(512*512), 5))
    print("Var (of abs. number):", np.round(np.var(count_noisy_pixel_abs_list), 3), ";  std (of abs. num):", np.round(np.std(count_noisy_pixel_abs_list), 3))
    print('-'*10)
    print("Number pixels changed wrt pancreas area:")
    print("Mean:", np.round(np.mean(ratio_noisy_pixel_wrt_gtarea_list), 3))
    print("Variance:", np.round(np.var(ratio_noisy_pixel_wrt_gtarea_list), 3))
    print('-'*10)
    print("Dice score (light noisy labels - high quality labels):")
    print("Mean:", np.round(np.mean(dice_score_list), 3))
    print("Variance:", np.round(np.var(dice_score_list), 3))


def create_noisy_labels_heavy():
    count = 0
    count_noisy_pixel_abs_list = []         # absolute number of changed pixels
    ratio_noisy_pixel_wrt_gtarea_list = []  # ratio of changed pixels : pancreas area
    dice_score_list = []                    # dice score noisy - HQ labels

    for patient_id in range(1, 83):
        path = "data/train/PATIENT_00" + ('0' if patient_id < 10 else '') + str(patient_id) + "/noisy_labels_heavy"
        # remove noisy labels if there already exist some
        if os.path.exists(path):
            shutil.rmtree(path)
        if patient_id in (config.hq_indices + config.lq_indices):
            count += 1
            print("Create heavy noisy labels for patient " + str(patient_id) + " (" + str(count) + "/" + str(len(config.hq_indices + config.lq_indices)) + ")")
            if patient_id < 10:
                prefix = '0'
            else:
                prefix = ''
            # create a new subfolder
            os.mkdir(path)

            # iterate over quality label slices and transform to noisy label slices
            for quality_label_slice_path in glob.iglob("data/train/PATIENT_00" + prefix + str(patient_id) + '/labels/*.npz'):
                quality_label_slice = np.load(quality_label_slice_path)['arr_0']
                noisy_label_slice = add_heavy_noise(quality_label_slice)
                count_noisy_pixels, ratio_gtarea, dice_score = compute_evaluation_metrics(quality_label_slice, noisy_label_slice)
                count_noisy_pixel_abs_list.append(count_noisy_pixels)
                ratio_noisy_pixel_wrt_gtarea_list.append(ratio_gtarea)
                dice_score_list.append(dice_score)
                slice_index = quality_label_slice_path.split('.')[0][-3:]
                file_name = "L_noisy_heavy-" + prefix + str(patient_id) + "-" + str(slice_index)
                dst_path = os.path.join(path, file_name)
                np.savez_compressed(dst_path, noisy_label_slice)

    print('-'*60)
    print("Number pixels changed:")
    print("Mean (of abs. number):", np.round(np.mean(count_noisy_pixel_abs_list), 3), "; ratio wrt to 512 x 512 pixels:", np.round(np.mean(count_noisy_pixel_abs_list)/(512*512), 5))
    print("Var (of abs. number):", np.round(np.var(count_noisy_pixel_abs_list), 3), ";  std (of abs. num):", np.round(np.std(count_noisy_pixel_abs_list), 3))
    print('-'*10)
    print("Number pixels changed wrt pancreas area:")
    print("Mean:", np.round(np.mean(ratio_noisy_pixel_wrt_gtarea_list), 3))
    print("Variance:", np.round(np.var(ratio_noisy_pixel_wrt_gtarea_list), 3))
    print('-'*10)
    print("Dice score (heavy noisy labels - high quality labels):")
    print("Mean:", np.round(np.mean(dice_score_list), 3))
    print("Variance:", np.round(np.var(dice_score_list), 3))


def create_noisy_labels_semiheavy():
    count = 0
    count_noisy_pixel_abs_list = []         # absolut number of changed pixels
    ratio_noisy_pixel_wrt_gtarea_list = []  # ratio of changed pixels : pancreas area
    dice_score_list = []                    # dice score noisy - HQ labels

    for patient_id in range(1, 83):
        path = "data/train/PATIENT_00" + ('0' if patient_id < 10 else '') + str(patient_id) + "/noisy_labels_semiheavy"
        # remove noisy labels if there already exist some
        if os.path.exists(path):
            shutil.rmtree(path)
        if patient_id in (config.hq_indices + config.lq_indices):
            count += 1
            print("Create semi-heavy noisy labels for patient " + str(patient_id) + " (" + str(count) + "/" + str(len(config.hq_indices + config.lq_indices)) + ")")
            if patient_id < 10:
                prefix = '0'
            else:
                prefix = ''
            # create a new subfolder
            os.mkdir(path)

            # iterate over quality label slices and transform to noisy label slices
            for quality_label_slice_path in glob.iglob("data/train/PATIENT_00" + prefix + str(patient_id) + '/labels/*.npz'):
                quality_label_slice = np.load(quality_label_slice_path)['arr_0']
                noisy_label_slice = add_semiheavy_noise(quality_label_slice)
                count_noisy_pixels, ratio_gtarea, dice_score = compute_evaluation_metrics(quality_label_slice, noisy_label_slice)
                count_noisy_pixel_abs_list.append(count_noisy_pixels)
                ratio_noisy_pixel_wrt_gtarea_list.append(ratio_gtarea)
                dice_score_list.append(dice_score)
                slice_index = quality_label_slice_path.split('.')[0][-3:]
                file_name = "L_noisy-" + prefix + str(patient_id) + "-" + str(slice_index)
                dst_path = os.path.join(path, file_name)
                np.savez_compressed(dst_path, noisy_label_slice)

    print('-'*60)
    print("Number pixels changed:")
    print("Mean (of abs. number):", np.round(np.mean(count_noisy_pixel_abs_list), 3), "; ratio wrt to 512 x 512 pixels:", np.round(np.mean(count_noisy_pixel_abs_list)/(512*512), 5))
    print("Var (of abs. number):", np.round(np.var(count_noisy_pixel_abs_list), 3), ";  std (of abs. num):", np.round(np.std(count_noisy_pixel_abs_list), 3))
    print('-'*10)
    print("Number pixels changed wrt pancreas area:")
    print("Mean:", np.round(np.mean(ratio_noisy_pixel_wrt_gtarea_list), 3))
    print("Variance:", np.round(np.var(ratio_noisy_pixel_wrt_gtarea_list), 3))
    print('-'*10)
    print("Dice score (semi-heavy noisy labels - high quality labels):")
    print("Mean:", np.round(np.mean(dice_score_list), 3))
    print("Variance:", np.round(np.var(dice_score_list), 3))


preprocessing(config.parent_dir)
create_noisy_labels()
create_noisy_labels_light()
create_noisy_labels_heavy() # not used in our experiments
create_noisy_labels_semiheavy() # in presentation slides referred to as "heavy labels"






