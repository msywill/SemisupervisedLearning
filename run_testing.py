import os
import glob
import nibabel as nib
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from monai.data import Dataset, DataLoader
from monai.transforms import LoadImaged, Compose, EnsureTyped, AddChanneld, ScaleIntensityd, EnsureType, Activations, \
    AsDiscrete
from monai.inferers import SimpleInferer

from dictionary import generate_test_paths, create_dict
import config


# load the pretrained model
def prediction(model, dataset, threshold):
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    post_trans = Compose(
        [
            EnsureType(),
            Activations(sigmoid=True),
            AsDiscrete(threshold=threshold)
        ]
    )

    # stack prediction and groundtruth back into 3D
    simple_inferer = SimpleInferer()
    groundtruth_3D = []
    prediction_3D_int16 = []

    # set the model to evaluate
    model.eval()
    with torch.no_grad():
        for test_batch in tqdm(test_dataloader):
            images, labels = test_batch["image"], test_batch["label"]
            labels = torch.round(labels)
            labels = labels.type(torch.int)

            # use fixed inference
            outputs = simple_inferer(inputs=images, network=model)
            # convert outputs into binary
            outputs = post_trans(outputs)
            # convert outputs and labels into numpy
            outputs_to_np = (torch.squeeze(outputs)).detach().numpy()
            label_to_np = (torch.squeeze(labels)).detach().numpy()
            prediction_3D_int16.append(outputs_to_np.astype(np.int16))  # save as int16 for nifti require
            groundtruth_3D.append(label_to_np.astype(np.int16))

        return np.array(groundtruth_3D), np.array(prediction_3D_int16)


def get_score(prediction, gt):
    intersection = np.logical_and(prediction, gt)
    dice_score = 2. * intersection.sum() / (prediction.sum() + gt.sum())

    return dice_score


def connected_domain(image, mask=True):
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    _input = sitk.GetImageFromArray(image.astype(np.uint8))
    output_ex = cca.Execute(_input)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(output_ex)
    num_label = cca.GetObjectCount()
    num_list = [i for i in range(1, num_label+1)]
    area_list = []
    for l in range(1, num_label+1):
        area_list.append(stats.GetNumberOfPixels(l))
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    largest_area = area_list[num_list_sorted[0] - 1]
    final_label_list = [num_list_sorted[0]]

    for idx, i in enumerate(num_list_sorted[1:]):
        if area_list[i-1] >= (largest_area//10):
            final_label_list.append(i)
        else:
            break
    output = sitk.GetArrayFromImage(output_ex)

    for one_label in num_list:
        if one_label in final_label_list:
            continue
        x, y, z, w, h, d = stats.GetBoundingBox(one_label)
        one_mask = (output[z:z+d, y:y+h, x:x+w] != one_label)
        output[z:z+d, y:y+h, x:x+w] *= one_mask

    if mask:
        output = (output>0).astype(np.uint8)
    else:
        output = ((output > 0)*255.).astype(np.uint8)
    return output


def testing(model_path=config.testing_model_path, testing_threshold=config.testing_threshold):
    patient_paths = generate_test_paths()

    # in order to save the np.array back into nifti, we need the affine parameters of original
    # nifti file, here i load one label from data_virgin
    func_filename = os.path.join('nifti_label_for_metadata', 'label0001.nii.gz')
    func = nib.load(func_filename)

    model = torch.load(model_path, map_location=torch.device('cpu'))

    load = Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ScaleIntensityd(keys=["image", "label"], minv=0, maxv=1),
        EnsureTyped(keys=["image", "label"], data_type='tensor'),
    ])

    dice_scores = []
    len_test = len(patient_paths)
    for (i, patient) in enumerate(patient_paths):
        dict = create_dict([patient])
        dataset = Dataset(data=dict, transform=load)

        print("Predict labels for Patient " + str(patient[-2:]) + " (" + str(i + 1) + "/" + str(len_test) + ")")
        gt, pred = prediction(model=model, dataset=dataset, threshold=testing_threshold)
        dice_score_ucc = get_score(pred, gt)
        mcc = connected_domain(pred)
        dice_score = get_score(mcc, gt)
        dice_scores.append(max(dice_score, dice_score_ucc))
        if config.save_niftis:
            save_nifti(mcc, gt, patient[-2:], func)

    mean_dice = sum(dice_scores) / len(dice_scores)
    variance = np.var(dice_scores)

    print("-"*20)
    print("Testing of model", model_path.split('/')[-1], "with threshold", testing_threshold, "done.")
    print("Dice scores:", np.around(dice_scores, 3))
    print("Mean dice score is", np.round(mean_dice, 3))
    print("Variance of dice scores is", np.round(variance, 3))
    print("-"*40, "\n")

    return np.round(mean_dice, 3)


def save_nifti(mcc, gt, patient, func):
    mcc = mcc.swapaxes(2, 0)
    mcc = mcc.swapaxes(1, 0)
    gt = gt.swapaxes(2, 0)
    gt = gt.swapaxes(1, 0)

    ni_img = nib.Nifti1Image(mcc, func.affine)
    nib.save(ni_img, 'models/testing/' + str(patient) + '_output.nii.gz')

    ni_img = nib.Nifti1Image(gt, func.affine)
    nib.save(ni_img, 'models/testing/' + str(patient) + '_groundtruth.nii.gz')


def run_one():
    testing()


def run_all():
    models_path = config.testing_model_path
    path_to_testing = '/'.join(models_path.split('/')[:-1]) + '/'
    scores = []
    for model_path in glob.glob(path_to_testing + "*best_model*"):
        num_string = model_path.split('/')[-1][:4]
        if num_string == "0075":
            testing_threshold = 0.075
        elif num_string == "02_b":
            testing_threshold = 0.2
        elif num_string == "05_b":
            testing_threshold = 0.5
        else:
            testing_threshold = None

        cur_score = testing(model_path=model_path, testing_threshold=testing_threshold)
        scores.append(cur_score)
    scores = np.array(scores)
    print(scores)
    print(np.max(scores), np.argmax(scores))


# without the following line an error could occure
# others can outcomment the line
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

run_one()

