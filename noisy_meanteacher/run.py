import sys
sys.path.append(".")
import numpy as np
import copy
import torch
from torch.utils.data.dataset import ConcatDataset

from monai.data import DataLoader, Dataset
from monai.transforms import RandRotated, LoadImaged, Compose, EnsureTyped, AddChanneld, ScaleIntensityd
from monai.networks.nets import UNet

import config
from dictionary import get_dictionaries_mean_teacher
from noisy_meanteacher.training import train


print("loading dataset...")

assert config.split_ratio in ["40 - 60", "60 - 40"]

if config.split_ratio == "40 - 60":
    lq_dict, hq_dict, val_dict = get_dictionaries_mean_teacher(noisy_type=config.noisy_labels_type)

    batch_size_hq = int(config.batch_size * 0.4)
    batch_size_lq = int(config.batch_size * 0.6)
else:
    hq_dict, lq_dict, val_dict = get_dictionaries_mean_teacher(noisy_type=config.noisy_labels_type)

    batch_size_hq = int(config.batch_size * 0.6)
    batch_size_lq = int(config.batch_size * 0.4)


load = Compose([
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    ScaleIntensityd(keys=["image", "label"], minv=0, maxv=1),
    EnsureTyped(keys=["image", "label"]),
])

load_augm = Compose([
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    ScaleIntensityd(keys=["image", "label"], minv=0, maxv=1),
    RandRotated(keys=["image", "label"], range_x=np.pi / 12, prob=1),
    EnsureTyped(keys=["image", "label"]),
])


combined_hq_ds = ConcatDataset(
    [Dataset(data=hq_dict, transform=load), Dataset(data=hq_dict, transform=load_augm)])
hq_dataloader = DataLoader(combined_hq_ds, batch_size=batch_size_hq, shuffle=True, collate_fn=lambda x: x )

combined_lq_ds = ConcatDataset(
    [Dataset(data=lq_dict, transform=load), Dataset(data=lq_dict, transform=load_augm)])
lq_dataloader = DataLoader(combined_lq_ds, batch_size=batch_size_lq, shuffle=True, collate_fn=lambda x: x )

val_ds = Dataset(data=val_dict, transform=load)
val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)


student_model = UNet(spatial_dims=2,
                     in_channels=1,
                     out_channels=1,
                     channels=config.channels,
                     strides=config.strides,
                     dropout=config.dropout).to(config.device)

teacher_model = copy.deepcopy(student_model)

if not config.continue_training:

    optimizer_student = torch.optim.Adam(student_model.parameters(), config.learning_rate)

    train(student=student_model, teacher=teacher_model,
          optimizer_student=optimizer_student, hq_dataloader=hq_dataloader, lq_dataloader=lq_dataloader, val_dataloader=val_dataloader)

else:
    checkpoint = torch.load( config.checkpoint_path + '/checkpoint.pt')
    student_model.load_state_dict(checkpoint['student_state_dict'])
    teacher_model.load_state_dict(checkpoint['teacher_state_dict'])

    optimizer_student = torch.optim.Adam(student_model.parameters(), checkpoint['learning_rate'])
    optimizer_student.load_state_dict(checkpoint['optimizer_state_dict'])

    print("continue")

    train(student=student_model, teacher=teacher_model,
          optimizer_student=optimizer_student, hq_dataloader=hq_dataloader, lq_dataloader=lq_dataloader, val_dataloader=val_dataloader,
          best_metric=checkpoint['best_metric'], best_metric_02=checkpoint['best_metric_02'], best_metric_0075=checkpoint['best_metric_0075'], best_metric_epoch=checkpoint['best_metric_epoch'],
          global_step=checkpoint['global_step'], start_epoch=checkpoint['epoch'], learning_rate=checkpoint['learning_rate'],
          epoch_student_val_losses=checkpoint['epoch_student_val_losses'], training_path=checkpoint['training_path'])


