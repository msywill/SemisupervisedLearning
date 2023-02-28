import sys
sys.path.append(".")
import numpy as np
import copy
import torch
from torch.utils.data.dataset import ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from monai.data import DataLoader, Dataset
from monai.transforms import RandRotated, LoadImaged, Compose, EnsureTyped, AddChanneld, ScaleIntensityd
from monai.networks.nets import UNet

import config
from dictionary import get_dictionaries_mean_teacher
from semisup_meanteacher.training import train


print("loading dataset...")

assert config.split_ratio in ["40 - 10", "40 - 20", "40 -30", "40 - 40", "40 - 50", "40 - 60", "60 - 40"]

if config.split_ratio == "40 - 60":
    unlabeled_dict, labeled_dict, val_dict = get_dictionaries_mean_teacher()
    batch_size_labeled = int(config.batch_size * 0.4)
    batch_size_unlabeled = int(config.batch_size * 0.6)

elif config.split_ratio == "60 - 40":
    labeled_dict, unlabeled_dict, val_dict = get_dictionaries_mean_teacher()
    batch_size_labeled = int(config.batch_size * 0.6)
    batch_size_unlabeled = int(config.batch_size * 0.4)

elif config.split_ratio == "40 - 10":
    unlabeled_dict, labeled_dict, val_dict = get_dictionaries_mean_teacher()
    batch_size_labeled = int(config.batch_size * 0.4)
    batch_size_unlabeled = int(config.batch_size * 0.1)

elif config.split_ratio == "40 - 20":
    unlabeled_dict, labeled_dict, val_dict = get_dictionaries_mean_teacher()
    batch_size_labeled = int(config.batch_size * 0.4)
    batch_size_unlabeled = int(config.batch_size * 0.2)

elif config.split_ratio == "40 - 30":
    unlabeled_dict, labeled_dict, val_dict = get_dictionaries_mean_teacher()
    batch_size_labeled = int(config.batch_size * 0.4)
    batch_size_unlabeled = int(config.batch_size * 0.3)

elif config.split_ratio == "40 - 40":
    unlabeled_dict, labeled_dict, val_dict = get_dictionaries_mean_teacher()
    batch_size_labeled = int(config.batch_size * 0.4)
    batch_size_unlabeled = int(config.batch_size * 0.4)

elif config.split_ratio == "40 - 50":
    unlabeled_dict, labeled_dict, val_dict = get_dictionaries_mean_teacher()
    batch_size_labeled = int(config.batch_size * 0.4)
    batch_size_unlabeled = int(config.batch_size * 0.5)


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

combined_labeled_ds = ConcatDataset(
    [Dataset(data=labeled_dict, transform=load), Dataset(data=labeled_dict, transform=load_augm)])
labeled_dataloader = DataLoader(combined_labeled_ds, batch_size=batch_size_labeled, shuffle=True, collate_fn=lambda x: x )

combined_unlabeled_ds = ConcatDataset(
    [Dataset(data=unlabeled_dict, transform=load), Dataset(data=unlabeled_dict, transform=load_augm)])
unlabeled_dataloader = DataLoader(combined_unlabeled_ds, batch_size=batch_size_unlabeled, shuffle=True, collate_fn=lambda x: x )

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
    writer = SummaryWriter()

    train(student_model, teacher_model, writer, optimizer_student, labeled_dataloader, unlabeled_dataloader, val_dataloader)

else:
    checkpoint = torch.load('./checkpoint.pt')
    student_model.load_state_dict(checkpoint['student_state_dict'])
    teacher_model.load_state_dict(checkpoint['teacher_state_dict'])

    optimizer_student = torch.optim.Adam(student_model.parameters(), checkpoint['learning_rate'])
    optimizer_student.load_state_dict(checkpoint['optimizer_state_dict'])

    print("continue")

    train(student_model, teacher_model, SummaryWriter(), optimizer_student,
          labeled_dataloader, unlabeled_dataloader, val_dataloader,
          checkpoint['best_metric'], checkpoint['best_metric_02'], checkpoint['best_metric_0075'], checkpoint['best_metric_epoch'],
          checkpoint['global_step'], checkpoint['epoch'], checkpoint['learning_rate'], checkpoint['epoch_student_val_losses'], checkpoint['training_path'])


