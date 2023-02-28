import sys
sys.path.append(".")
import os
from monai.data import DataLoader, Dataset
from monai.transforms import RandRotated, LoadImaged, Compose, EnsureTyped, AddChanneld, ScaleIntensityd, EnsureType, \
    Activations, AsDiscrete
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from torch.utils.data.dataset import ConcatDataset
import torch
import numpy as np

from dictionary import get_dictionaries_mean_teacher
import config
from baseline.training import training

# only for apirl's computer so you can comment it out
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# TENSORBOARD: open new terminal and enter: tensorboard --logdir=runs

#############
# LOAD DATA #
#############
print("loading dataset...")

_, train_dict, val_dict = get_dictionaries_mean_teacher()

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
    RandRotated(keys=["image", "label"], range_x=np.pi/12, prob=1),
    EnsureTyped(keys=["image", "label"]), 
])

post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])      # if this throws an error, update monai to 0.8.0

train_ds = Dataset(data=train_dict, transform=load)
train_ds_augm = Dataset(data=train_dict, transform=load_augm)
       
combined_ds = ConcatDataset([train_ds, train_ds_augm])
train_dataloader = DataLoader(combined_ds, batch_size=config.batch_size, shuffle=True, collate_fn=lambda x: x )

val_ds = Dataset(data=val_dict, transform=load)
val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

##########################
# Load/Set Configuration #
##########################

learning_rate = config.learning_rate
dropout = config.dropout
channels = config.channels
strides = config.strides
batch_size = config.batch_size
loss_function = DiceLoss(sigmoid=True)

######################
#      Training      #
######################

model = UNet(spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=channels,
                strides=strides,
                dropout=dropout).to(config.device)

optimizer = torch.optim.Adam(model.parameters(), learning_rate)

loss = training(model=model, optimizer=optimizer,loss_function=loss_function,
                train_dataloader=train_dataloader, val_dataloader=val_dataloader, post_trans=post_trans,
                lr=learning_rate, do=dropout, chs=channels, bs=batch_size, mode = 'pretraining')


