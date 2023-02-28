import sys
sys.path.append(".")
from monai.data import DataLoader, Dataset
from monai.transforms import RandRotated, LoadImaged, Compose, EnsureTyped, AddChanneld, ScaleIntensityd, EnsureType, \
    Activations, AsDiscrete
from monai.losses import DiceLoss
from monai.networks.nets import UNet
import torch
import numpy as np
from itertools import product
import config
from training import training
from dictionary import get_dictionaries_baseline
from torch.utils.data.dataset import ConcatDataset

# TENSORBOARD: open new terminal and enter: tensorboard --logdir=runs

#############
# LOAD DATA #
#############
print("loading dataset...")

train_dict, val_dict = get_dictionaries_baseline()

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

val_ds = Dataset(data=val_dict, transform=load)
val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

loss_function = DiceLoss(sigmoid=True)




################################
# Start Grid Search / Training #  --> depends on whether use_grid_search = True / False in config.py
################################

assert config.use_grid_search in [True, False]

# Gridsearch
if config.use_grid_search:

    results_grid_search = []
    print("Hyperparameter grid search will be performed on the following grid points:")
    for c in product(*config.hyperparams.values()):
        print(c)

    for current_params in product(*config.hyperparams.values()):
        print("Current hyperparameters: ", current_params)

        for (i, key) in enumerate(config.hyperparams.keys()):

            value = current_params[i]
            if key == "learning_rate":
                learning_rate_grid_search = value
            elif key == "dropout":
                dropout_grid_search = value
            elif key == "channels":
                channels_grid_search = value
                strides_grid_search = (2,) * (len(channels_grid_search) - 1)
            elif key == "batch_size":
                batch_size_grid_search = value

        train_dataloader_grid_search = DataLoader(combined_ds, batch_size=batch_size_grid_search, shuffle=True, collate_fn=lambda x: x)

        model_grid_search = UNet(spatial_dims=2,
                    in_channels=1,
                    out_channels=1,
                    channels=channels_grid_search,
                    strides=strides_grid_search,
                    dropout=dropout_grid_search).to(config.device)

        optimizer_grid_search = torch.optim.Adam(model_grid_search.parameters(), learning_rate_grid_search)

        # train model with current hyperparameters
        loss_grid_search = training(model=model_grid_search, optimizer=optimizer_grid_search, loss_function=loss_function,
                        train_dataloader=train_dataloader_grid_search, val_dataloader=val_dataloader, post_trans=post_trans,
                        lr=learning_rate_grid_search, do=dropout_grid_search, chs=channels_grid_search, bs=batch_size_grid_search)

        params_grid_search = {
            "learning_rate": learning_rate_grid_search,
            "dropout": dropout_grid_search,
            "channels": channels_grid_search,
            "batch_size": batch_size_grid_search
        }
        
        results_grid_search.append([loss_grid_search, params_grid_search])

    # Save the best performing hyperparameter-combinations to a txt-file
    results_grid_search.sort(key=lambda y: y[0])

    with open('./models/gridsearch_results.txt', 'w') as f:
        for result in results_grid_search:
            f.write("%s\n" % result)

    for i in range(min(len(results_grid_search), 2)):
        print("\n Top", i + 1, "performing hyperparameter combination:")

        print("Loss:", results_grid_search[i][0])
        print("with params:", results_grid_search[i][1])


# normal training
else:

    learning_rate = config.learning_rate
    dropout = config.dropout
    channels = config.channels
    batch_size = config.batch_size
    strides = config.strides

    train_dataloader = DataLoader(combined_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

    model = UNet(spatial_dims=2,
                    in_channels=1,
                    out_channels=1,
                    channels=channels,
                    strides=strides,
                    dropout=dropout).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    loss = training(model=model, optimizer=optimizer, loss_function=loss_function,
                        train_dataloader=train_dataloader, val_dataloader=val_dataloader, post_trans=post_trans,
                        lr=learning_rate, do=dropout, chs=channels, bs=batch_size)

    print("Loss:", loss)

    


