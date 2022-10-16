import argparse
import logging
from pathlib import Path



dir_img = Path('./data/imgs/') #create a path object
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')

def train_net(net,
              device,
              epochs: int = 10,
              batch_size: int = 2,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_check_point: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # create data set:

