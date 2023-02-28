import os
import sys
sys.path.append(".")

import torch
import time
from datetime import datetime
import pytz
import config
from monai.inferers import SimpleInferer
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from baseline.visualizer import Visualizer
import copy


def training(model, optimizer, loss_function, train_dataloader, val_dataloader,
             post_trans, lr, do, chs, bs, mode='baseline'):
    
    # initializations
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    patience = config.patience if config.use_early_stopping else config.num_epochs
    no_imp_vals = 0
    best_metric = -1
    best_metric_epoch = -1
    lr_updates = 0
    epoch_loss_values = list()
    val_loss_values = list()
    val_dice_values = list()
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    simple_inferer = SimpleInferer()
    visualizer = Visualizer(val_dataloader=copy.deepcopy(val_dataloader), inferer=copy.deepcopy(simple_inferer), num_samples=4, writer=writer, post_trans=post_trans)

    # create unique folder for saving models of this training
    tz = pytz.timezone('Europe/Berlin')
    now = str(datetime.now(tz))
    folder_name = config.user + "_" + now[:10] + "_" + now[11:19].replace(':', '-')
    if mode == 'baseline':
        training_path = './models/baseline/' + folder_name + '/'
    else:
        training_path = './models/pretraining/' + folder_name + '/'

    os.makedirs(training_path)

    params = {
        "learning_rate": lr,
        "dropout": do,
        "channels": chs,
        "batch_size": bs
    }

    with open(training_path + "config.txt", 'w') as file:
        # store the hyperparameters
        file.write("%s\n" % params)

    # training timer
    t_start = time.time()

    ############
    # TRAINING #
    ############
    for epoch in range(config.num_epochs):
        print("-" * 10)
        if no_imp_vals == patience:
            print("early stopping after ", epoch, " epochs")
            break

        print(f"epoch {epoch + 1}/{config.num_epochs}")
        if config.use_learning_rate_decay and epoch in config.lr_decay_epochs:
            print("Learning rate decay - old lr:", lr)
            lr = lr / config.lr_decay_divs[lr_updates]
            optimizer = torch.optim.Adam(model.parameters(), lr)
            print("Updated optimizer with new lr:", lr)
            lr_updates += 1

        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in tqdm(train_dataloader):
            step += 1
            inputs = torch.unsqueeze(torch.cat([b['image'] for b in batch_data]), 1).to(config.device)
            labels = torch.unsqueeze(torch.cat([b['label'] for b in batch_data]), 1).to(config.device)
            labels = torch.round(labels)
            labels = labels.type(torch.int)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        writer.add_scalar('train loss', epoch_loss, epoch)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        ##############
        # VALIDATION #
        ##############

        if (epoch + 1) % config.val_interval == 0:
            print("validation in progress...")
            model.eval()
            visualizer.visualize(model=model, epoch=epoch)
            no_imp_vals += 1
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                val_losses = list()
                for val_data in val_dataloader:
                    val_images, val_labels = val_data['image'].to(config.device), val_data['label'].to(config.device)
                    val_labels = torch.round(val_labels)
                    val_labels = val_labels.type(torch.int)
                    val_outputs = simple_inferer(val_images, model)
                    
                    # compute validation dice loss:
                    dice_loss = loss_function(val_outputs, val_labels).item()
                    val_losses.append(dice_loss)

                    # compute validation dice metric for current iteration:
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                val_loss = sum(val_losses) / len(val_losses)
                val_loss_values.append(val_loss)
                writer.add_scalar('validation loss', val_loss, epoch)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                val_dice_values.append(metric)
                writer.add_scalar('validation metric', metric, epoch)

                # set model as best_model if new best val_loss + delete last bst model
                if metric > best_metric:
                    no_imp_vals = 0
                    print("New best metric!")
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    for file in os.listdir(training_path):
                        if file.startswith("best_model"):
                            os.remove(os.path.join(training_path, file))
                    torch.save(model,
                               training_path + "best_model_ep" + str(epoch + 1) + "_" + str(round(best_metric, 3)).replace('.', '-') + ".pth")
                    print("Saved model as new best_model")

                print("Current epoch: {}, current dice metric: {:.4f}, current dice loss: {:.4f} \n Best dice metric: {:.4f} at epoch {}".format(
                        epoch + 1, metric, val_loss, best_metric, best_metric_epoch))

        # save current model
        if (epoch + 1) % config.save_model_interval == 0:
            name = "model_epoch_" + str(epoch + 1) + ".pth"
            torch.save(model, training_path + name)
            print("Saved current model of epoch ", epoch + 1)

    # after training:
    print(f"train completed, best metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    t_end = time.time()
    print("Duration of training: ", (t_end - t_start) / 60, "mins")

    return best_metric

