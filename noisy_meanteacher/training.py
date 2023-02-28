import sys
sys.path.append(".")
import os
import numpy as np
from datetime import datetime
import pytz
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from monai.data import decollate_batch
from monai.transforms import Compose, EnsureType, Activations, AsDiscrete
from monai.losses import DiceLoss, FocalLoss
from monai.metrics import DiceMetric, MSEMetric
from monai.inferers import SimpleInferer

import config
from ssdm import ssdm
from ssdm_gpu import ssdm_gpu

# only for apirl's computer so you can comment it out
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# TENSORBOARD: open new terminal and enter: tensorboard --logdir=runs


#####################
# HANDLE SAVE MODEL #
#####################

def make_training_directory():
    # create unique folder for saving models of this training
    tz = pytz.timezone('Europe/Berlin')
    now = str(datetime.now(tz))
    folder_name = config.user + "_" + now[:10] + "_" + now[11:19].replace(':', '-')
    training_path = './models/noisy_mt/' + folder_name + '/'
    os.makedirs(training_path)

    params = {
        "learning_rate": config.learning_rate,
        "dropout": config.dropout,
        "channels": config.channels,
        "batch_size": config.batch_size,
        "alpha": config.alpha,
        "small_update_alpha" : config.small_update_alpha,
        "use beta": config.use_beta,
        "variance_additive_noise": config.variance_additive_noise,
        "variance_multiplicative_noise": config.variance_multiplicative_noise,
        "split_ratio": config.split_ratio,
        "startepoch_ssdm": config.startepoch_ssdm,
        "ssdm_prune_method": config.ssdm_prune_method,
        "ssdm_smoothing_tau": config.ssdm_smoothing_tau,
        "noisy_labels_type" : config.noisy_labels_type
    }

    with open(training_path + "config.txt", 'w') as file:
        # store the hyperparameters
        file.write("### Noisy Mean Teacher ###\n")
        file.write("%s\n" % params)

    return training_path


def handle_save_new_best_model(model, training_path, epoch, best_metric, threshold="0.5"):
    if threshold == "0.5":
        for file in os.listdir(training_path):
            if file.startswith("05_best_model"):
                os.remove(os.path.join(training_path, file))
        torch.save(model, training_path + "05_best_model_ep" + str(epoch + 1) + "_" + str(round(best_metric, 3)).replace('.', '-') + ".pth")
        print("Saved model as new 05_best_model")
    elif threshold == "0.075":
        for file in os.listdir(training_path):
            if file.startswith("0075_best_model"):
                os.remove(os.path.join(training_path, file))
        torch.save(model, training_path + "0075_best_model_ep" + str(epoch + 1) + "_" + str(round(best_metric, 3)).replace('.', '-') + ".pth")
        print("Saved model as new 0075_best_model")
    else:
        for file in os.listdir(training_path):
            if file.startswith("02_best_model"):
                os.remove(os.path.join(training_path, file))
        torch.save(model, training_path + "02_best_model_ep" + str(epoch + 1) + "_" + str(round(best_metric, 3)).replace('.', '-') + ".pth")
        print("Saved model as new 02_best_model")



######################
# NOISY MEAN TEACHER #
######################

def add_additive_and_multiplicative_noise(tensor):
    additive_noisy_tensor = tensor + (torch.randn(tensor.size()) * config.variance_additive_noise + 0).to(config.device)
    multi_noisy_tensor = additive_noisy_tensor * (torch.randn(tensor.size()) * config.variance_multiplicative_noise + 1).to(config.device)
    # clip to values from 0 to 1
    result = torch.maximum(torch.minimum(multi_noisy_tensor, torch.ones_like(multi_noisy_tensor)), torch.zeros_like(multi_noisy_tensor))
    return result


def update_ema_variables(student_model, teacher_model, alpha, global_step):
    # reference for ramp up https://arxiv.org/abs/1610.02242
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(alpha).add_(alpha=1 - alpha, other=student_param.data)


def train(student, teacher, optimizer_student,
          hq_dataloader, lq_dataloader, val_dataloader,
          best_metric=-1, best_metric_02=-1, best_metric_0075=-1, best_metric_epoch=-1,
          global_step=0, start_epoch=0, learning_rate=config.learning_rate,
          epoch_student_val_losses=[], training_path=None):

    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_trans_02 = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.2)])
    post_trans_0075 = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.075)])

    writer = SummaryWriter()  # Writer will output to ./runs/ directory by default

    simple_inferer = SimpleInferer()
    lr_updates = 0

    consistency_loss_func = MSEMetric(get_not_nans=False)
    dice_loss = DiceLoss(sigmoid=True)
    Focal_Loss = FocalLoss()
    CE_Loss = torch.nn.modules.loss.CrossEntropyLoss()
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_metric_02 = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_metric_0075 = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    use_ssdm = start_epoch >= config.startepoch_ssdm
    print("use ssdm:", use_ssdm)

    small_teacher_update = False

    if start_epoch == 0:
        training_path = make_training_directory()


    ##################
    # START TRAINING #
    ##################
    for param in teacher.parameters():
        param.detach_()

    for epoch in range(start_epoch, config.num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{config.num_epochs}")
        if config.use_learning_rate_decay and epoch in config.lr_decay_epochs:
            print("Learning rate decay - old lr:", learning_rate)
            learning_rate = learning_rate / config.lr_decay_divs[lr_updates]
            optimizer_student = torch.optim.Adam(student.parameters(), learning_rate)
            print("Updated optimizer with new lr:", learning_rate)
            lr_updates += 1

        if epoch == config.startepoch_ssdm:
            print("NOW WITH SSDM")
            use_ssdm = True

        student.train()
        teacher.train()
        step = 0

        # TENSORBOARD
        epoch_loss = 0
        epoch_supervised_loss = 0
        epoch_consistency_loss = 0
        epoch_ssdm_loss = 0
        epoch_num_labels_changed = 0
        epoch_ssdm_duration = 0

        for (hq_batch, lq_batch) in tqdm(zip(hq_dataloader, lq_dataloader), total=min(len(hq_dataloader), len(lq_dataloader))):
            step += 1
            global_step += 1

            # (1) get hq images and labels + add noise for teacher
            hq_images = torch.unsqueeze(torch.cat([b['image'] for b in hq_batch]), 1).to(config.device)
            hq_labels = torch.unsqueeze(torch.cat([b['label'] for b in hq_batch]), 1).to(config.device)
            hq_labels = torch.round(hq_labels)
            hq_labels = hq_labels.type(torch.int) # form: tensor (4,1,512,512)
            hq_images_noised = add_additive_and_multiplicative_noise(hq_images)

            # (2) get lq images and labels + add noise for teacher
            lq_images = torch.unsqueeze(torch.cat([b['image'] for b in lq_batch]), 1).to(config.device)
            lq_labels = torch.unsqueeze(torch.cat([b['label'] for b in lq_batch]), 1).to(config.device)
            lq_labels = torch.round(lq_labels)
            lq_labels = lq_labels.type(torch.int) # form: tensor (4,1,512,512)
            lq_images_noised = add_additive_and_multiplicative_noise(lq_images)

            # (3) calculate outputs
            optimizer_student.zero_grad()

            # student outputs
            out_hq = student(hq_images)
            out_lq = student(lq_images)

            # teacher outputs
            with torch.no_grad():
                out_hq_noised = teacher(hq_images_noised)
                out_lq_noised = teacher(lq_images_noised)

            # (4) compute losses
            outputs_student = torch.cat((out_hq, out_lq), 0)
            outputs_teacher = torch.cat((out_hq_noised, out_lq_noised), 0)
            consistency_loss = consistency_loss_func(torch.sigmoid(outputs_student), torch.sigmoid(outputs_teacher)).mean() * config.cons_loss_factor
            supervised_loss = dice_loss(out_hq, hq_labels)
            if not config.disable_cupy and config.device == 'cuda':
                ssdm_loss, num_labeles_changed, duration = ssdm_gpu(out_lq_noised, out_lq, lq_labels, CE_Loss, Focal_Loss) if use_ssdm else (torch.tensor(0), 0, 0)
            else:
                ssdm_loss, num_labeles_changed, duration = ssdm(out_lq_noised, out_lq, lq_labels, CE_Loss, Focal_Loss) if use_ssdm else (torch.tensor(0), 0, 0)

            # ramp-up adapted from: https://arxiv.org/abs/1903.01248
            beta = np.exp(-5 * (1 - (global_step / 400)) ** 2) if config.use_beta and global_step <= 400 else 1
            student_loss = supervised_loss + beta * consistency_loss + 0.5 * ssdm_loss

            # (5) backward pass & optimizer step for student; ema update for teacher
            student_loss.backward()
            optimizer_student.step()
            update_ema_variables(student, teacher, alpha=config.alpha if not small_teacher_update else config.small_update_alpha, global_step=global_step)

            # TENSORBOARD
            epoch_loss += student_loss.item()
            epoch_supervised_loss += supervised_loss.item()
            epoch_consistency_loss += consistency_loss.item()
            epoch_ssdm_loss += ssdm_loss.item()
            epoch_num_labels_changed += num_labeles_changed
            epoch_ssdm_duration += duration

        # TENSORBOARD

        epoch_loss /= step
        writer.add_scalar('train loss', epoch_loss, epoch)
        writer.add_scalar('supervised loss', epoch_supervised_loss/step, epoch)
        writer.add_scalar('consistency loss', epoch_consistency_loss/step, epoch)
        writer.add_scalar('ssdm loss', epoch_ssdm_loss/step, epoch)
        writer.add_scalar('num labels changed by the ssdm', epoch_num_labels_changed/step, epoch)
        writer.add_scalar('avg. ssdm duration (seconds)', epoch_ssdm_duration/step, epoch)

        print(f"epoch {epoch + 1} average student loss: {epoch_loss:.4f} (sup. loss: {epoch_supervised_loss/step:.4f}, cons. loss: {epoch_consistency_loss / step:.4f}, ssdm loss: {epoch_ssdm_loss / step:.4f})")

        ##############
        # VALIDATION #
        ##############

        if (epoch + 1) % config.val_interval == 0:
            print("validation in progress...")
            teacher.eval()
            student.eval()

            with torch.no_grad():

                val_losses = list()
                val_losses_student = list()

                for val_data in val_dataloader:
                    val_images, val_labels = val_data["image"].to(config.device), val_data["label"].to(config.device)
                    val_labels = torch.round(val_labels)
                    val_labels = val_labels.type(torch.int)
                    val_outputs = simple_inferer(val_images, teacher)
                    val_outputs_student = simple_inferer(val_images, student)

                    # compute validation dice loss
                    current_dice_loss = dice_loss(val_outputs, val_labels).item()
                    val_losses.append(current_dice_loss)
                    val_outputs_05 = [post_trans(i) for i in decollate_batch(val_outputs)]
                    val_outputs_02 = [post_trans_02(i) for i in decollate_batch(val_outputs)]
                    val_outputs_0075 = [post_trans_0075(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs_05, y=val_labels)
                    dice_metric_02(y_pred=val_outputs_02, y=val_labels)
                    dice_metric_0075(y_pred=val_outputs_0075, y=val_labels)

                    # compute validation dice loss STUDENT    all DEBUG
                    current_dice_loss_student = dice_loss(val_outputs_student, val_labels).item()
                    val_losses_student.append(current_dice_loss_student)
                    val_outputs_student = [post_trans(i) for i in decollate_batch(val_outputs_student)]

                # aggregate the final mean dice loss
                val_loss = sum(val_losses) / len(val_losses)
                writer.add_scalar('validation loss', val_loss, epoch)

                # aggregate the final mean dice loss STUDENT  all DEBUG
                val_loss_student = sum(val_losses_student) / len(val_losses_student)
                epoch_student_val_losses.append(val_loss_student)
                writer.add_scalar('validation loss student', val_loss_student, epoch)

                # aggregate the final mean dice metric
                metric = dice_metric.aggregate().item()
                metric_02 = dice_metric_02.aggregate().item()
                metric_0075 = dice_metric_0075.aggregate().item()

                # reset the metric status for next validation round
                dice_metric.reset()
                dice_metric_02.reset()
                dice_metric_0075.reset()

                if metric > best_metric:
                    print("New best metric!")
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    handle_save_new_best_model(teacher, training_path, epoch, best_metric)

                if metric_02 > best_metric_02:
                    print("New best metric with threshold 0.2!")
                    best_metric_02 = metric_02
                    handle_save_new_best_model(teacher, training_path, epoch, best_metric_02, "0.2")

                if metric_0075 > best_metric_0075:
                    print("New best metric with threshold 0.075!")
                    best_metric_0075 = metric_0075
                    handle_save_new_best_model(teacher, training_path, epoch, best_metric_0075, "0.075")

                print(
                    "Current epoch: {}, current dice metric: {:.4f}, current dice loss: {:.4f} \n Best dice metric: {:.4f} at epoch {}".format(
                        epoch + 1, metric, val_loss, best_metric, best_metric_epoch
                    )
                )

                # update teacher weights only if student val did not get worse than average of last 5 epochs
                avg_loss_student = np.mean(epoch_student_val_losses[(len(epoch_student_val_losses) - 5):])

                if val_loss_student > avg_loss_student:
                    small_teacher_update = True
                else:
                    small_teacher_update = False
                print("Next epoch: Do BIG teacher updates" if not small_teacher_update else "Next epoch: Do SMALL teacher updates")

        # save current model
        if (epoch+1) > 10 and (epoch+1) % config.save_model_interval == 0:
            torch.save({
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer_student.state_dict(),
                'learning_rate': learning_rate,
                'best_metric': best_metric,
                'best_metric_02': best_metric_02,
                'best_metric_epoch': best_metric_epoch,
                'best_metric_0075': best_metric_0075,
                'global_step': global_step,
                'epoch_student_val_losses': epoch_student_val_losses,
                'training_path': training_path
                }, training_path + 'checkpoint.pt')
            print("Saved current model of epoch ", epoch + 1, "in", training_path + 'checkpoint.pt')

    print("Training finished.")

