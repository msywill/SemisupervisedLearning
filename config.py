import torch
import numpy as np

# please change this path to where you downloaded the CT-data
parent_dir = './data'

# for COLAB
#parent_dir = '/gdrive/MyDrive/data'

user = "noisy-MT"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device used: " + str(device))

test_indices = [26, 20, 74, 18, 23, 35, 7, 76, 44, 17, 66, 30, 14, 24] # kick out 62 and 81
val_indices = [1, 43, 57, 50, 68, 28, 52, 39, 11, 55, 64, 21, 13]
train_indices = [x for x in np.arange(1, 83) if x not in test_indices + val_indices + [25, 70, 80]]
lq_indices = [46, 75, 63, 38, 12, 56, 78, 54, 69, 33, 16, 82, 32, 5, 41, 6, 53, 2, 60, 61, 9, 48, 49, 71, 34, 47, 36, 77, 73, 45, 3]
hq_indices = [29, 79, 10, 40, 59, 8, 58, 37, 65, 42, 15, 27, 19, 72, 51, 4, 31, 67, 22]

### split 40-10 ###
# lq_indices = [46, 75, 63, 38, 49]
# ratio labeled/unlabeled = 4


### split 40-20 ###
# lq_indices = [5, 45, 54, 33, 63, 56, 78, 36, 49]
# ratio labeled/unlabeled = 2


### split 40-30 ###
#lq_indices =  [54, 77, 6, 3, 53, 5, 47, 71, 60, 38, 73, 61, 56, 69, 63]
# ratio labeled/unlabeled = 4/3


### split 40-40 ###
#lq_indices = [63, 78, 41, 56, 36, 82, 61, 77, 5, 47, 60, 45, 33, 48, 9, 16, 3, 38, 46, 6]
# or
# lq_indices = [3, 56, 16, 45, 77, 69, 36, 9, 48, 60, 73, 53, 6, 32, 63, 61, 2, 47, 71, 46, 82]
# for both of them, ratio labeled/unlabeled = 4/4 =1


### split 40-50 ###
# lq_indices = [46, 36, 71, 33, 53, 61, 63, 49, 48, 12, 34, 45, 2, 32, 60, 41, 75, 9, 77, 16, 47, 54, 73, 69, 82, 56]
# or
# lq_indices = [5, 49, 38, 61, 53, 45, 33, 73, 82, 9, 63, 77, 41, 32, 46, 3, 6, 48, 78, 71, 54, 60, 36, 34, 75]
# or
# lq_indices = [60, 48, 69, 2, 49, 73, 9, 78, 32, 47, 12, 63, 71, 33, 56, 46, 77, 75, 6, 36, 34, 16, 82, 38, 61, 45]
# ratio labeled/unlabeled = 4/5


num_epochs = 75
use_early_stopping = True
patience = 200  # number of validations(!) without improvement that are tolerated before early-stopping
val_interval = 1
save_model_interval = 3

########################################################################
#  Set model configuration for baseline as well as mean teacher models #
########################################################################

use_grid_search = False

# applies if: use_grid_search = True
# hyperparameter configurations for gridsearch ONLY for baseline model
hyperparams = {
    "learning_rate": [0.001],
    "dropout": [0.0],
    "channels": [(16, 32, 64, 128, 256)],
    "batch_size": [10]
}

# applies if: use_grid_search = False
# mode configuration for normal baseline training + all mean teacher models
learning_rate = 0.001
use_learning_rate_decay = True
lr_decay_epochs = [100, 101]                  # epochs where lr is updated
lr_decay_divs = [10, 5]                     # same len as lr_decay_epochs! lr' := lr / divs
dropout = 0.2                               # set to 0.0 to deactivate dropout
channels = (16, 32, 64, 128, 256)
batch_size = 10
strides = (2,) * (len(channels) - 1)




#####################
# CONTINUE TRAINING #
#####################
continue_training = False
checkpoint_path = './models/noisy_mt/Noisy-MT_2022-02-17_23-20-19'

################
# MEAN TEACHER #
################

# debug version
# pretrained_teacher = './final_results/semisupervised_mt/debug_teacher_small_2022-01-28_18-29-46/models/model_epoch_10.pth'
# big version
use_pretrained = False  # if false, train from scratch
pretrained_teacher = '../models/pretraining/best_model_ep11_0-504.pth'

# parameters used for both semisupervised and noisy mean teacher
# alpha 0.999(lrate): noise bigger; batchsize 5;20
alpha = 0.99
small_update_alpha = 0.999
consistency_loss = "mse" # no other option used anymore
use_beta = True     # use beta for consistency loss ramp-up
cons_loss_factor = 1
variance_additive_noise = 0.025
variance_multiplicative_noise = 0.075
split_ratio = "40 - 60"  # labeled - unlabeled; options: "40 - 10", "40 - 20", "40 -30", "40 - 40", "40 - 50" "40 - 60", "60 - 40"

# parameters used for only noisy mean teacher
startepoch_ssdm = 3    # start epoch for the ssdm
ssdm_prune_method = 'prune_by_class'       # either 'prune_by_class' or 'both' => try which works better!
ssdm_smoothing_tau = 0.8
noisy_labels_type = "noisy_labels"     # options: "noisy_labels", "noisy_labels_light", "noisy_labels_heavy"

disable_cupy = False

continue_model_student = './models/noisy_mt/Maxi_retrain_best_result_2022-02-11_18-55-58/continue_training/student1.pth'
continue_model_teacher = './models/noisy_mt/Maxi_retrain_best_result_2022-02-11_18-55-58/continue_training/teacher1.pth'


###########
# TESTING #
###########
testing_threshold = 0.075   # for exectuion of runone()
testing_model_path = './models/testing/best_model_ep14_0-559.pth'
save_niftis = True
