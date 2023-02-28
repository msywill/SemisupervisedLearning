import numpy as np
import torch
import cleanlab
import time

import config


def ssdm(teacher_output, student_output, noisy_labels, CELoss, FocalLoss):
    t_0 = time.time()
    with torch.no_grad():
        model_output_two_probs = torch.cat((torch.sigmoid(teacher_output), 1 - torch.sigmoid(teacher_output)), dim=1)     # (6, 2, 512, 512); logits for labels 1 and 0
        softmaxed_values = model_output_two_probs.cpu().detach().numpy()    # (6, 2, 512, 512); probabilities for two classes 1 and 0

    # reshape the noisy_label and teacher_predicted_prob_map into (n,) and (n, 2), n = 6 * 512 * 512 = 1,572,864
    predictions_swapped = np.swapaxes(np.swapaxes(softmaxed_values, 1, 2), 2, 3)    # (6, 512, 512, 2)
    predicted_2d = predictions_swapped.reshape(-1, 2)   # (n, 2)
    predicted_2d[:, 0], predicted_2d[:, 1] = predicted_2d[:, 1], predicted_2d[:, 0].copy()

    noisy_labels = noisy_labels.cpu().detach().numpy()
    noisy_labels_1d = noisy_labels.reshape(-1).astype(np.int8)     # (n,)

    assert noisy_labels_1d.shape[0] == predicted_2d.shape[0]

    labels_to_change = cleanlab.pruning.get_noise_indices(s=noisy_labels_1d, psx=predicted_2d, prune_method=config.ssdm_prune_method)
    labels_to_change = labels_to_change.reshape(-1, 1, 512, 512).astype(np.int8)    # reshape to (6, 1, 512, 512) -> original shape

    tau = config.ssdm_smoothing_tau   # smoothing argument HYPERPARAMETER

    # from paper, formular (3)
    updated_labels = noisy_labels + labels_to_change * np.power(-1, noisy_labels) * tau
    updated_labels = torch.from_numpy(updated_labels).to(config.device)

    # "the CL loss loss_cl is composed of cross-entropy loss and focal loss with equal weights"
    ce_loss = CELoss(torch.squeeze(torch.sigmoid(student_output)), torch.squeeze(updated_labels)) / updated_labels.shape[0]
    focal_loss = FocalLoss(student_output, updated_labels)

    cl_loss = ce_loss + focal_loss

    t_1 = time.time()

    return cl_loss, labels_to_change.sum(), t_1 - t_0

