# adapted from cleanlabs pruning.py

import cupy as cp
import multiprocessing
from tensorflow.math import confusion_matrix

MIN_NUM_PER_CLASS = 5

def value_counts(x):
    try:
        return x.value_counts()
    except:
        if type(x[0]) is int and (cp.array(x) >= 0).all():
            return cp.bincount(x)
        else:
            return cp.unique(x, return_counts=True)[1]

def round_preserving_row_totals(confident_joint):
    return cp.apply_along_axis(
        func1d=round_preserving_sum,
        axis=1,
        arr=confident_joint,
    ).astype(int)

def round_preserving_sum(iterable):
   
    floats = cp.asarray(iterable, dtype=float)
    ints = floats.round()
    orig_sum = cp.sum(floats).round()
    int_sum = cp.sum(ints).round()
    # Adjust the integers so that they sum to orig_sum
    while abs(int_sum - orig_sum) > 1e-6:
        diff = cp.round(orig_sum - int_sum)
        increment = -1 if int(diff < 0.) else 1
        changes = min(int(abs(diff)), len(iterable))
        # Orders indices by difference. Increments # of changes.
        indices = cp.argsort(floats - ints)[::-increment][:changes]
        for i in indices:
            ints[i] = ints[i] + increment
        int_sum = cp.sum(ints).round()
    return ints.astype(int)

def calibrate_confident_joint(confident_joint, s, multi_label=False):

    if multi_label:
        s_counts = value_counts([x for lst in s for x in lst])
    else:
        s_counts = value_counts(s)
    # Calibrate confident joint to have correct p(s) prior on noisy labels.
    calibrated_cj = (
            confident_joint.T / confident_joint.sum(axis=1) * s_counts
    ).T
    # Calibrate confident joint to sum to:
    # The number of examples (for single labeled datasets)
    # The number of total labels (for multi-labeled datasets)
    calibrated_cj = calibrated_cj / cp.sum(calibrated_cj) * sum(s_counts)
    return round_preserving_row_totals(calibrated_cj)

def compute_confident_joint(
    s,
    psx,
    K=None,
    thresholds=None,
    calibrate=True,
    multi_label=False,
    return_indices_of_off_diagonals=False,
):

    # s needs to be a numpy array
    s = cp.asarray(s)

    # Find the number of unique classes if K is not given
    
    K = len(cp.unique(s))

    # Estimate the probability thresholds for confident counting
    if thresholds is None:
        # P(we predict the given noisy label is k | given noisy label is k)
        thresholds = [cp.mean(psx[:, k][s == k]) for k in range(K)]
    thresholds = cp.asarray(thresholds)


    # psx_bool is a bool matrix where each row represents a training example as
    # a boolean vector of size K, with True if the example confidently belongs
    # to that class and False if not.
    psx_bool = (psx >= thresholds - 1e-6)
    num_confident_bins = psx_bool.sum(axis=1)
    at_least_one_confident = num_confident_bins > 0
    more_than_one_confident = num_confident_bins > 1
    psx_argmax = psx.argmax(axis=1)
    # Note that confident_argmax is meaningless for rows of all False
    confident_argmax = psx_bool.argmax(axis=1)
    # For each example, choose the confident class (greater than threshold)
    # When there is 2+ confident classes, choose the class with largest prob.
    true_label_guess = cp.where(
        more_than_one_confident,
        psx_argmax,
        confident_argmax,
    )
    # y_confident omits meaningless all-False rows
    y_confident = true_label_guess[at_least_one_confident]
    s_confident = s[at_least_one_confident]
    confident_joint = confusion_matrix(y_confident, s_confident).T
    
    if calibrate:
        confident_joint = calibrate_confident_joint(confident_joint, s)

    if return_indices_of_off_diagonals:
        y_neq_s = y_confident != s_confident
        indices = cp.arange(len(s))[at_least_one_confident][y_neq_s]

        return confident_joint, indices

    return confident_joint

def keep_at_least_n_per_class(prune_count_matrix, n, frac_noise=1.0):

    prune_count_matrix_diagonal = cp.diagonal(prune_count_matrix)

    # Set diagonal terms less than n, to n.
    new_diagonal = cp.maximum(prune_count_matrix_diagonal, n)

    # Find how much diagonal terms were increased.
    diff_per_col = new_diagonal - prune_count_matrix_diagonal

    # Count non-zero, non-diagonal items per column
    # np.maximum(*, 1) makes this never 0 (we divide by this next)
    num_noise_rates_per_col = cp.maximum(
        cp.count_nonzero(prune_count_matrix, axis=0) - 1.,
        1.,
    )

    # Uniformly decrease non-zero noise rates by the same amount
    # that the diagonal items were increased
    new_mat = prune_count_matrix - diff_per_col / num_noise_rates_per_col

    # Originally zero noise rates will now be negative, fix them back to zero
    new_mat[new_mat < 0] = 0

    # Round diagonal terms (correctly labeled examples)
    cp.fill_diagonal(new_mat, new_diagonal)

    # Reduce (multiply) all noise rates (non-diagonal) by frac_noise and
    # increase diagonal by the total amount reduced in each column
    # to preserve column counts.
    new_mat = reduce_prune_counts(new_mat, frac_noise)

    # These are counts, so return a matrix of ints.
    return round_preserving_row_totals(new_mat).astype(int)

def reduce_prune_counts(prune_count_matrix, frac_noise=1.0):


    new_mat = prune_count_matrix * frac_noise
    cp.fill_diagonal(new_mat, prune_count_matrix.diagonal())
    cp.fill_diagonal(new_mat, prune_count_matrix.diagonal() +
                     cp.sum(prune_count_matrix - new_mat, axis=0))

    # These are counts, so return a matrix of ints.
    return new_mat.astype(int)

def _prune_by_class(k, args=None):

    if args:  # Single processing - params are passed in
        s, s_counts, prune_count_matrix, psx, multi_label = args

    if s_counts[k] > MIN_NUM_PER_CLASS:  # No prune if not MIN_NUM_PER_CLASS
        num_errors = s_counts[k] - prune_count_matrix[k][k]
        # Get rank of smallest prob of class k for examples with noisy label k
        s_filter = cp.array(
            [k in lst for lst in s]) if multi_label else s == k
        class_probs = psx[:, k]
        rank = cp.partition(class_probs[s_filter], num_errors)[num_errors]
        return s_filter & (class_probs < rank)
    else:
        return cp.zeros(len(s), dtype=bool)

def _prune_by_count(k, args=None):

    if args:  # Single processing - params are passed in
        s, s_counts, prune_count_matrix, psx, multi_label = args

    noise_mask = cp.zeros(len(psx), dtype=bool)
    psx_k = psx[:, k]
    K = len(s_counts)
    if s_counts[k] <= MIN_NUM_PER_CLASS:  # No prune if not MIN_NUM_PER_CLASS
        return cp.zeros(len(s), dtype=bool)
    for j in range(K):  # j is true label index (k is noisy label index)
        num2prune = prune_count_matrix[j][k]
        # Only prune for noise rates, not diagonal entries
        if k != j and num2prune > 0:
            # num2prune'th largest p(true class k) - p(noisy class k)
            # for x with true label j
            margin = psx[:, j] - psx_k
            s_filter = cp.array(
                [k in lst for lst in s]
            ) if multi_label else s == k
            cut = -cp.partition(-margin[s_filter], num2prune - 1)[num2prune - 1]
            noise_mask = noise_mask | (s_filter & (margin >= cut))
    return noise_mask


def get_noise_indices_gpu(s, psx, prune_method='prune_by_noise_rate', inverse_noise_matrix=None,
        confident_joint=None,
        frac_noise=1.0,
        num_to_remove_per_class=None,
        sorted_index_method=None,
        multi_label=False,
        n_jobs=None,
        verbose=0,):


    # Set-up number of multiprocessing threads
    #n_jobs = multiprocessing.cpu_count()
    n_jobs = 1
    
    # Number of examples in each class of s
    s_counts = value_counts(s)
    # Number of classes s
    K = len(psx.T)
    # Boolean set to true if dataset is large
    big_dataset = K * len(s) > 1e8
    # Ensure labels are of type cp.array()
    s = cp.asarray(s)

    confident_joint = compute_confident_joint(
        s=s,
        psx=psx,
        multi_label=None,
    )

    # Leave at least MIN_NUM_PER_CLASS examples per class.
    # NOTE prune_count_matrix is transposed (relative to confident_joint)
    prune_count_matrix = keep_at_least_n_per_class(
        prune_count_matrix=confident_joint.T,
        n=MIN_NUM_PER_CLASS,
        frac_noise=frac_noise,
    )

    if num_to_remove_per_class is not None:
        # Estimate joint probability distribution over label errors
        psy = prune_count_matrix / cp.sum(prune_count_matrix, axis=1)
        noise_per_s = psy.sum(axis=1) - psy.diagonal()
        # Calibrate s.t. noise rates sum to num_to_remove_per_class
        tmp = (psy.T * num_to_remove_per_class / noise_per_s).T
        cp.fill_diagonal(tmp, s_counts - num_to_remove_per_class)
        prune_count_matrix = round_preserving_row_totals(tmp)

    # if n_jobs > 1:  # Prepare multiprocessing shared data
    #     if multi_label:
    #         _s = RawArray('I', int2onehot(s).flatten())
    #     else:
    #         _s = RawArray('I', s)
    #     _s_counts = RawArray('I', s_counts)
    #     _prune_count_matrix = RawArray(
    #         'I', prune_count_matrix.flatten())
    #     _psx = RawArray(
    #         'f', psx.flatten())
    # else:  # Multiprocessing is turned off. Create tuple with all parameters
    args = (s, s_counts, prune_count_matrix, psx, multi_label)

    # Perform Pruning with threshold probabilities from BFPRT algorithm in O(n)
    # Operations are parallelized across all CPU processes
    if prune_method == 'prune_by_class' or prune_method == 'both':
        # if n_jobs > 1:  # parallelize
        #     with multiprocessing_context(
        #             n_jobs,
        #             initializer=_init,
        #             initargs=(_s, _s_counts, _prune_count_matrix,
        #                       prune_count_matrix.shape, _psx, psx.shape,
        #                       multi_label),
        #     ) as p:
        #         if verbose:
        #             print('Parallel processing label errors by class.')
        #         sys.stdout.flush()
        #         if big_dataset and tqdm_exists:
        #             noise_masks_per_class = list(
        #                 tqdm.tqdm(p.imap(_prune_by_class, range(K)), total=K),
        #             )
        #         else:
        #             noise_masks_per_class = p.map(_prune_by_class, range(K))
        #else:  # n_jobs = 1, so no parallelization
        #     noise_masks_per_class = [_prune_by_class(k, args) for k in range(K)]
        # label_errors_mask = np.stack(noise_masks_per_class).any(axis=0)

        noise_masks_per_class = [_prune_by_class(k, args) for k in range(K)]
        label_errors_mask = cp.stack(noise_masks_per_class).any(axis=0)

    if prune_method == 'both':
        label_errors_mask_by_class = label_errors_mask

    if prune_method == 'prune_by_noise_rate' or prune_method == 'both':
        # if n_jobs > 1:  # parallelize
        #     with multiprocessing_context(
        #             n_jobs,
        #             initializer=_init,
        #             initargs=(_s, _s_counts, _prune_count_matrix,
        #                       prune_count_matrix.shape, _psx, psx.shape,
        #                       multi_label),
        #     ) as p:
        #         if verbose:
        #             print('Parallel processing label errors by noise rate.')
        #         sys.stdout.flush()
        #         if big_dataset and tqdm_exists:
        #             noise_masks_per_class = list(
        #                 tqdm.tqdm(p.imap(_prune_by_count, range(K)), total=K)
        #             )
        #         else:
        #             noise_masks_per_class = p.map(_prune_by_count, range(K))
        #else:  # n_jobs = 1, so no parallelization
        noise_masks_per_class = [_prune_by_count(k, args) for k in range(K)]
        label_errors_mask = cp.stack(noise_masks_per_class).any(axis=0)

    if prune_method == 'both':
        label_errors_mask = label_errors_mask & label_errors_mask_by_class
    
    pred = psx.argmax(axis=1)
    for i, pred_label in enumerate(pred):
        if multi_label and cp.all(pred_label == s[i]) or \
                not multi_label and pred_label == s[i]:
            label_errors_mask[i] = False

    return label_errors_mask
