import config
import dictionary
import numpy
from tqdm import tqdm
import random
from dictionary import get_path_list_with_indices, create_dict
from itertools import combinations


quality_indices = [29, 79, 10, 40, 59, 8, 58, 37, 65, 42, 15, 27, 19, 72, 51, 4, 31, 67, 22]
noisy_indices = [46, 75, 63, 38, 12, 56, 78, 54, 69, 33, 16, 82, 32, 5, 41, 6, 53, 2, 60, 61, 9, 48, 49, 71, 34, 47, 36, 77, 73, 45, 3]
# print(len(quality_indices)/(len(noisy_indices)+len(quality_indices)))


# just change the number of patient to get different ranges of indices's number

def random_try(ratio, noisy_indices):
    best_error = 0.1
    possible_com = []
    possible_result = []
    for i in tqdm(range(10000)):
        random.shuffle(noisy_indices)
        # print(noisy_indices)
        split = noisy_indices[:25]
        image_path = get_path_list_with_indices(config.parent_dir + '/train/*', split)
        image_dict = create_dict(image_path, "labels")
        number_slices = len(image_dict)
        result = 1740 / number_slices
        error = abs(result - ratio)
        # print(number_slices, error)
        if error < best_error:
            best_error = error
            possible_com.append(split)
            possible_result.append(result)

    return possible_result, possible_com


result, com = random_try(4/5, noisy_indices)
print(result, com)
print(result.pop(), com.pop())


# combo = list(combinations(noisy_indices, 14))
# print(len(combo))
# ignore the below thing


def get_closest_ratio(ratio_1, ratio_2, combination):
    print('the requiring ratio = ', ratio_1, ratio_2)
    possible_com_1 = []
    possible_result_1 = []
    # possible_com_2 = []
    # possible_result_2 = []
    best_error_1 = 0.2
    # best_error_2 = 0.2
    for combo in tqdm(combination):
        noisy_indices = list(combo)
        image_path = get_path_list_with_indices(config.parent_dir + '/train/*', noisy_indices)
        image_dict = create_dict(image_path, "labels")
        number_slices = len(image_dict)
        print(number_slices)
        result = 1740/number_slices

        error_1 = abs(result-ratio_1)
        # print(number_slices, error_1)
        if error_1 < best_error_1:
            best_error_1 = error_1
            possible_com_1.append(combo)
            possible_result_1.append(result)

        # error_2 = abs(result-ratio_2)
        # print(number_slices, error_2)
        # if error_2 < best_error_2:
        #    best_error_2 = error_2
        #    possible_com_2.append(combo)
        #    possible_result_2.append(result)


    return possible_com_1, possible_result_1


#com_1, ratio_1 = get_closest_ratio(4/3, 1, combo)
#print(com_1, ratio_1)





#no_train_pat = config.train_indices
#print(len(no_train_pat))


#c=[x for x in config.train_indices if x not in (quality_indices+noisy_indices)]
#d=[y for y in c if y not in noisy_indices]
#print(c)
#print(d)