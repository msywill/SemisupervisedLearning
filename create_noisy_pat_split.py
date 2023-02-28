import config
import random

no_train_pat = len(config.train_indices)
print(no_train_pat)
no_noisy_pat = int(0.6 * no_train_pat)
print(no_noisy_pat)
print(no_noisy_pat/no_train_pat)
print(config.train_indices)
random.shuffle(config.train_indices)
print(config.train_indices)
noisy_indices = config.train_indices[:no_noisy_pat]
print(noisy_indices)
print(len(noisy_indices))


#possiblue issue: index 0 is contained in the quality labels
#OUTPUT OF THE RUN USED:
# 51 --> if patient 0 removed exactly 50 --> 60/40% split perfect
# 30
# 0.5882352941176471
# [0, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 19, 22, 27, 29, 31, 32, 33, 34, 36, 37, 38, 40, 41, 42, 45, 46, 47, 48, 49, 51, 53, 54, 56, 58, 59, 60, 61, 63, 65, 67, 69, 71, 72, 73, 75, 77, 78, 79, 82]
# [46, 75, 63, 38, 12, 56, 78, 54, 69, 16, 82, 32, 5, 41, 6, 53, 2, 60, 61, 9, 48, 49, 71, 34, 47, 36, 77, 73, 45, 3, 29, 79, 10, 0, 40, 59, 8, 58, 33, 37, 65, 42, 15, 27, 19, 72, 51, 4, 31, 67, 22]
# [46, 75, 63, 38, 12, 56, 78, 54, 69, 16, 82, 32, 5, 41, 6, 53, 2, 60, 61, 9, 48, 49, 71, 34, 47, 36, 77, 73, 45, 3]
# 30