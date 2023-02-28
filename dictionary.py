import glob
import config


def get_path_list(dir):
    pathlist = []
    for path in glob.iglob(dir):
        pathlist.append(path)
    return sorted(pathlist)


# call function with eg config.lq_indices to get all paths of patients with low-quality/noisy labels
def get_path_list_with_indices(dir, indices):
    dir = dir[:-1] + "PATIENT_00"
    pathlist = []
    for patient_id in indices:

        if patient_id < 10:
            patient_id = "0" + str(patient_id)
        else:
            patient_id = str(patient_id) 

        path = dir + patient_id
        pathlist.append(path)

    return sorted(pathlist)


def create_dict(patient_path_list, mode="labels"):
    assert mode in ["labels", "noisy_labels", "noisy_labels_light", "noisy_labels_heavy", "noisy_labels_semiheavy"]
    dict = []
    for patient in patient_path_list:
        image_path_list = get_path_list(patient + '/images/*.npz')
        label_path_list = get_path_list(patient + '/' + mode + '/*.npz')

        for image, label in zip(image_path_list, label_path_list):
            dict.append({'image': image, 'label': label})

    return dict


def get_dictionaries_baseline():

    train_path = get_path_list(config.parent_dir + '/train/*')
    val_path = get_path_list(config.parent_dir + '/val/*')

    train_dict = create_dict(train_path)
    val_dict = create_dict(val_path)

    return train_dict, val_dict


# switch between different noisy labels, call the get_dictionaries_mean_teacher function with noisy_type parameter
# when no specified noisy_type, by default is noisy_labels
def get_dictionaries_mean_teacher(noisy_type="noisy_labels"):
    lq_patients = config.lq_indices
    hq_patients = config.hq_indices
    
    lq_patients_train_path = get_path_list_with_indices(config.parent_dir + '/train/*', lq_patients) # return path list of patient for hq and lq
    hq_patients_train_path = get_path_list_with_indices(config.parent_dir + '/train/*', hq_patients)
    val_path = get_path_list(config.parent_dir + '/val/*')

    hq_train_dict = create_dict(hq_patients_train_path, "labels")

    assert noisy_type in ["noisy_labels", "noisy_labels_light", "noisy_labels_heavy", "noisy_labels_semiheavy"]
    lq_train_dict = create_dict(lq_patients_train_path, noisy_type)

    val_dict = create_dict(val_path)

    return lq_train_dict, hq_train_dict, val_dict


def generate_test_paths():
    patient_dir = []
    start = config.parent_dir + '/test/PATIENT_00'
    for patient_id in config.test_indices:
        if patient_id < 10:
            patient_id = "0" + str(patient_id)
        path = start + str(patient_id)
        patient_dir.append(path)

    return patient_dir



