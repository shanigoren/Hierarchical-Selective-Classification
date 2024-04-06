import os
import pickle
import shutil
import tarfile
from glob import glob
from itertools import combinations

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler, BatchSampler, DistributedSampler

from .custom_dataset import CustomDataset, DummyDataset, get_open_img_transforms
from .log_utils import Timer
# from .project_paths import ImageNet_21K_METADATA, ImageNet_20K_METADATA, ImageNet_21K_BASE_FOLDER, \
# ImageNet_1K_METADATA, ImageNet_1K_BASE_FOLDER, RESULTS_AND_STATS_BASE_FOLDER, ALL_21K_TARS_DIR, ImageNet_O_METADATA, \
# RESULTS_AND_STATS_IM_O_BASE_FOLDER, ImageNet_O_BASE_FOLDER


def save_pickle(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not path.endswith('.pkl'):
        path += '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    obj = None
    if not path.endswith('.pkl'):
        path += '.pkl'
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)

    return obj


def load_ds_metadata(dataset_name):
    """
    :param dataset_name: the dataset to load metadata for
    :return: a dictionary of metadata
    that most importantly contains contains instances paths and labels.
    """

    metadata_path = {'ImageNet_21K': ImageNet_21K_METADATA, 'ImageNet_20K': ImageNet_20K_METADATA,
                     'ImageNet_1K': ImageNet_1K_METADATA}[dataset_name]

    metadata = load_pickle(metadata_path)
    return metadata


def load_ds_img_paths_and_labels(dataset_name, ds_subset=None):
    """
    :param dataset_name: dataset to load metadata for.
    :param ds_subset: the subset of the metadata (e.g. 'training', 'validation', 'test').
    (relevant for ImageNet_1K)
    :return: tuple: img_paths, labels
    """
    metadata_path = {'ImageNet_21K': ImageNet_21K_METADATA, 'ImageNet_20K': ImageNet_20K_METADATA,
                     'ImageNet_1K': ImageNet_1K_METADATA, 'ImageNet_O': ImageNet_O_METADATA}[dataset_name]

    metadata = load_pickle(metadata_path)
    img_paths = norm_paths(dataset_name, metadata['image_files'])
    labels = metadata['labels']

    if ds_subset is not None:
        subset_idx = metadata[ds_subset + '_idx']
        img_paths = img_paths[subset_idx]
        labels = labels[subset_idx]

    return img_paths, labels


def norm_paths(dataset_name, img_paths):
    if dataset_name not in ('ImageNet_21K', 'ImageNet_20K', 'ImageNet_1K', 'ImageNet_O'):
        raise ValueError(f'dataset {dataset_name} not supported yet.')

    base_folders_mappings = {'ImageNet_21K': (ImageNet_21K_BASE_FOLDER, 'fall11_whole/'),
                             'ImageNet_20K': (ImageNet_21K_BASE_FOLDER, 'fall11_whole/'),
                             'ImageNet_1K': (ImageNet_1K_BASE_FOLDER, 'ImageNet_1K/'),
                             'ImageNet_O': (ImageNet_O_BASE_FOLDER, 'imagenet-o/')}

    new_base_folder, old_base_folder = base_folders_mappings[dataset_name]
    img_paths = [os.path.join(new_base_folder, img.split(old_base_folder)[1]) for img in img_paths]

    return np.array(img_paths)


def combine_datasets(dataset_names, ds_subsets):
    all_img_paths = []
    all_labels = []
    offset = 0
    for dataset_name, ds_subset in zip(dataset_names, ds_subsets):
        img_paths, labels = load_ds_img_paths_and_labels(dataset_name, ds_subset)
        classes, labels = np.unique(labels, return_inverse=True)
        labels += offset
        offset += len(classes)
        all_labels.extend(labels)

        all_img_paths.extend(img_paths)

    return np.array(all_img_paths), np.array(all_labels)


sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def create_data_loader(dataset_names, ds_subsets, batch_size, num_workers, shuffle=False, offset=0, num_shots=None,
                       transform=None, sampler_opt=None):
    """
    :return: returns a PyTorch data loader.
    """

    if dataset_names[0].startswith('random'):
        # TODO: don't forget to delete this
        _, num_samples_per_class, num_classes = dataset_names[0].split('_')
        subset = DummyDataset(int(num_samples_per_class), int(num_classes), transform)
    else:
        if isinstance(dataset_names, list):
            img_paths, labels = combine_datasets(dataset_names, ds_subsets)

            weights = 1. / np.unique(labels, return_counts=True)[1]
            weights = weights[labels]
        else:
            img_paths, labels = load_ds_img_paths_and_labels(dataset_names, ds_subsets)
            _, labels = np.unique(labels, return_inverse=True)
            weights = 1. / np.unique(labels, return_counts=True)[1]
            weights = weights[labels]
            labels += offset

        print(f'loaded dataset will have {len(labels)} samples.')
        assert len(img_paths) == len(labels)
        subset = CustomDataset(img_paths, labels, num_shots, transform)

    sampler = None
    batch_sampler = None
    sampler_type = sampler_opt['sampler_type'] if sampler_opt is not None else None
    if sampler_type == 'balanced_sampler':
        # uniform sampling i.e. each class has an equal opportunity to be drawn
        sampler = WeightedRandomSampler(weights, len(weights))
        shuffle = False

    if sampler_type == 'batched_balanced_sampler':
        # same as balanced_sampler but each batch is of fixed size
        batch_sampler = BatchSampler(WeightedRandomSampler(weights, len(weights)), batch_size, drop_last=True)
        shuffle = False
        batch_size = 1

    worker_init_fn = None
    if sampler_type == 'distributed':
        sampler = DistributedSampler(subset, shuffle=shuffle)
        worker_init_fn = set_worker_sharing_strategy

    # noinspection PyArgumentList
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                                                num_workers=num_workers, sampler=sampler,
                                                batch_sampler=batch_sampler, worker_init_fn=worker_init_fn)

    return subset_loader


def save_h5(save_path, dictionary):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    f = h5py.File(save_path, "w")
    for k, v in dictionary.items():
        f.create_dataset(k, data=v, compression="gzip")
    f.close()


def load_h5(data_path):
    hf = h5py.File(data_path, 'r')

    d = dict()
    for k in hf.keys():
        # with Timer(f'time to load {k}:', print_human_readable=False):
        d[k] = hf.get(k)

    return d


def test_save_and_load_times(test_store):
    folder = '/media/mohammed/Elements1/tmp'

    if test_store:
        base_centroids = np.random.rand(1000, 8192)
        counts = np.random.randint(0, 1000, 1000)
        d = {'base_centroids': base_centroids, 'base_counts': counts}
        with Timer('pkl_time store:', print_human_readable=False):
            save_pickle(os.path.join(folder, 'centroids.pkl'), d)

        with Timer('h5_time store:', print_human_readable=False):
            save_h5(os.path.join(folder, 'centroids.h5'), d)
    else:
        with Timer('pkl_time load:', print_human_readable=False):
            d = load_pickle(os.path.join(folder, 'centroids.pkl'))

        with Timer('h5_time load:', print_human_readable=False):
            d = load_h5(os.path.join(folder, 'centroids.h5'))

    if test_store:
        confidences = {'softmax_conf': np.random.rand(18000), 'entropy_conf': np.random.rand(18000),
                       'correct': np.random.randint(0, 2, 18000), 'dists_conf': np.random.rand(18000),
                       'predictions': np.random.randint(0, 1000, 18000), 'labels': np.random.randint(0, 1000, 18000)}

        stats = {'attackers_centroids': np.random.rand(1700, 8192),
                 'attackers_class_counts': np.random.randint(0, 1000, 17000),
                 'attackers_avg_softmax': np.random.rand(1700),
                 }
        stats.update(confidences)
        with Timer('pkl_time store:', print_human_readable=False):
            save_pickle(os.path.join(folder, 'stats.pkl'), stats)

        with Timer('h5_time store:', print_human_readable=False):
            save_h5(os.path.join(folder, 'stats.h5'), stats)
    else:
        with Timer('pkl_time load:', print_human_readable=False):
            stats = load_pickle(os.path.join(folder, 'stats.pkl'))

        with Timer('h5_time load:', print_human_readable=False):
            stats = load_h5(os.path.join(folder, 'stats.h5'))


def save_model_results(model_name, data, data_name, is_im_O=False):
    if not is_im_O:
        save_path = os.path.join(RESULTS_AND_STATS_BASE_FOLDER, model_name, data_name)
    else:
        save_path = os.path.join(RESULTS_AND_STATS_IM_O_BASE_FOLDER, model_name, data_name)
    save_pickle(save_path, data)


def load_model_results(model_name, data_name, is_im_O=False):
    if not is_im_O:
        save_path = os.path.join(RESULTS_AND_STATS_BASE_FOLDER, model_name, data_name)
    else:
        save_path = os.path.join(RESULTS_AND_STATS_IM_O_BASE_FOLDER, model_name, data_name)
    data = load_pickle(save_path)
    return data


def load_model_results_df(model_name, data_name):
    load_path = os.path.join(RESULTS_AND_STATS_BASE_FOLDER, model_name, data_name)
    data = pd.read_csv(load_path)
    return data


def save_model_results_df(df: pd.DataFrame, model_name, data_name):
    save_path = os.path.join(RESULTS_AND_STATS_BASE_FOLDER, model_name, data_name)
    df.to_csv(save_path, index=False)


def split_dataset(labels, percentage):
    train_idx = []
    val_idx = []
    classes = np.unique(labels)
    classes_samples = {c: [] for c in classes}

    for s, l in enumerate(labels):
        classes_samples[l].append(s)

    for c in classes:
        class_samples = np.random.permutation(np.array(classes_samples[c]))
        num_train_samples = int(percentage * len(class_samples))
        assert len(class_samples) == 200
        assert num_train_samples == 150
        train_idx.extend(class_samples[:num_train_samples])
        val_idx.extend(class_samples[num_train_samples:])

    return np.array(train_idx), np.array(val_idx)


def check_corruption():
    all_content = glob(f'{ImageNet_21K_BASE_FOLDER}/*')
    class_folders = [d for d in all_content if os.path.isdir(d)]
    # class_folders = np.sort(class_folders)
    corrupt_images_log = './corrupt_images_log3.txt'
    corrupt_images = './corrupt_images3.txt'
    num_classes = len(class_folders)
    problematic_images = []
    num_images_per_class = []
    open_t = get_open_img_transforms()
    for i, class_folder in enumerate(class_folders):
        class_files = glob(f'{class_folder}/*.JPEG')
        num_images_per_class.append(len(class_files))

        if num_images_per_class[-1] != 200:
            with open('./global_corruption_logger.txt', mode='a') as f:
                print(f'class {class_folder} has {num_images_per_class[-1]} samples!')

        for img_path in class_files:
            try:
                img = open_t(img_path)
            except Exception as e:
                problematic_images.append(img_path)
                print(f'couldnt open {img_path} because of {e}')
                with open(corrupt_images_log, mode='a') as f:
                    print(f'couldnt open {img_path} because of {e}', file=f)

                with open(corrupt_images, mode='a') as f:
                    print(f'{img_path}', file=f)

            img.close()

        print(f'finished class {i}/{num_classes}')


def replace_corrupt_files():
    with open('../corrupt_images2.txt') as f:
        corrupt_images_orig = f.readlines()

    tmp_folder = './temp_tar_problematic_classes'
    corrupt_images = np.unique([c.replace('\n', '') for c in corrupt_images_orig])
    corrupt_images = [os.path.join(tmp_folder, img.split('fall11_whole/')[1]) for img in corrupt_images]

    corrupt_images_classes = [os.path.basename(os.path.dirname(c)) for c in corrupt_images]
    corrupt_images_classes, counts = np.unique(corrupt_images_classes, return_counts=True)

    all_tars_dir = ALL_21K_TARS_DIR
    os.makedirs(tmp_folder, exist_ok=True)
    all_tar_files = glob(f'{all_tars_dir}/*.tar')
    open_t = get_open_img_transforms()

    good_samples = []
    # corrupt_images_classes = corrupt_images_classes[[4,6, 7]]
    for idx, (class_wid, num_to_replace) in enumerate(zip(corrupt_images_classes, counts)):
        shutil.copy(f'{all_tars_dir}/{class_wid}.tar', tmp_folder)

        current_class_samples = glob(f'{ImageNet_21K_BASE_FOLDER}/{class_wid}/*.JPEG')
        current_class_samples = [os.path.join(tmp_folder, img.split('fall11_whole/')[1]) for img in
                                 current_class_samples]
        print(f'{len(current_class_samples)}')

        tar_file = os.path.join(tmp_folder, f'{class_wid}.tar')
        destination = extract_tar(tar_file)
        all_class_samples = glob(os.path.join(destination, '*.JPEG'))
        print(f'{len(all_class_samples)}')

        valid_samples = np.setdiff1d(all_class_samples, current_class_samples)
        valid_samples = np.setdiff1d(valid_samples, corrupt_images)
        counter = 0
        for valid_replacement in valid_samples:
            try:
                open_t(valid_replacement)
                good_samples.append(valid_replacement)
                counter += 1
                if counter == num_to_replace:
                    break
            except Exception as e:
                print(e)

    for s in good_samples:
        class_wid = os.path.basename(os.path.dirname(s))
        class_dir = os.path.join(ImageNet_21K_BASE_FOLDER, class_wid)
        shutil.copy(s, class_dir)

    for s in corrupt_images:
        img = os.path.basename(s)
        class_wid = os.path.basename(os.path.dirname(s))
        class_dir = os.path.join(ImageNet_21K_BASE_FOLDER, class_wid)
        img_path = os.path.join(class_dir, img)
        os.remove(img_path)


def extract_tar(tar_file_path, destination_folder_path=None):
    """
    extracts the given tarfile to the given destination.
    if target_folder_path is None then the contents will be saved in a newly created folder
    with the same name as the tarfile in the same location as the tarfile
    :param tar_file_path: tar file path to extract
    :param destination_folder_path: the folder where the contents of the tarfile will be saved
    :return: destination_folder_path.
    """

    if destination_folder_path is None:
        destination_folder_path = tar_file_path[:-4]
        os.makedirs(destination_folder_path, exist_ok=True)

    with tarfile.open(tar_file_path) as my_tar:
        my_tar.extractall(destination_folder_path)

    return destination_folder_path


if __name__ == '__main__':
    # test_save_and_load_times(test_store=False)
    # check_corruption()
    print('running data_utils')
    # replace_corrupt_files()
