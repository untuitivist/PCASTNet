from __future__ import annotations

import os
import random
from copy import deepcopy
from collections import defaultdict

from torch.utils.data import DataLoader

from data_loader import CustomImageDataset
from sampler import InfiniteSamplerWrapper
from stc_utils import concat_datasets


def _clone_dataset_with_samples(base_dataset, samples):
    dataset = deepcopy(base_dataset)
    dataset.samples = list(samples)
    return dataset


def build_encoder_runtime_split_dataloaders(stc, batch_size: int = 32):
    """Build encoder train/valid/test loaders from style/train in memory only.

    No split directory, manifest file, symlink, or copied image is created.
    The legacy STC classifier evaluator requires a test loader, so the
    validation holdout is reused as test for encoder pretraining.
    """
    source_dir = stc.style_dataset_dir + '/train'
    print('[+]Load encoder source style train data from:', source_dir)
    base_dataset = CustomImageDataset(source_dir, transform=stc.train_transfrom, scale=int(1e20))

    by_label = defaultdict(list)
    for sample in base_dataset.samples:
        _path, label = sample
        by_label[label].append(sample)

    rng = random.Random(getattr(stc, "encoder_runtime_seed", 42))
    monitored_samples = []
    sample_scale = int(getattr(stc, "encoder_sample_scale", len(base_dataset)) or len(base_dataset))
    if sample_scale <= 0:
        raise ValueError("encoder_sample_scale must be positive.")
    per_class_base, per_class_extra = divmod(sample_scale, len(by_label))
    for idx, label in enumerate(sorted(by_label)):
        samples = list(by_label[label])
        rng.shuffle(samples)
        want = per_class_base + (1 if idx < per_class_extra else 0)
        if want > len(samples):
            raise ValueError(
                f"Requested {want} encoder samples for class {label}, "
                f"but only {len(samples)} are available in {source_dir}."
            )
        monitored_samples.extend(samples[:want])

    by_selected_label = defaultdict(list)
    for sample in monitored_samples:
        _path, label = sample
        by_selected_label[label].append(sample)

    train_samples = []
    valid_samples = []
    ratio = float(getattr(stc, "encoder_train_ratio", 0.8))
    for label in sorted(by_selected_label):
        samples = list(by_selected_label[label])
        rng.shuffle(samples)
        split_at = max(1, int(len(samples) * ratio))
        if split_at >= len(samples) and len(samples) > 1:
            split_at = len(samples) - 1
        train_samples.extend(samples[:split_at])
        valid_samples.extend(samples[split_at:])

    if not valid_samples:
        raise ValueError("Encoder runtime split produced no validation samples.")

    train_dataset = _clone_dataset_with_samples(base_dataset, train_samples)
    val_dataset = _clone_dataset_with_samples(base_dataset, valid_samples)
    test_dataset = _clone_dataset_with_samples(base_dataset, valid_samples)

    print('[+]Encoder runtime split source size:', len(base_dataset), '\tclasses num:', len(base_dataset.class_names))
    print('[+]Encoder monitored sample size:', len(monitored_samples), '\tpaper protocol: only monitored samples')
    print('[+]Encoder runtime train size:', len(train_dataset))
    print(' - Encoder runtime valid size:', len(val_dataset))
    print(' - Encoder runtime test size:', len(test_dataset), '(reuses valid holdout for legacy evaluation)')

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=stc.n_threads),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=stc.n_threads),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=stc.n_threads),
    )


def build_datasets(
    stc,
    class_or_transfer: bool,
    test_classes_or_random: bool = True,
    use_train_datasets: list | None = None,
    use_val_datasets: list | None = None,
    use_test_datasets: list | None = None,
    test_style_num: int | None = None,
    test_content_num: int | None = None,
):
    ttf = stc.train_transfrom
    tsf = stc.train_transfrom
    tcf = stc.train_transfrom
    # tsf = stc.test_style_tf
    # tcf = stc.test_content_tf
    if class_or_transfer:
        train_datasets = []
        val_datasets = []
        test_datasets = []
        if 'content' in use_train_datasets:
            print('[+]Load content train data from: ', stc.content_dataset_dir, '\tsize each class: ', int(stc.content_train_dataset_scale/stc.num_classes))
            content_train_dataset = CustomImageDataset(stc.content_dataset_dir+'/train',
                                        transform=ttf,
                                        scale=int(stc.content_train_dataset_scale/stc.num_classes))
            train_datasets.append(content_train_dataset)
        if 'content' in use_val_datasets or 'content' in use_test_datasets:
            raise ValueError("Paper demo protocol evaluates only monitored/style validation and test data.")

        if 'style' in use_train_datasets:
            print('[+]Load style train data from:', stc.style_dataset_dir, '\tsize each class: ', int(stc.style_train_dataset_scale/stc.num_classes))
            style_train_dataset = CustomImageDataset(stc.style_dataset_dir+'/train',
                                        transform=ttf,
                                        scale=int(stc.style_train_dataset_scale/stc.num_classes))
            train_datasets.append(style_train_dataset)
        if 'style' in use_val_datasets:
            print('[+]Load style valid data from:', stc.style_dataset_dir, '\tsize each class: ', int(stc.style_valid_dataset_scale/stc.num_classes))
            style_val_dataset = CustomImageDataset(stc.style_dataset_dir+'/valid',
                                        transform=ttf,
                                        scale=int(stc.style_valid_dataset_scale/stc.num_classes))
            val_datasets.append(style_val_dataset)
        if 'style' in use_test_datasets:
            print('[+]Load style test data from: ', stc.style_dataset_dir, '\tsize each class: ', int(stc.style_test_dataset_scale/stc.num_classes))
            style_test_dataset = CustomImageDataset(stc.style_dataset_dir+'/test',
                                        transform=tsf,
                                        scale=int(stc.style_test_dataset_scale/stc.num_classes))
            test_datasets.append(style_test_dataset)

        if 'style_transfer' in use_train_datasets:
            assert len(os.listdir(stc.style_transfer_dataset_dir)) != 0
            print('[+]Load style_transfer train data from', stc.style_transfer_dataset_dir, '\tsize each class: ', int((stc.content_train_dataset_scale-stc.style_train_dataset_scale)/stc.num_classes))
            style_transfer_train_dataset = CustomImageDataset(stc.style_transfer_dataset_dir,
                                                transform=ttf,
                                                scale=int((stc.content_train_dataset_scale-stc.style_train_dataset_scale)/stc.num_classes))
            train_datasets.append(style_transfer_train_dataset)

        train_dataset = concat_datasets(train_datasets)
        val_dataset = concat_datasets(val_datasets)
        test_dataset = concat_datasets(test_datasets)

        print('[+]Train dataset size:', len(train_dataset), '\tclasses num: ', len(train_dataset.class_names))
        print(' - Val dataset size:', len(val_dataset), '\tclasses num: ', len(val_dataset.class_names))
        print(' - Test dataset size:', len(test_dataset), '\tclasses num: ', len(test_dataset.class_names))

        return train_dataset, val_dataset, test_dataset

    print('[+]Load content train data from', stc.content_dataset_dir)
    content_train_dataset = CustomImageDataset(stc.content_dataset_dir+'/train',
                                transform=ttf,
                                scale=int(stc.content_train_dataset_scale/stc.num_classes))
    print('[+]Load style train data from', stc.style_dataset_dir)
    style_train_dataset = CustomImageDataset(stc.style_dataset_dir+'/train',
                                transform=ttf,
                                scale=int(stc.style_train_dataset_scale/stc.num_classes))
    print('[+]Load style valid data from', stc.style_dataset_dir)
    style_val_dataset = CustomImageDataset(stc.style_dataset_dir+'/valid',
                                transform=ttf,
                                scale=int(stc.style_valid_dataset_scale/stc.num_classes))
    print('[+]Load style test data from', stc.style_dataset_dir)
    style_test_dataset = CustomImageDataset(stc.style_dataset_dir+'/test',
                                transform=tsf,
                                scale=int(stc.style_test_dataset_scale/stc.num_classes))

    style_dataset = style_train_dataset
    content_dataset = content_train_dataset
    if test_classes_or_random:
        content_test_images = content_train_dataset.each_classes_samples()
        style_test_images = style_test_dataset.each_classes_samples()
    else:
        content_test_images = content_train_dataset.random_samples(test_content_num)
        style_test_images = style_test_dataset.random_samples(test_style_num)
    return content_dataset, style_dataset, content_test_images, style_test_images


def build_classifier_dataloaders(
    stc,
    use_train_datasets: list,
    use_val_datasets: list,
    use_test_datasets: list,
    batch_size: int = 32,
):
    if getattr(stc, "encoder_runtime_split", False):
        return build_encoder_runtime_split_dataloaders(stc, batch_size=batch_size)

    train_dataset, val_dataset, test_dataset = build_datasets(
        stc, True, True, use_train_datasets, use_val_datasets, use_test_datasets
    )

    classifier_train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=stc.n_threads
    )

    classifier_val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=stc.n_threads
    )
    classifier_test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=stc.n_threads
    )

    return classifier_train_loader, classifier_val_loader, classifier_test_loader


def build_style_transfer_iters(
    stc,
    batch_size: int = 32,
    test_classes_or_random: bool = True,
    test_content_num: int = 10,
    test_style_num: int = 10,
):
    content_dataset, style_dataset, content_test_images, style_test_images = build_datasets(
        stc,
        False,
        test_classes_or_random=test_classes_or_random,
        test_content_num=test_content_num,
        test_style_num=test_style_num,
    )

    content_iter = iter(DataLoader(
        content_dataset, batch_size=batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=stc.n_threads))
    style_iter = iter(DataLoader(
        style_dataset, batch_size=batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=stc.n_threads))

    return content_iter, style_iter, content_test_images, style_test_images
