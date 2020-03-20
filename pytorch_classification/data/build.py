import torch
import torchvision
from pytorch_classification.config.dataset_catalog import DATASET_CATALOG
from pytorch_classification.data import datasets as D
from pytorch_classification.data import transforms as T
from pytorch_classification.utils.comm import get_world_size
from torch.utils.data.dataset import ConcatDataset


def build_dataset(dataset_list, transforms, is_train):
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError("dataset_list should be a list of strings, got {}".format(dataset_list))

    datasets = []

    for dataset_name in dataset_list:
        dataset_module_name = DATASET_CATALOG[dataset_name][0]
        dataset_module = getattr(D, dataset_module_name)

        args = dict(DATASET_CATALOG[dataset_name][1])
        args["transforms"] = transforms

        datasets.append(dataset_module(**args))

    dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    return dataset


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_data_loader(cfg, is_train=True, is_distributed=False):
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST

    transforms = T.build_transforms(cfg, is_train=is_train)
    datasets = build_dataset(dataset_list, transforms, is_train)

    sampler = make_data_sampler(datasets, is_train, is_distributed)

    num_gpus = get_world_size()
    batch_size = cfg.SOLVER.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE
    batch_size = batch_size // num_gpus

    data_loader = torch.utils.data.DataLoader(
        datasets,
        batch_size=batch_size,
        num_workers=cfg.WORKERS,
        pin_memory=True,
        sampler=sampler,
    )

    return data_loader
