DATASET_CATALOG = {
    "cifar-10_train": [
        "GeneralDataset",
        [
            ["img_dir", "datasets/cifar-10/train"],
            ["anno_file", "datasets/cifar-10/annotations/train.txt"],
        ],
    ],
    "cifar-10_val": [
        "GeneralDataset",
        [
            ["img_dir", "datasets/cifar-10/val"],
            ["anno_file", "datasets/cifar-10/annotations/val.txt"],
        ],
    ],
    "imagenet_train": [
        "ImagenetDataset",
        [["anno_file", "/data/hy-data/dataset/imagenet/imagenet.train.nori.list"],],
    ],
    "imagenet_val": [
        "ImagenetDataset",
        [["anno_file", "/data/hy-data/dataset/imagenet/imagenet.val.nori.list"],],
    ],
}
