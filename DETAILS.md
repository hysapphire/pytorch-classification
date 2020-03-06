## DETAILS
Some details about building your own image classification model with custom datasets will be introduced in this part. Creating new modules is also included in this part.


### File Tree
```
pytorch-classification
├── configs
│   ├── config_cifar10_R50_1gpu.yaml
│   ├── config_cifar10_R50_4gpu.yaml
│   ├── config_cifar10_R50_8gpu.yaml
├── demo
│   ├── demo.py
│   ├── predictor.py
├── DETAILS.md
├── LICENSE
├── pytorch_classification
│   ├── config
│   │   ├── dataset_catalog.py
│   │   ├── defaults.py
│   │   ├── __init__.py
│   ├── data
│   │   ├── build.py
│   │   ├── datasets
│   │   │   ├── general.py
│   │   │   ├── __init__.py
│   │   │   ├── new_dataset.py
│   │   ├── __init__.py
│   │   └── transforms
│   │       ├── build.py
│   │       ├── __init__.py
│   │       └── transforms.py
│   ├── engine
│   │   ├── tester.py
│   │   └── trainer.py
│   ├── __init__.py
│   ├── loss
│   │   ├── build.py
│   │   ├── __init__.py
│   │   ├── loss.py
│   │   ├── new_loss.py
│   ├── network
│   │   ├── build.py
│   │   ├── __init__.py
│   │   ├── network.py
│   │   ├── new_network.py
│   │   └── resnet.py
│   ├── solver
│   │   ├── __init__.py
│   │   └── solver.py
│   └── utils
│       ├── checkpoint.py
│       ├── comm.py
│       ├── __init__.py
│       ├── logger.py
│       ├── metric_logger.py
│       ├── miscellaneous.py
├── README.md
├── requirements.txt
└── tools
    ├── __init__.py
    ├── test.py
    └── train.py
```


### Config
The default config is in `./pytorch_classification/config/defaults.py`. You can create your own `config.yaml` to override the parameters follow the examples in `./configs`. 
```python
from yacs.config import CfgNode as CN

_C = CN()

# The number of classes of the datasets.
_C.NUM_CLASS = 10
# The output dir.
_C.OUTPUT_DIR = ""
# The path to the checkpoint which is used for resume training or accuracy evaluation.
# If is null, resume training from last-checkpoint.pth or test accuracy by the
# final-model.pth in the OUTPUT_DIR.
_C.CHECKPOINT = ""
# Epoch period for save the checkpoint.
_C.SAVE_EPOCH_PERIOD = 10
# Iteration period for print the train logs.
_C.PRINT_ITER_PERIOD = 20
# Epoch period for test the model while training.
_C.TEST_EPOCH_PERIOD = 10
# Set the visible GPUs. "0,1,2,3,4,5,6,7"
_C.CUDA_VISIBLE_DEVICES = ""
# Number of workers for load the datasets.
_C.WORKERS = 4

_C.MODEL = CN()
# Device mode: "cuda" or "cpu".
_C.MODEL.DEVICE = "cuda"
# Network name and its parameters.
_C.MODEL.NETWORK = ["resnet101", []]
# The stride of the network at different stages.
_C.MODEL.NETWORK_STRIDE = [2, 2, 2, 2, 2]
# The pretrained weight for the network.
_C.MODEL.WEIGHT = ""
# Loss name and its parameters.
_C.MODEL.LOSS = ["CrossEntropyLoss", []]

_C.SOLVER = CN()
# The max training epoch.
_C.SOLVER.MAX_EPOCH = 120
# The training batch size of all GPUs.
_C.SOLVER.BATCH_SIZE = 256
# Learning rate.
_C.SOLVER.BASE_LR = 0.1
# Momentum.
_C.SOLVER.MOMENTUM = 0.9
# Weight decay.
_C.SOLVER.WEIGHT_DECAY = 0.0001
# The decay factor of the learning rate.
_C.SOLVER.GAMMA = 0.1
# The decay epoch of the learning rate.
_C.SOLVER.STEPS = [10, 20]

_C.DATASETS = CN()
# The train datasets.
_C.DATASETS.TRAIN = [
    "cifar-10_train",
]
# The test datasets.
_C.DATASETS.TEST = [
    "cifar-10_val",
]

_C.TRAIN = CN()
# The train data transforms.
_C.TRAIN.TRANSFORMS = [
    ["RandomCrop", [["size", 32], ["padding", 4],]],
    ["RandomHorizontalFlip", []],
    ["ToTensor", []],
    ["Normalize", [["mean", [0.485, 0.456, 0.406]], ["std", [0.229, 0.224, 0.225]],]],
]

_C.TEST = CN()
# The test batch size.
_C.TEST.BATCH_SIZE = 128
# The test data transforms.
_C.TEST.TRANSFORMS = [
    ["Resize", [["size", 32],]],
    ["ToTensor", []],
    ["Normalize", [["mean", [0.485, 0.456, 0.406]], ["std", [0.229, 0.224, 0.225]],]],
]
```


### Parameter-is-Module
Parameter-is-Module (PM) is a new parameter type defined in this codebase. It can help you create `Dataset`, `Transform`, `Network` and `Loss` module conveniently.

The format of PM is defined as `["ModuleName", [["arg_name0", arg0], ["arg_name1", arg1], ]]`. The argument list of the module is organized as `[["arg_name0", arg0], ["arg_name1", arg1], ]`. The `arg_name` is the module's initialization argument. If you use the default initialization arguments of the module, set the argument list to `[]`.


### Parameters in Config
#### NETWORK_STRIDE
Network may have some layers with stride more than 1, such as convolution layer or pool layer with stride equal to 2. The `NETWORK_STRIDE` is used to control the stride in the networks. For example, the origin ResNet-50 network has 5 layers with stride equals to 2, one convolution layer with kernel size of 7, one max pool layer and three convolution with kernel size of 3. If you set `NETWORK_STRIDE` to `[2, 1, 2, 2, 2]`, the stride of the max pool layer will be set to 1. Certainly, you can also set the `network_stride` by setting the `NETWORK` to `["resnet50", [["network_stride", [2, 1, 2, 2, 2]], ]]`. It is not recommended to set `network_stride` in `NETWORK`. Because network stride is so important that it should be independent from the network.

#### DATASETS
The parameter of `DATASETS` is a little different from other parameters. Different datasets may use the same `DatasetModule` to load data. So there is not a one-to-one correspondence between dataset and `DatasetModule`. You can use `DatasetAlias` to represent a specific dataset. The `DatasetAlias` will be mapped to a PM, which is defined as `["DatasetModuleName", [["arg_name0", arg0], ["arg_name1", arg1], ]]`. In this case, different `DatasetAlias` will be mapped to different PMs. These PMs may have the same `DatasetModuleName` with different argument list. Once the dataset is created, it can be well-determined by the path. In order to choose the dataset by `DatasetAlias` conveniently, you need register this dataset and set the PM in `./pytorch_classification/config/data_catalog.py`. Multiple datasets are support by using `DatasetAlias` list.

#### OUTPUT_DIR, CHECKPOINT and WEIGHT
The `OUTPUT_DIR` is the dictionary to save checkpoints and logs during training. If you train a model, it will search for `last-checkpoint.pth` from `OUTPUT_DIR`. If `last-checkpoint.pth` does not exist, it will look for `CHECKPOINT` and `WEIGHT` in proper order. If there is no `CHECKPOINT` and `WEIGHT` it will initialize the network randomly.


### GeneralDataset
`GeneralDataset` is a simple dataset type defined is this code base. It needs two arguments `img_dir` and `anno_file`. The `img_dir` is the image dictionary. The `anno_file` is a path to the annotation file. The annotation file is or organized as follow.
```
img_name0.jpg label_0
img_name1.jpg label_1
img_name2.jpg label_2
...
```


### Use Custom Dataset
You should register the dataset name and corresponding `Datasetmodule` in `./pytorch_classification/config/data_catalog.py`.
```
DATASET_CATALOG= {
    "dataset_name": ["DatasetModuleName", [[arg_name0, arg0], [arg_name1, arg1], ]],
}
```


### Create A New Dataset Module
You can create a new dataset module follow `./pytorch_classification/data/datasets/new_dataset.py`. Then, you should import this dataset module and add it to `__all__` in `./pytorch_classification/data/datasets/__init__.py`.
```python
from torch.utils.data.dataset import Dataset

class NewDataset(Dataset):
    def __init__(self, img_dir, anno_file, transforms=None):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass
```


### Create A New Data Transform Module
You can create a new data transforms module follow `./pytorch_classification/data/transforms/transforms.py`. Then, you should add this module to `__all__`.
```python
class NewTrans(object):
    def __init__(self):
        pass

    def __call__(self, image):
        pass
```


### Create A new Network Module
You can create a new network module follow `./pytorch_classification/network/new_network.py`. Then, you should import this network module and add it to `__all__` in `./pytorch_classification/network/network.py`.
```python
import torch.nn as nn


__all__ = ["NewNetwork"]


class NewNetwork(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, x):
        pass
```


### Create A New Loss Module
You can create a new loss module follow `./pytorch_classification/loss/new_loss.py`. Then, you should import your network module and add it to `__all__` in `./pytorch_classification/loss/loss.py`.
```python
__all__ = ["NewLoss"]


class NewLoss(object):
    def __init__(self):
        pass

    def __call__(self, image):
        pass
```
