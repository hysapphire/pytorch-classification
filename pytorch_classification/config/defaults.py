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
