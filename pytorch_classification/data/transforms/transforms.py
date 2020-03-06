from torchvision.transforms import *

__all__ = [
    "ToTensor",
    "ToPILImage",
    "Normalize",
    "Resize",
    "CenterCrop",
    "RandomCrop",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomResizedCrop",
    "ColorJitter",
    "RandomRotation",
    "RandomAffine",
    "RandomErasing",
]


# Add new data transforms here.
# class NewTrans(object):
#     def __init__(self):
#         pass

#     def __call__(self, image):
#         pass
