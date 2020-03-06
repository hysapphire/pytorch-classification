import sys

sys.path.append("../")

import cv2

from predictor import ClsDemo
from pytorch_classification.config import cfg


config = "/path_to_config"
img_path = "/path_to_image"
checkpoint_path = "/path_to_pre-trained_model"

cfg.merge_from_file(config)
cfg.merge_from_list(["CHECKPOINT", checkpoint_path])

cls_demo = ClsDemo(cfg)

image = cv2.imread(img_path)
pred = cls_demo.run_on_openv_image(image)
print(pred)