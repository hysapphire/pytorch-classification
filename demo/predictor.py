import cv2

import torch
from pytorch_classification.data.transforms import transforms as T
from pytorch_classification.network import build_classification_model
from pytorch_classification.utils.checkpoint import load_checkpoint_from_cfg

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class ClsDemo(object):
    def __init__(self, cfg):
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model = build_classification_model(cfg).to(self.device)
        load_checkpoint_from_cfg(cfg, self.model)
        self.model.eval()

    def run_on_openv_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        image = transform(image)[None].to(self.device)

        with torch.no_grad():
            logits = self.model(image)
            pred = torch.nn.Softmax(dim=1)(logits)

        score, index = pred.topk(1, dim=1)
        index = index.squeeze().item()
        score = score.squeeze().item()

        return {"class": CLASSES[index], "score": score}
