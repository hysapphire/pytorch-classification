from pytorch_classification.data.transforms import transforms as T


def build_transforms(cfg, is_train=False):
    transforms = cfg.TRAIN.TRANSFORMS if is_train else cfg.TEST.TRANSFORMS

    transform = T.Compose([getattr(T, t[0])(**dict(t[1])) for t in transforms])

    return transform
