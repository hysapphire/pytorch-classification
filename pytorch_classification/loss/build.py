from pytorch_classification.loss import loss as L


def make_criterion(cfg, device):
    loss_module = getattr(L, cfg.MODEL.LOSS[0])

    args = dict(cfg.MODEL.LOSS[1])

    return loss_module(**args)
