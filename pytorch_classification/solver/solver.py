from bisect import bisect_right

import torch


def adjust_learning_rate(cfg, optimizer, epoch):
    lr = cfg.SOLVER.BASE_LR * (cfg.SOLVER.GAMMA ** bisect_right(cfg.SOLVER.STEPS, epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def make_optimizer(cfg, model):
    optimizer = torch.optim.SGD(
        model.parameters(),
        cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    return optimizer
