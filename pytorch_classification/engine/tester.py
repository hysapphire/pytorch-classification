import logging
import os

from tqdm import tqdm

import torch
from pytorch_classification.data import make_data_loader
from pytorch_classification.network import build_classification_model
from pytorch_classification.utils.checkpoint import load_checkpoint
from pytorch_classification.utils.comm import (
    all_gather_tensor,
    get_rank,
    get_world_size,
    is_main_process,
    synchronize,
)
from pytorch_classification.utils.metric_logger import AverageMeter


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def _accumulate_values_from_multiple_gpus(values_per_gpu):
    all_values = all_gather_tensor(values_per_gpu)
    if not is_main_process():
        return None
    all_values = torch.cat(all_values, dim=0)

    return all_values


def inference(
    model, data_loader, device,
):
    model.to(device)
    model.eval()

    logger = logging.getLogger("Classification.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on validation dataset({} images).".format(len(dataset)))

    top1 = AverageMeter(name="top1")
    top5 = AverageMeter(name="top5")
    with torch.no_grad():
        for _, (images, targets) in enumerate(tqdm(data_loader, disable=not is_main_process())):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            synchronize()
            outputs = _accumulate_values_from_multiple_gpus(outputs)
            targets = _accumulate_values_from_multiple_gpus(targets)
            synchronize()
            if not is_main_process():
                continue

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(value=acc1, size=images.size(0))
            top5.update(value=acc5, size=images.size(0))

    if is_main_process():
        return dict(top1=top1.avg, top5=top5.avg)
    else:
        return None


def run_test(cfg, local_rank, distributed, model=None):
    logger = logging.getLogger("Classification.tester")

    device = torch.device(cfg.MODEL.DEVICE)
    if model is None:
        model = build_classification_model(cfg)
        model.to(device)

        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False,
            )

        checkpoint_path = (
            cfg.CHECKPOINT if cfg.CHECKPOINT else os.path.join(cfg.OUTPUT_DIR, "final-model.pth")
        )
        checkpoint = load_checkpoint(checkpoint_path, distributed)

        model.load_state_dict(checkpoint["state_dict"])

    val_loader = make_data_loader(cfg, is_train=False, is_distributed=distributed,)

    acc = inference(model, val_loader, device)

    return acc
