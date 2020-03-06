import logging
import os
import shutil
import sys

import torch
from pytorch_classification.utils.comm import get_world_size, is_main_process, synchronize
from torch.hub import HASH_REGEX, download_url_to_file, urlparse


def save_checkpoint(state, output_dir, epoch, is_final=False):
    filename = os.path.join(output_dir, "epoch-{}.pth".format(epoch))
    torch.save(state, filename)
    shutil.copyfile(filename, os.path.join(output_dir, "last-checkpoint.pth"))
    if is_final:
        shutil.copyfile(filename, os.path.join(output_dir, "final-model.pth"))


def load_checkpoint_from_cfg(cfg, model, optimizer=None):
    distributed = get_world_size() > 1
    logger = logging.getLogger("Classification.trainer")

    if cfg.CHECKPOINT:
        checkpoint_path = cfg.CHECKPOINT
        log_info = "Loading checkpoint {}".format(checkpoint_path)
    elif has_checkpoint(cfg.OUTPUT_DIR):
        checkpoint_path = os.path.join(cfg.OUTPUT_DIR, "last-checkpoint.pth")
        log_info = "Loading the last checkpoint {}".format(checkpoint_path)
    elif cfg.MODEL.WEIGHT:
        checkpoint_path = cfg.MODEL.WEIGHT
        if checkpoint_path.startswith("http"):
            checkpoint_path = download_from_url(checkpoint_path)
        log_info = "Loading the pretrained weight {}".format(checkpoint_path)
    else:
        log_info = "Have no initialized weights."
        logger.info(log_info)
        return None

    logger.info(log_info)
    checkpoint = load_checkpoint(checkpoint_path, distributed)

    model.load_state_dict(checkpoint.pop("state_dict"))
    if "optimizer" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint.pop("optimizer"))

    return checkpoint


def load_checkpoint(checkpoint_path, distributed=False):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    if "state_dict" not in checkpoint:
        checkpoint = dict(state_dict=checkpoint)

    save_distributed = list(checkpoint["state_dict"].keys())[0].startswith("module")

    if save_distributed == distributed:
        return checkpoint

    if distributed:
        new_state_dict = dict()
        for k, v in checkpoint["state_dict"].items():
            name = "module." + k
            new_state_dict[name] = v
        checkpoint["state_dict"] = new_state_dict
    else:
        new_state_dict = dict()
        for k, v in checkpoint["state_dict"].items():
            name = k[7:]
            new_state_dict[name] = v
        checkpoint["state_dict"] = new_state_dict

    return checkpoint


def has_checkpoint(output_dir):
    save_file = os.path.join(output_dir, "last-checkpoint.pth")
    return os.path.exists(save_file)


def download_from_url(url):
    torch_home = os.path.expanduser(os.getenv("TORCH_HOME", "~/.cache"))
    model_dir = os.getenv("TORCH_MODEL_ZOO", os.path.join(torch_home, "torch", "checkpoints"))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    parts = urlparse(url)
    filename = os.path.basename(parts.path)

    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file) and is_main_process():
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = HASH_REGEX.search(filename)
        if hash_prefix is not None:
            hash_prefix = hash_prefix.group(1)
            if len(hash_prefix) < 6:
                hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)
    synchronize()
    return cached_file
