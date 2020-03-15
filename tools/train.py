import sys

sys.path.append(".")

import argparse
import os

import torch
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.tensorboard import SummaryWriter
from pytorch_classification.config import cfg
from pytorch_classification.engine.trainer import run_train
from pytorch_classification.engine.tester import run_test
from pytorch_classification.utils.miscellaneous import (
    mkdir,
    save_config,
    save_dict_data,
    print_dict_data,
)
from pytorch_classification.utils.logger import setup_logger
from pytorch_classification.utils.comm import get_rank, synchronize


def main():
    parser = argparse.ArgumentParser(description="PyTorch Classification Training.")
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file", type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test", dest="skip_test", help="Do not test the final model", action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if cfg.MODEL.DEVICE == "cuda" and cfg.CUDA_VISIBLE_DEVICES is not "":
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    output_dir = cfg.OUTPUT_DIR
    tb_dir = os.path.join(output_dir, 'tb_log')
    mkdir(tb_dir)
    
    tb_writer = SummaryWriter(tb_dir)

    logger = setup_logger("Classification", output_dir, get_rank())

    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + get_pretty_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)

    model = run_train(cfg, args.local_rank, distributed, tb_writer)

    if not args.skip_test:
        acc = run_test(cfg, args.local_rank, distributed, model)
        save_dict_data(acc, os.path.join(cfg.OUTPUT_DIR, "acc.txt"))
        print_dict_data(acc)


if __name__ == "__main__":
    main()
