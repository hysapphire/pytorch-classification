import datetime
import logging
import time

import torch
from pytorch_classification.data import make_data_loader
from pytorch_classification.engine.tester import accuracy, inference
from pytorch_classification.loss import make_criterion
from pytorch_classification.network import build_classification_model
from pytorch_classification.solver import adjust_learning_rate, make_optimizer
from pytorch_classification.utils.checkpoint import load_checkpoint_from_cfg, save_checkpoint
from pytorch_classification.utils.comm import get_rank, is_main_process, synchronize
from pytorch_classification.utils.metric_logger import AverageMeter, MetricLogger


def train(
    model, epoch, max_epoch, data_loader, optimizer, criterion, device, print_iter_period, logger, tb_writer,
):
    meters = MetricLogger()
    max_iter = len(data_loader)
    model.train()
    end = time.time()

    for iteration, (images, targets) in enumerate(data_loader):
        iteration = iteration + 1

        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        meters.update(
            size=images.size(0), loss=loss, top1=acc1, top5=acc5,
        )

        if get_rank() == 0:
            tb_writer.add_scalar('Iter Loss', meters.loss.value, epoch * max_iter + iteration)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(size=1, time=batch_time)

        eta_seconds = meters.time.avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % print_iter_period == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
    if get_rank() == 0:
        tb_writer.add_scalar('Epoch Loss', meters.loss.avg, epoch + 1)
        tb_writer.add_scalar('Train Accuracy', meters.top1.avg, epoch + 1)


def run_train(
    cfg, local_rank, distributed, tb_writer,
):
    logger = logging.getLogger("Classification.trainer")

    model = build_classification_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    criterion = make_criterion(cfg, device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False,
        )

    checkpoint = load_checkpoint_from_cfg(cfg, model, optimizer)

    start_epoch = checkpoint["epoch"] if checkpoint is not None and "epoch" in checkpoint else 0

    save_to_disk = get_rank() == 0

    max_epoch = cfg.SOLVER.MAX_EPOCH
    save_epoch_period = cfg.SAVE_EPOCH_PERIOD
    test_epoch_peroid = cfg.TEST_EPOCH_PERIOD
    print_iter_period = cfg.PRINT_ITER_PERIOD

    train_loader = make_data_loader(cfg, is_train=True, is_distributed=distributed,)

    if test_epoch_peroid > 0:
        val_loader = make_data_loader(cfg, is_train=False, is_distributed=distributed,)
    else:
        val_loader = None

    time_meter = AverageMeter("epoch_time")
    start_training_time = time.time()
    end = time.time()

    logger.info("Start training")
    for epoch in range(start_epoch, max_epoch):
        logger.info("Epoch {}".format(epoch + 1))

        adjust_learning_rate(cfg, optimizer, epoch)

        train(
            model,
            epoch,
            max_epoch,
            train_loader,
            optimizer,
            criterion,
            device,
            print_iter_period,
            logger,
            tb_writer,
        )

        if save_to_disk and ((epoch + 1) % save_epoch_period == 0 or (epoch + 1) == max_epoch):

            state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            is_final = True if (epoch + 1) == max_epoch else False
            save_checkpoint(state, cfg.OUTPUT_DIR, epoch + 1, is_final)

        if val_loader is not None and (epoch + 1) % test_epoch_peroid == 0:
            acc = inference(model, val_loader, device)

            if acc is not None:
                logger.info("Top1 accuracy: {}. Top5 accuracy: {}.".format(acc["top1"], acc["top5"]))

                tb_writer.add_scalar('Test Accuracy', acc["top1"], epoch + 1)

        epoch_time = time.time() - end
        end = time.time()

        time_meter.update(epoch_time)
        eta_seconds = time_meter.avg * (max_epoch - epoch - 1)
        epoch_string = str(datetime.timedelta(seconds=int(epoch_time)))
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        logger.info("Epoch time-consuming: {}. Eta: {}.\n".format(epoch_string, eta_string))

    synchronize()
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / epoch)".format(
            total_time_str, total_training_time / (max_epoch)
        )
    )

    return model
