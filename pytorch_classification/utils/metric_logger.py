from collections import defaultdict

import torch


class AverageValue(object):
    def __init__(self):
        self.value = None
        self.count = 0
        self.sum = 0.0

    def update(self, value, size=1):
        self.value = value
        self.count += size
        self.sum += value * size

    def reset(self):
        self.value = None
        self.count = 0
        self.sum = 0.0

    @property
    def avg(self):
        return self.sum / self.count


class AverageMeter(object):
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.avg_val = AverageValue()
        self.reset()

    def reset(self):
        self.avg_val.reset()

    def update(self, value, size=1):
        if isinstance(value, torch.Tensor):
            value = value.item()
        assert isinstance(value, (float, int))

        self.avg_val.update(value, size)

    @property
    def value(self):
        return self.avg_val.value

    @property
    def avg(self):
        return self.avg_val.avg

    def __str__(self):
        fmtstr = "{} {" + self.fmt + "} ({" + self.fmt + "})"
        fmtstr = fmtstr.format(self.name, self.avg_val.value, self.avg_val.avg)
        return fmtstr


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(AverageValue)
        self.delimiter = delimiter

    def update(self, size=1, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, size)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        key_list = self.meters.keys()
        for name in sorted(key_list):
            meter = self.meters[name]
            loss_str.append("{}: {:.4f} ({:.4f})".format(name, meter.value, meter.avg))
        return self.delimiter.join(loss_str)
