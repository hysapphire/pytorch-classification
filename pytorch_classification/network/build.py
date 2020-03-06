from pytorch_classification.network import network as N


def build_classification_model(cfg):
    network_module = getattr(N, cfg.MODEL.NETWORK[0])

    args = dict(num_classes=cfg.NUM_CLASS, network_stride=cfg.MODEL.NETWORK_STRIDE)
    extra_args = dict(cfg.MODEL.NETWORK[1])

    args.update(extra_args)

    return network_module(**args)
