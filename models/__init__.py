from .simclr_model import SimCLR


def get_encoder_architecture(args):
    if args.pretraining_dataset == 'cifar10':
        return SimCLR()
    elif args.pretraining_dataset == 'stl10':
        return SimCLR()
    else:
        raise ValueError('Unknown pretraining dataset: {}'.format(args.pretraining_dataset))


def get_encoder_architecture_usage(args):
    if args.encoder_usage_info == 'cifar10':
        return SimCLR()
    elif args.encoder_usage_info == 'stl10':
        return SimCLR()
    else:
        raise ValueError('Unknown pretraining dataset: {}'.format(args.pretraining_dataset))
