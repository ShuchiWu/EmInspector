import numpy as np

from .cifar10_dataset import get_pretraining_cifar10_fedsimclr, get_memory_data_for_testing, get_shadow_cifar10, get_shadow_cifar10_each_malicious, get_inspection_data_cifar10, get_server_data_cifar10, get_downstream_cifar10
from .stl10_dataset import get_pretraining_stl10_fedsimclr, get_memory_data_for_testing, get_shadow_stl10, get_shadow_stl10_each_malicious, get_inspection_data_stl10, get_server_data_stl10, get_downstream_stl10
from .cifar100_dataset import get_pretraining_cifar100_fedsimclr, get_memory_data_for_testing, get_shadow_cifar100, get_shadow_cifar100_each_malicious, get_inspection_data_cifar100, get_server_data_cifar100, get_downstream_cifar100
from .sampling import cifar_iid, cifar_noniid


def get_usergroup(args):
    if args.pretraining_dataset == 'stl10':
        train_data = np.load(f'./data/{args.pretraining_dataset}/train_unlabeled.npz')
    else:
        train_data = np.load(f'./data/{args.pretraining_dataset}/train.npz')
    if args.iid:
        user_groups = cifar_iid(train_data['x'], args.num_users)
    else:
        user_groups = cifar_noniid(train_data, args.num_users)

    return user_groups


def get_pretraining_dataset(args, user_groups, idx):
    if args.pretraining_dataset == 'cifar10':
        return get_pretraining_cifar10_fedsimclr(args.data_dir, user_groups, idx)
    elif args.pretraining_dataset == 'stl10':
        return get_pretraining_stl10_fedsimclr(args.data_dir, user_groups, idx)
    elif args.pretraining_dataset == 'cifar100':
        return get_pretraining_cifar100_fedsimclr(args.data_dir, user_groups, idx)


def get_testing_dataset(args):
    return get_memory_data_for_testing(args.data_dir)


def get_backdoored_dataset(args, attackers, user_groups):
    if args.shadow_dataset == 'cifar10':
        return get_shadow_cifar10(args, attackers, user_groups)
    elif args.shadow_dataset == 'stl10':
        return get_shadow_stl10(args, attackers, user_groups)
    elif args.shadow_dataset == 'cifar100':
        return get_shadow_cifar100(args, attackers, user_groups)
    else:
        raise NotImplementedError


def get_backdoored_dataset_each_malicious_client(args, attackers, attacker_count, user_groups):
    if args.shadow_dataset == 'cifar10':
        return get_shadow_cifar10_each_malicious(args, attackers, attacker_count, user_groups)
    elif args.shadow_dataset == 'stl10':
        return get_shadow_stl10_each_malicious(args, attackers, attacker_count, user_groups)
    elif args.shadow_dataset == 'cifar100':
        return get_shadow_cifar100_each_malicious(args, attackers, attacker_count, user_groups)
    else:
        raise NotImplementedError


def get_inspection_data(args):
    return get_inspection_data_cifar10(args.inspection_data_dir)


def get_server_data(args):
    return get_server_data_cifar10(args.inspection_data_dir)


def get_dataset_evaluation(args):
    if args.dataset =='cifar10':
        return get_downstream_cifar10(args)
    elif args.dataset == 'cifar100':
        return get_downstream_cifar100(args)
    elif args.dataset == 'stl10':
        return get_downstream_stl10(args)
    elif args.dataset == 'gtsrb':
        return get_downstream_gtsrb(args)
    else:
        raise NotImplementedError