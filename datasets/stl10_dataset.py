from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn

from datasets.backdoored_dataset import CIFAR10Pair, CIFAR10Mem, BackdooredDataset, TestBackdoor, InspectionData, CIFAR10Pair_fltrust, ReferenceImg

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

finetune_transform_cifar10 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_stl10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

backdoor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def get_pretraining_stl10_fedsimclr(data_dir, user_groups, idx):
    train_data = CIFAR10Pair(numpy_file=data_dir + "train_unlabeled.npz", class_type=classes, user_groups=user_groups, idx=idx, transform=train_transform)

    return train_data

def get_memory_data_for_testing(data_dir):
    memory_data = CIFAR10Mem(numpy_file=data_dir + "train.npz", class_type=classes, transform=test_transform_cifar10)
    test_data = CIFAR10Mem(numpy_file=data_dir + "test.npz", class_type=classes, transform=test_transform_cifar10)
    return memory_data, test_data

def get_shadow_stl10(args, attacker, usergroups):

    index = usergroups[attacker]
    index = np.asarray(index, dtype=int)

    print('loading from the training data')

    shadow_dataset = BackdooredDataset(
        numpy_file=args.data_dir + 'train_unlabeled.npz',
        trigger_file=args.global_trigger,
        reference_file=args.reference_file,
        class_type=classes,
        indices=index,
        transform=train_transform,
        bd_transform=test_transform_cifar10,
        ftt_transform=finetune_transform_cifar10
    )

    memory_data = CIFAR10Mem(numpy_file=args.data_dir+'train.npz', class_type=classes, transform=test_transform_cifar10)
    test_data_backdoor = TestBackdoor(numpy_file=args.data_dir+'test.npz', trigger_file=args.global_trigger, reference_label=args.reference_label,  transform=test_transform_cifar10)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+'test.npz', class_type=classes, transform=test_transform_cifar10)

    return shadow_dataset, memory_data, test_data_clean, test_data_backdoor


def get_shadow_stl10_each_malicious(args, attacker, attacker_count, usergroups):

    index = usergroups[attacker]
    index = np.asarray(index, dtype=int)

    print('loading from the training data')

    if attacker_count == 1:
        trigger_file = args.local_trigger1
    elif attacker_count == 2:
        trigger_file = args.local_trigger2
    elif attacker_count == 3:
        trigger_file = args.local_trigger3
    elif attacker_count == 4:
        trigger_file = args.local_trigger4

    shadow_dataset = BackdooredDataset(
        numpy_file=args.data_dir + 'train_unlabeled.npz',
        trigger_file=trigger_file,
        reference_file=args.reference_file,
        class_type=classes,
        indices=index,
        transform=train_transform,
        bd_transform=test_transform_cifar10,
        ftt_transform=finetune_transform_cifar10
    )

    return shadow_dataset

def get_server_data_stl10(data_dir):
    server_data = CIFAR10Pair_fltrust(numpy_file=data_dir, class_type=classes, transform=train_transform)
    return server_data

def get_inspection_data_stl10(data_dir):
    detect_data = InspectionData(numpy_file=data_dir, transform=test_transform_cifar10)
    return detect_data

def get_downstream_stl10(args):
    training_file_name = 'train.npz'
    testing_file_name = 'test.npz'

    if args.encoder_usage_info == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
    elif args.encoder_usage_info == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10

    else:
        raise NotImplementedError

    target_dataset = ReferenceImg(reference_file=args.reference_file, transform=test_transform)
    memory_data = CIFAR10Mem(numpy_file=args.data_dir+training_file_name, class_type=classes, transform=test_transform)
    test_data_backdoor = TestBackdoor(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+testing_file_name, class_type=classes, transform=test_transform)

    return target_dataset, memory_data, test_data_clean, test_data_backdoor