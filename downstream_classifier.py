import os
import argparse
import random
import pandas as pd

import torchvision
import numpy as np
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import get_dataset_evaluation
from models import get_encoder_architecture_usage
from evaluation import create_torch_dataloader, NeuralNet, net_train, net_test, predict_feature


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate the clean or backdoored encoders')
    parser.add_argument('--dataset', default='gtsrb', type=str, help='downstream dataset')
    parser.add_argument('--reference_label', default=12, type=int, help='target class in the target downstream task')
    # parser.add_argument('--trigger_file', default='./trigger/trigger_pt_white_21_10_ap_replace.npz', type=str, help='path to the trigger file (default: none)')

    parser.add_argument('--trigger_file', default='./trigger/global_trigger_v2.npz', type=str,
                        help='path to the trigger file (default: none)')
    parser.add_argument('--encoder_usage_info', default='cifar10', type=str, help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')
    parser.add_argument('--encoder', default='./result/fed_badencoder/cifar100_gtsrb_0.2_single30.pth', type=str, help='path to the image encoder')
    parser.add_argument('--results_dir', default='./result/downstream_task', type=str, metavar='PATH', help='path to save the results (default: none)')
    parser.add_argument('--gpu', default='0', type=str, help='the index of gpu used to train the model')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--seed', default=100, type=int, help='seed')
    parser.add_argument('--nn_epochs', default=200, type=int)
    parser.add_argument('--hidden_size_1', default=512, type=int)
    parser.add_argument('--hidden_size_2', default=256, type=int)
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    ## note that the reference_file is not needed to train a downstream classifier
    parser.add_argument('--reference_file', default='./reference/cifar100/priority.npz', type=str, help='path to the reference file (default: none)')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


    assert args.reference_label >= 0, 'Enter the correct target class'


    args.data_dir = f'./data/{args.dataset}/'
    target_dataset, train_data, test_data_clean, test_data_backdoor = get_dataset_evaluation(args)  # stl10作为downstream dataset
    # target class是reference image，train_data, test_data_clean分别是stl10文件中的train.npz和test.npz，test_data_backdoor则是加了trigger的

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
                              num_workers=2,
                              pin_memory=True)
    test_loader_clean = DataLoader(test_data_clean, batch_size=args.batch_size, shuffle=False,
                                   num_workers=2,
                                   pin_memory=True)
    test_loader_backdoor = DataLoader(test_data_backdoor, batch_size=args.batch_size, shuffle=False,
                                      num_workers=2,
                                      pin_memory=True)

    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=2,
                               pin_memory=True)

    num_of_classes = len(train_data.classes)

    model = get_encoder_architecture_usage(args).cuda()
    if args.encoder != '':
        print('Loaded from: {}'.format(args.encoder))
        checkpoint = torch.load(args.encoder)
        if args.encoder_usage_info in ['CLIP', 'imagenet'] and 'clean' in args.encoder:
            model.visual.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])

    if args.encoder_usage_info in ['CLIP', 'imagenet']:
        feature_bank_training, label_bank_training = predict_feature(model.visual, train_loader)
        feature_bank_testing, label_bank_testing = predict_feature(model.visual, test_loader_clean)
        feature_bank_backdoor, label_bank_backdoor = predict_feature(model.visual, test_loader_backdoor)
        feature_bank_target, label_bank_target = predict_feature(model.visual, target_loader)
    else:
        feature_bank_training, label_bank_training = predict_feature(model.f, train_loader)
        feature_bank_testing, label_bank_testing = predict_feature(model.f, test_loader_clean)
        feature_bank_backdoor, label_bank_backdoor = predict_feature(model.f, test_loader_backdoor)
        feature_bank_target, label_bank_target = predict_feature(model.f, target_loader)

    nn_train_loader = create_torch_dataloader(feature_bank_training, label_bank_training, args.batch_size)
    nn_test_loader = create_torch_dataloader(feature_bank_testing, label_bank_testing, args.batch_size)
    nn_backdoor_loader = create_torch_dataloader(feature_bank_backdoor, label_bank_backdoor, args.batch_size)

    input_size = feature_bank_training.shape[1]

    criterion = nn.CrossEntropyLoss()

    net = NeuralNet(input_size, [args.hidden_size_1, args.hidden_size_2], num_of_classes).cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    results = {'BA': [], 'ASR': []}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    for epoch in range(1, args.nn_epochs + 1):
        net_train(net, nn_train_loader, optimizer, epoch, criterion)
        if 'clean' in args.encoder:
            net_test(net, nn_test_loader, epoch, criterion, 'Clean Accuracy (CA)')
            net_test(net, nn_backdoor_loader, epoch, criterion, 'Attack Success Rate-Baseline (ASR-B)')
        else:
            ba = net_test(net, nn_test_loader, epoch, criterion, 'Backdoored Accuracy (BA)')
            results['BA'].append(ba)
            asr = net_test(net, nn_backdoor_loader, epoch, criterion, 'Attack Success Rate (ASR)')
            results['ASR'].append(asr)

    data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
    data_frame.to_csv(args.results_dir + '/stl10_gtsrb_backdoored.csv', index_label='epoch')