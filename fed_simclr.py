import os
import copy
import numpy as np
from tqdm import tqdm
import argparse
import torch
from PIL import Image
from torch.utils.data import DataLoader
import json
import math
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import random


from models import get_encoder_architecture
from datasets import get_pretraining_dataset, get_usergroup, get_testing_dataset
from evaluation import knn_predict
from aggregator import aggregating


def local_train(net, data_loader, train_optimizer, epoch, args):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for im_1, im_2 in train_bar:
        im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)
        feature_1, out_1 = net(im_1)
        feature_2, out_2 = net(im_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.knn_t)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * args.batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * args.batch_size, -1)
        # compute loss （cosine similarity)
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.knn_t)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Local Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.local_epoch, train_optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num, net.state_dict()


def test(net, memory_data_loader, test_data_clean_loader, epoch, args):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_num, feature_bank = 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_clean_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fedrated pretraining encoder')
    parser.add_argument('--aggregator', default='fedavg', type=str, help='aggregating algorithm used by the server')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--pretraining_dataset', type=str, default='cifar10')
    parser.add_argument('--results_dir', default='./result/pretrained_encoders', type=str, metavar='PATH',
                        help='path to save the results (default: none)')

    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu the code runs on')
    parser.add_argument('--knn-t', default=0.5, type=float, help='softmax temperature in kNN monitor')
    parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')

    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--epochs', type=int, default=200, help="number of rounds of global training")
    parser.add_argument('--frac', type=float, default=0.2, help='the fraction of clients selected for training in each round: C')
    parser.add_argument('--local_epoch', type=int, default=2, help="the number of local epochs: E")
    parser.add_argument('--iid', type=int, default=0, help='Default set to Non-IID. Set to 1 for IID.')
    CUDA_LAUNCH_BLOCKING = 1
    args = parser.parse_args()

    # Set the random seeds and GPU information
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    args.data_dir = f'./data/{args.pretraining_dataset}/'

    # load user groups(dataset of each group)
    user_groups = get_usergroup(args)

    memory_data, test_data_clean = get_testing_dataset(args)

    memory_loader = DataLoader(
        memory_data,
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=2,
        pin_memory=True
    )

    test_loader_clean = DataLoader(
        test_data_clean,
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=2,
        pin_memory=True
    )

    # initialize the global model
    global_model = get_encoder_architecture(args).cuda()

    # logging
    results = {'train_loss_avg_of_all_clients': [], 'test_acc@1_of_global_model': []}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    # Dump args
    with open(args.results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    train_loss, train_accuracy = [], []

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print("==========================================================")
        local_weights, local_losses = [], []
        print(f'\n|Global Training Round : {epoch}|\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            train_data = get_pretraining_dataset(args, user_groups=user_groups, idx=idx)

            local_model = copy.deepcopy(global_model)
            # Define the optimizer
            optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=1e-6)

            train_loader = DataLoader(
                train_data,
                batch_size=args.batch_size,
                shuffle=True,
                # num_workers=2,
                pin_memory=True,
                drop_last=True
            )

            print(f'--------Training process of client{idx}--------')
            for e in range(1, args.local_epoch + 1):
                loss, weights = local_train(local_model, train_loader, optimizer, e, args)
                local_losses.append(copy.deepcopy(loss))
            local_weights.append(copy.deepcopy(weights))

        # update global weights
        global_weights = aggregating(args, local_weights)
        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        results['train_loss_avg_of_all_clients'].append(loss_avg)
        train_loss.append(loss_avg)
        print(f'Average loss of all clients:{loss_avg}')

        # knn to monitor the global model
        test_acc_1 = test(global_model.f, memory_loader, test_loader_clean, epoch, args)
        results['test_acc@1_of_global_model'].append(test_acc_1)
        # Save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(args.results_dir + '/simclr_cifar10.csv', index_label='epoch')

        if epoch % args.epochs == 0:
            torch.save({'epoch': epoch, 'state_dict': global_model.state_dict(), 'optimizer': optimizer.state_dict()}, args.results_dir + '/cifar10_simclr' + str(epoch) + '.pth')
