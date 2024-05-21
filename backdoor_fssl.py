import os
import copy
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from models import get_encoder_architecture, get_encoder_architecture_usage
from datasets import get_usergroup, get_pretraining_dataset, get_backdoored_dataset, get_pretraining_dataset, get_testing_dataset, get_backdoored_dataset_each_malicious_client, get_server_data
from evaluation import test
from aggregator import aggregating
from aggregator.aggregating_algorithm import get_grad, fltrust


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

def server_train(net, data_loader, train_optimizer, epoch, args):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for im_1, im_2 in train_bar:
        im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)
        feature_1, out_1 = net(im_1)
        feature_2, out_2 = net(im_2)
        out = torch.cat([out_1, out_2], dim=0)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.knn_t)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * data_loader.batch_size, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * data_loader.batch_size, -1)
        # compyute loss （cosine similarity)
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.knn_t)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Local Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.local_epoch, train_optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num, net.state_dict()


def backdoor_train(backdoored_encoder, clean_encoder, data_loader, train_optimizer, poison_ep, args):
    backdoored_encoder.train()
    # freeze the BN layer
    for module in backdoored_encoder.modules():   # Returns an iterator over all modules in the network.
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):    # Return whether the object has an attribute with the given name.
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    clean_encoder.eval()

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_loss_0, total_loss_1, total_loss_2 = 0.0, 0.0, 0.0

    for img_clean, img_backdoor_list, reference_list, reference_aug_list in train_bar:

        img_clean = img_clean.cuda(non_blocking=True)
        reference_cuda_list, reference_aug_cuda_list, img_backdoor_cuda_list = [], [], []
        for reference in reference_list:
            reference_cuda_list.append(reference.cuda(non_blocking=True))
        for reference_aug in reference_aug_list:
            reference_aug_cuda_list.append(reference_aug.cuda(non_blocking=True))
        for img_backdoor in img_backdoor_list:
            img_backdoor_cuda_list.append(img_backdoor.cuda(non_blocking=True))

        clean_feature_reference_list = []

        with torch.no_grad():
            clean_feature_raw = clean_encoder(img_clean)
            clean_feature_raw = F.normalize(clean_feature_raw, dim=-1)
            for img_reference in reference_cuda_list:
                clean_feature_reference = clean_encoder(img_reference)
                clean_feature_reference = F.normalize(clean_feature_reference, dim=-1)
                clean_feature_reference_list.append(clean_feature_reference)

        feature_raw = backdoored_encoder(img_clean)
        feature_raw = F.normalize(feature_raw, dim=-1)

        feature_backdoor_list = []
        for img_backdoor in img_backdoor_cuda_list:
            feature_backdoor = backdoored_encoder(img_backdoor)
            feature_backdoor = F.normalize(feature_backdoor, dim=-1)
            feature_backdoor_list.append(feature_backdoor)

        feature_reference_list = []
        for img_reference in reference_cuda_list:
            feature_reference = backdoored_encoder(img_reference)
            feature_reference = F.normalize(feature_reference, dim=-1)
            feature_reference_list.append(feature_reference)

        feature_reference_aug_list = []
        for img_reference_aug in reference_aug_cuda_list:
            feature_reference_aug = backdoored_encoder(img_reference_aug)
            feature_reference_aug = F.normalize(feature_reference_aug, dim=-1)
            feature_reference_aug_list.append(feature_reference_aug)

        loss_0_list, loss_1_list = [], []
        for i in range(len(feature_reference_list)):
            loss_0_list.append(- torch.sum(feature_backdoor_list[i] * feature_reference_list[i], dim=-1).mean())  # 余弦相似度
            loss_1_list.append(- torch.sum(feature_reference_aug_list[i] * clean_feature_reference_list[i], dim=-1).mean())
        loss_2 = - torch.sum(feature_raw * clean_feature_raw, dim=-1).mean()

        loss_0 = sum(loss_0_list)/len(loss_0_list)
        loss_1 = sum(loss_1_list)/len(loss_1_list)

        loss = loss_0 + args.lambda1 * loss_1 + args.lambda2 * loss_2

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        total_loss_0 += loss_0.item() * data_loader.batch_size
        total_loss_1 += loss_1.item() * data_loader.batch_size
        total_loss_2 += loss_2.item() * data_loader.batch_size
        train_bar.set_description('Poisoning Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.6f}, Loss0: {:.6f}, Loss1: {:.6f},  Loss2: {:.6f}'.format(poison_ep, args.poison_epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num,  total_loss_0 / total_num , total_loss_1 / total_num,  total_loss_2 / total_num))

    return total_loss / total_num, backdoored_encoder.f.state_dict()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Poisoning the local client to get a backdoor encoder')
    parser.add_argument('--aggregator', default='flare', type=str, help='aggregating algorithm used by the server')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr1', default=0.05, type=float, help='initial learning rate')
    parser.add_argument('--pretraining_dataset', type=str, default='stl10')
    parser.add_argument('--inspection_data_dir', type=str, default='./data/stl10/inspection_data.npz')

    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--lambda1', default=1.0, type=np.float64, help='value of labmda1')
    parser.add_argument('--lambda2', default=1.0, type=np.float64, help='value of labmda2')
    parser.add_argument('--poison_epochs', type=int, default=9, help='the number of poisoning epochs')
    parser.add_argument('--local_epoch', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--epochs', type=int, default=30, help="number of rounds of global training")
    parser.add_argument('--num_users', type=int, default=25, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.32, help='the fraction of clients: C')
    parser.add_argument('--num_attackers', type=int, default=2, help="number of attackers")
    parser.add_argument('--iid', type=int, default=0, help='Default set to Non-IID. Set to 1 for IID.')
    parser.add_argument('--reference_file', default='./reference/stl10/airplane.npz', type=str, help='path to the reference inputs')
    parser.add_argument('--reference_label', default=0, type=int, help='the target label of the malicious')
    parser.add_argument('--shadow_dataset', default='stl10', type=str, help='backdoored dataset')
    parser.add_argument('--encoder_usage_info', default='stl10', type=str, help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')
    parser.add_argument('--pretrained_encoder', default='./result/pretrained_encoders/fedsimclr_stl10_encoder200.pth', type=str,
                        help='path to the clean encoder used to finetune the backdoored encoder')

    parser.add_argument('--results_dir', default='./result/backdoored_models', type=str, metavar='PATH',
                        help='path to save the backdoored encoder')

    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu the code runs on')

    parser.add_argument('--knn-t', default=0.5, type=float, help='softmax temperature in kNN monitor')
    parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')

    parser.add_argument('--global_trigger', default='./trigger/global_trigger.npz', type=str, help='path to the global trigger')
    parser.add_argument('--local_trigger1', default='./trigger/global_trigger.npz', type=str, help='path to the local trigger1')
    parser.add_argument('--local_trigger2', default='./trigger/global_trigger.npz', type=str, help='path to the local trigger2')
    parser.add_argument('--local_trigger3', default='./trigger/global_trigger.npz', type=str, help='path to the local trigger3')
    parser.add_argument('--local_trigger4', default='./trigger/global_trigger.npz', type=str, help='path to the local trigger4')

    parser.add_argument('--wrong_mal', type=int, default=0)
    parser.add_argument('--right_ben', type=int, default=0)
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--noise', type=float, default=0.001)

    args = parser.parse_args()

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
    print(args)

    user_groups = get_usergroup(args)

    train_loss, train_accuracy = [], []

    attacker_list = list(set(np.random.choice(args.num_users, args.num_attackers, replace=False)))
    print('attacker list:', attacker_list)

    clean_model = get_encoder_architecture_usage(args).cuda()
    model = get_encoder_architecture_usage(args).cuda()

    if args.pretrained_encoder != '':
        print(f'load the clean model from {args.pretrained_encoder}')

        checkpoint = torch.load(args.pretrained_encoder)
        clean_model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(checkpoint['state_dict'])

    global_model = copy.deepcopy(clean_model)
    detect_model = copy.deepcopy(clean_model)

    results = {'BA': [], 'ASR_TEST': []}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    for epoch in range(1, args.epochs + 1):
        print("==========================================================")
        local_weights, local_losses = [], []
        print(f'\n|Global Training Round : {epoch}|\n')

        m = max(int(args.frac * args.num_users), 0)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        global_model.train()

        for idx in idxs_users:
            train_data = get_pretraining_dataset(args, user_groups=user_groups, idx=idx)

            local_model = copy.deepcopy(global_model)
            local_training_optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=1e-6)

            train_loader = DataLoader(
                train_data,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                drop_last=True
            )
            print(f'--------Training process of client{idx}--------')
            for e in range(1, args.local_epoch + 1):
                loss, weights = local_train(local_model, train_loader, local_training_optimizer, e, args)

                local_losses.append(copy.deepcopy(loss))
            local_weights.append(copy.deepcopy(weights))

        attacker_count = 1
        if epoch % 1 == 0:
            for attacker in attacker_list:

                model = copy.deepcopy(global_model)

                # Define the optimizer
                print("Fine-tune Optimizer: SGD")
                finetune_optimizer = torch.optim.SGD(model.f.parameters(), lr=args.lr1, weight_decay=5e-4, momentum=0.9)
                print(f'-!-!-!-!- Attacker{attacker}  Begin to poison the encoder -!-!-!-!-')

                shadow_data = get_backdoored_dataset_each_malicious_client(args, attacker, attacker_count, user_groups)

                attacker_count += 1

                backdoor_train_loader = DataLoader(
                    shadow_data,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=8,
                    pin_memory=True,
                    drop_last=True
                )

                # run normal training for 1 local epoch first
                normal_train_data = get_pretraining_dataset(args, user_groups, attacker)
                normal_loader = DataLoader(
                    normal_train_data,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=8,
                    pin_memory=True,
                    drop_last=True
                )
                for i in range(1):
                    _, model_para = local_train(model, normal_loader, local_training_optimizer, i + 1, args)
                model.load_state_dict(model_para)
                for e in range(1, args.poison_epochs + 1):
                    backdoored_loss, backdoored_weights = backdoor_train(model.f, clean_model.f, backdoor_train_loader, finetune_optimizer, e, args)

                model.load_state_dict(backdoored_weights, strict=False)
                local_weights.append(copy.deepcopy(model.state_dict()))


        # aggregating local models
        if args.aggregator != 'fltrust':
            global_weight = aggregating(args, local_weights, detect_model, global_model)
        else:
            def get_grad(update, model):
                '''get the update weight'''
                grad = {}
                for key, var in update.items():
                    grad[key] = update[key] - model[key]
                return grad


            global_para = global_model.state_dict()

            grad_list = []
            for local_update in local_weights:
                grad_list.append(get_grad(local_update, global_para))

            server_model = copy.deepcopy(global_model)
            server_data = get_server_data(args)

            server_loader = DataLoader(
                server_data,
                batch_size=32,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                drop_last=True
            )

            print('=' * 15, 'FLTrust', '=' * 15)

            for ep in range(3):
                _, fltrust_norm = server_train(server_model, server_loader, local_training_optimizer, ep+1, args)

            fltrust_norm = get_grad(fltrust_norm, global_para)
            global_weight = fltrust(grad_list, fltrust_norm, global_para)


        global_model.load_state_dict(global_weight)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        print(f'Average loss of all clients:{loss_avg}')
        print('-----Global model clean & backdoored test-----:')

        shadow_data, memory_data, test_data_clean, test_data_backdoor = get_backdoored_dataset(args, 0, user_groups)
        test_loader_backdoor = DataLoader(
            test_data_backdoor,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        memory_loader = DataLoader(
            memory_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )

        test_loader_clean = DataLoader(
            test_data_clean,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )

        print('================global model test=================')
        test_global_ba_acc, ba, asr_test = test(global_model.f, memory_loader, test_loader_clean, test_loader_backdoor, epoch, args)

        results['BA'].append(ba)
        results['ASR_TEST'].append(asr_test)
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(args.results_dir + '/backdoor_fssl_log.csv', index_label='epoch')

        if epoch % args.epochs == 0:
            torch.save({'epoch': epoch, 'state_dict': global_model.state_dict(), 'localtraining_optimizer': local_training_optimizer.state_dict(), 'finetune_optimizer': finetune_optimizer.state_dict()},
                args.results_dir + '/backdoored_model' + str(epoch) + '.pth')
