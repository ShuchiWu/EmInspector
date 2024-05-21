import math
import torch
import torch.nn as nn
import copy
import random
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import hdbscan
import warnings
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sklearn.metrics.pairwise as smp
from datasets import get_inspection_data
import logging


def get_grad(update, model):
    grad = {}
    for key, var in update.items():
        grad[key] = update[key] - model[key]
    return grad


def get_2_norm(params_a, params_b):
    sum = 0
    if isinstance(params_a,dict):
        for i in params_a.keys():
            if i.split('.')[-1] != 'num_batches_tracked':
                if len(params_a[i]) == 1:
                    sum += pow(np.linalg.norm(params_a[i].cpu().numpy()-\
                        params_b[i].cpu().numpy(), ord=2),2)
                else:
                    a = copy.deepcopy(params_a[i].cpu().numpy())
                    b = copy.deepcopy(params_b[i].cpu().numpy())
                    for j in range(len(a)):
                        x = copy.deepcopy(a[j].flatten())
                        y = copy.deepcopy(b[j].flatten())
                        sum += pow(np.linalg.norm(x-y, ord=2),2)
    else:
        sum += pow(np.linalg.norm(params_a-params_b, ord=2),2)
    norm = np.sqrt(sum)
    return norm


def value_replace(w, value_sequence):  # w模型形式 ,value_sequence 数组形式，
    w_rel = copy.deepcopy(w)
    m =0
    print('-------Value Replacement------')
    for i in tqdm(w.keys()):
        for index, element in np.ndenumerate(w[i].cpu().numpy()): #顺序获取每一个值
            w_rel[i][index] = torch.tensor(value_sequence[m])
            m =m +1
    return w_rel


def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)


def parameters_dict_to_vector(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)


def fedavg(weights):
    """
    Returns the average of the weights.
    """
    print('=' * 15, 'FedAVG', '=' * 15)
    w_avg = copy.deepcopy(weights[0])
    for key in w_avg.keys():
        for i in range(1, len(weights)):
            w_avg[key] = w_avg[key] + weights[i][key]
        w_avg[key] = torch.div(w_avg[key], len(weights))
    return w_avg


def krum(w, c):
    print('=' * 15, 'Krum', '=' * 15)
    euclid_dist_list = []
    euclid_dist_matrix = [[0 for i in range(len(w))] for j in range(len(w))]
    for i in tqdm(range(len(w))):
        for j in range(i, len(w)):
            euclid_dist_matrix[i][j] = get_2_norm(w[i],w[j])
            euclid_dist_matrix[j][i] = euclid_dist_matrix[i][j]
        euclid_dist = euclid_dist_matrix[i][:]
        euclid_dist.sort()
        if len(w) >= (len(w)-c-2):
            euclid_dist_list.append(sum(euclid_dist[:len(w)-c-2]))
        else:
            euclid_dist_list.append(sum(euclid_dist))

    s_w = euclid_dist_list.index(min(euclid_dist_list))
    print('choosed index =',s_w)
    w_avg = w[s_w]
    return w_avg


def trimmed_mean(beta, w):
    if isinstance(w[0], dict):
        w_list = []
        print('=' * 15, 'Trimmmed-mean', '=' * 15)
        for i in tqdm(range(len(w))):
            values_w = []
            for k in w[i].keys():
                values_w += list(w[i][k].view(-1).cpu().numpy())
            w_list.append(values_w)
        w_array = np.transpose(np.array(w_list))
        w_array.sort()

        w_list_sum = sum(np.array(w_list))
        w_min_sum = sum(np.transpose(w_array[: ,:beta]))  # 转置
        w_max_sum = sum(np.transpose(w_array[: ,-beta:]))

        w_avg_value = (w_list_sum - w_min_sum -w_max_sum ) /(len(w ) - 2 *beta)
        print(w_avg_value)
        w_avg = value_replace(w[0], w_avg_value)
    else:
        print('\nNot dict')

    return w_avg


def fltrust(params, central_param, global_parameters):

    FLTrustTotalScore = 0
    score_list = []
    central_param_v = parameters_dict_to_vector_flt(central_param)
    central_norm = torch.norm(central_param_v)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    sum_parameters = None
    for local_parameters in params:
        local_parameters_v = parameters_dict_to_vector_flt(local_parameters)
        client_cos = cos(central_param_v, local_parameters_v)
        client_cos = max(client_cos.item(), 0)
        client_clipped_value = central_norm/torch.norm(local_parameters_v)
        score_list.append(client_cos)
        FLTrustTotalScore += client_cos
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in local_parameters.items():
                sum_parameters[key] = client_cos * \
                    client_clipped_value * var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + client_cos * client_clipped_value * local_parameters[var]
    if FLTrustTotalScore == 0:
        print(score_list)
        return global_parameters
    for var in global_parameters:
        temp = (sum_parameters[var] / FLTrustTotalScore)
        if global_parameters[var].type() != temp.type():
            temp = temp.type(global_parameters[var].type())
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
        else:
            global_parameters[var] += temp * 1
    print(score_list)
    return global_parameters


def no_defence_balance(params, global_parameters):
    total_num = len(params)
    sum_parameters = None
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + params[i][var]
    for var in global_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
            continue
        global_parameters[var] += (sum_parameters[var] / total_num)

    return global_parameters


def flame(local_model, update_params, global_model, attacker_num,args):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list = []
    local_model_vector = []
    for param in local_model:
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1 - cos(local_model_vector[i], local_model_vector[j])
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(attacker_num)
    num_benign_clients = num_clients - num_malicious_clients
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients // 2 + 1, min_samples=1, allow_single_cluster=True).fit(cos_list)

    print(clusterer.labels_)
    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster = 0
    max_cluster_index = 0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model)):
            benign_client.append(i)
            norm_list = np.append(norm_list, torch.norm(parameters_dict_to_vector(update_params[i]), p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max() + 1):
            if len(clusterer.labels_[clusterer.labels_ == index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_ == index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
    for i in range(len(local_model_vector)):
        norm_list = np.append(norm_list, torch.norm(parameters_dict_to_vector(update_params[i]), p=2).item())
    print(benign_client)

    for i in range(len(benign_client)):
        if benign_client[i] < num_malicious_clients:
            args.wrong_mal += 1
        else:
            args.right_ben += 1
    args.turn += 1

    clip_value = np.median(norm_list)
    for i in range(len(benign_client)):
        gama = clip_value / norm_list[i]
        if gama < 1:
            for key in update_params[benign_client[i]]:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                update_params[benign_client[i]][key] *= gama
    global_model_weight = no_defence_balance([update_params[i] for i in benign_client], global_model)
    # add noise
    for key, var in global_model_weight.items():
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        temp = copy.deepcopy(var)
        noise = torch.FloatTensor(temp.shape).normal_(mean=0, std=args.noise * clip_value).to('cuda:0')
        var = temp + noise
    return global_model_weight

logger = logging.getLogger('logger')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
def gap_statistics(data, num_sampling, K_max, n):
    num_cluster = 0
    data = np.reshape(data, (data.shape[0], -1))
    # Linear transformation
    data_c = np.ndarray(shape=data.shape)
    for i in range(data.shape[1]):
        data_c[:, i] = (data[:, i] - np.min(data[:, i])) / \
                       (np.max(data[:, i]) - np.min(data[:, i]))
    gap = []
    s = []
    for k in range(1, K_max + 1):
        k_means = KMeans(n_clusters=k, init='k-means++').fit(data_c)
        predicts = (k_means.labels_).tolist()
        centers = k_means.cluster_centers_
        v_k = 0
        for i in range(k):
            for predict in predicts:
                if predict == i:
                    v_k += np.linalg.norm(centers[i] - \
                                          data_c[predicts.index(predict)])
        # perform clustering on fake data
        v_kb = []
        for _ in range(num_sampling):
            data_fake = []
            for i in range(n):
                temp = np.ndarray(shape=(1, data.shape[1]))
                for j in range(data.shape[1]):
                    temp[0][j] = random.uniform(0, 1)
                data_fake.append(temp[0])
            k_means_b = KMeans(n_clusters=k, init='k-means++').fit(data_fake)
            predicts_b = (k_means_b.labels_).tolist()
            centers_b = k_means_b.cluster_centers_
            v_kb_i = 0
            for i in range(k):
                for predict in predicts_b:
                    if predict == i:
                        v_kb_i += np.linalg.norm(centers_b[i] - data_fake[predicts_b.index(predict)])
            v_kb.append(v_kb_i)
        # gap for k
        v = 0
        for v_kb_i in v_kb:
            if v_kb_i == 0:
                continue
            v += math.log(v_kb_i)
        v /= num_sampling
        gap.append(v - math.log(v_k))
        sd = 0
        for v_kb_i in v_kb:
            if v_kb_i == 0:
                continue
            sd += (math.log(v_kb_i) - v) ** 2
        sd = math.sqrt(sd / num_sampling)
        s.append(sd * math.sqrt((1 + num_sampling) / num_sampling))
    # select smallest k
    for k in range(1, K_max + 1):
        print(gap[k - 1] - gap[k] + s[k - 1])
        if k == K_max:
            num_cluster = K_max
            break
        if gap[k - 1] - gap[k] + s[k - 1] > 0:
            num_cluster = k
            break
    return num_cluster


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

def get_top_50_percent_indices(lst):
    half_length = len(lst) // 2
    sorted_lst = sorted(lst, reverse=False)
    top_50_percent = sorted_lst[:half_length]
    indices = [lst.index(value) for value in top_50_percent]
    return indices



def flare(args, local_weights, detect_model):
    MMD = MMDLoss()
    print('=' * 15, 'FLARE', '=' * 15)
    server_detect_data = get_inspection_data(args)
    detecet_loader = DataLoader(
        server_detect_data,
        batch_size=100,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    detecet_feature_set = []
    for weight in local_weights:
        local_detecet_feature_list = []
        detect_model.load_state_dict(weight)
        detect_model.eval()

        for detect_img in detecet_loader:
            detect_img = detect_img.cuda(non_blocking=True)
            feature = detect_model.f(detect_img)
            feature = F.normalize(feature, dim=-1)
            local_detecet_feature_list.append(feature)
        detecet_feature_set.append(local_detecet_feature_list)


    nearst_neighbor_count = [0 for i in range(len(local_weights))]

    for client in tqdm(range(len(local_weights))):
        mmd_list = []
        for another_client in range(len(local_weights)):
            if another_client == client:
                continue
            for f in range(len(detecet_feature_set[client])):
                for f_ in range(len(detecet_feature_set[another_client])):
                    mmd = MMD(detecet_feature_set[client][f].detach().cpu(), detecet_feature_set[another_client][f_].detach().cpu())
            mmd_list.append(mmd)

        top_50_id = get_top_50_percent_indices(mmd_list)

        for id in top_50_id:
            if id <= client:
                nearst_neighbor_count[id] += 1
            else:
                nearst_neighbor_count[id+1] += 1


    trust_score = [0 for i in range(len(nearst_neighbor_count))]


    sum = np.sum(np.exp(nearst_neighbor_count))

    for i in range(len(trust_score)):
        trust_score[i] = np.exp(nearst_neighbor_count[i])/sum

    print('The trust score for each client:', trust_score)

    global_w = copy.deepcopy(local_weights[0])

    for key in global_w.keys():
        global_w[key] = global_w[key] * trust_score[0]
        for i in range(1, len(local_weights)):
            global_w[key] = global_w[key] + local_weights[i][key] * trust_score[i]

    return global_w


def RFLBAT(gradients, weights):
    eps1 = 10
    eps2 = 4
    dataAll = gradients

    pca = PCA(n_components=2)
    pca = pca.fit(dataAll)
    X_dr = pca.transform(dataAll)

    # Compute sum eu distance
    eu_list = []
    for i in range(len(X_dr)):
        eu_sum = 0
        for j in range(len(X_dr)):
            if i==j:
                continue
            eu_sum += np.linalg.norm(X_dr[i]-X_dr[j])
        eu_list.append(eu_sum)
    accept = []
    x1 = []
    for i in range(len(eu_list)):
        if eu_list[i] < eps1 * np.median(eu_list):
            accept.append(i)
            x1 = np.append(x1, X_dr[i])
        else:
            logger.info("RFLBAT: discard update {0}".format(i))
    x1 = np.reshape(x1, (-1, X_dr.shape[1]))
    num_clusters = gap_statistics(x1, num_sampling=5, K_max=10, n=len(x1))
    logger.info("RFLBAT: the number of clusters is {0}".format(num_clusters))
    k_means = KMeans(n_clusters=num_clusters, init='k-means++').fit(x1)
    predicts = k_means.labels_

    # select the most suitable cluster
    v_med = []
    for i in range(num_clusters):
        temp = []
        for j in range(len(predicts)):
            if predicts[j] == i:
                temp.append(dataAll[accept[j]])
        if len(temp) <= 1:
            v_med.append(1)
            continue
        v_med.append(np.median(np.average(smp\
            .cosine_similarity(temp), axis=1)))
    temp = []
    for i in range(len(accept)):
        if predicts[i] == v_med.index(min(v_med)):
            temp.append(accept[i])
    accept = temp

    # compute eu list again to exclude outliers
    temp = []
    for i in accept:
        temp.append(X_dr[i])
    X_dr = temp
    eu_list = []
    for i in range(len(X_dr)):
        eu_sum = 0
        for j in range(len(X_dr)):
            if i==j:
                continue
            eu_sum += np.linalg.norm(X_dr[i]-X_dr[j])
        eu_list.append(eu_sum)
    temp = []
    for i in range(len(eu_list)):
        if eu_list[i] < eps2 * np.median(eu_list):
            temp.append(accept[i])
        else:
            logger.info("RFLBAT: discard update {0}"\
                .format(i))
    accept = temp
    logger.info("RFLBAT: the final clients accepted are {0}"\
        .format(accept))

    weights_for_agg =[]
    print('Accepted clients:',accept)

    # aggregate
    for i in range(len(gradients)):
        if i in accept:
            weights_for_agg.append(weights[i])

    global_w = fedavg(weights_for_agg)

    return global_w






































def eminspector(args, local_weights, detect_model, global_model):
    print('='*15,'EmInspector','='*15)
    server_detect_data = get_inspection_data(args)
    detecet_loader = DataLoader(
        server_detect_data,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    detecet_feature_set = []
    for weight in local_weights:
        local_detecet_feature_list = []
        detect_model.load_state_dict(weight)
        detect_model.eval()

        for detect_img in tqdm(detecet_loader):
            detect_img = detect_img.cuda(non_blocking=True)
            feature = detect_model.f(detect_img)
            feature = F.normalize(feature, dim=-1)
            local_detecet_feature_list.append(feature)
        detecet_feature_set.append(local_detecet_feature_list)

    # initialize the malicious score
    malicious_score = []
    for i in range(len(detecet_feature_set)):
        malicious_score.append(0)

    print('=========== Malicious scores computing ==========')
    # caculate the cosine simlarity
    for detect_iter in tqdm(range(len(local_detecet_feature_list))):
        detecet_similarity_list = []
        for c in range(len(detecet_feature_set)):
            similarity = 0
            for client in range(len(detecet_feature_set)):
                similarity += torch.sum(detecet_feature_set[c][detect_iter] * detecet_feature_set[client][detect_iter], dim=-1).mean()
            similarity = similarity.cpu().detach().numpy()
            detecet_similarity_list.append(similarity)

        detecet_similarity_list = np.asarray(detecet_similarity_list).reshape(-1, 1)

        # compute avg
        sim_avg = np.mean(detecet_similarity_list)
        client_index = 0
        for sim in detecet_similarity_list:
            if sim >= sim_avg:
                malicious_score[client_index] += 1
            else:
                malicious_score[client_index] -= 1
            client_index += 1

    print('malicious scores = ', malicious_score)
    malicious_clients_index = []
    client_index = 0
    for s in malicious_score:
        if s > 0:
            malicious_clients_index.append(client_index)
        client_index += 1

    print('malicious client are:', malicious_clients_index)
    malicious_clients_index.reverse()

    for attacker in malicious_clients_index:
        local_weights.pop(attacker)

    if len(local_weights) == 0:
        global_weight = copy.deepcopy(global_model.state_dict())
    else:
        global_weight = fedavg(local_weights)

    return global_weight