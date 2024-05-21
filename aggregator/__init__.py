import numpy as np

from aggregator.aggregating_algorithm import fedavg, eminspector, krum, trimmed_mean, fltrust, get_grad, flame, RFLBAT, flare
import copy
import torch


def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.view(-1).to('cpu'))
    return torch.cat(vec)

def aggregating(args, local_weights, detect_model, global_model):
    if args.aggregator == 'fedavg':
        return fedavg(local_weights)
    elif args.aggregator == 'krum':
        return krum(local_weights, 3)
    elif args.aggregator == 'trimmed-mean':
        return trimmed_mean(2, local_weights)
    elif args.aggregator == 'flame':
        global_para = copy.deepcopy(global_model.state_dict())
        grad_list = []
        for local_update in local_weights:
            grad_list.append(get_grad(local_update, global_para))
        return flame(local_weights, grad_list, global_para, args.num_attackers, args)
    elif args.aggregator == 'rflbat':
        global_para = copy.deepcopy(global_model.state_dict())
        grad_list = []
        for local_update in local_weights:
            grad_list.append(get_grad(local_update, global_para))
        for i in range(len(grad_list)):
            grad_list[i] = parameters_dict_to_vector_flt(grad_list[i])

        return RFLBAT(grad_list, local_weights)
    elif args.aggregator == 'flare':
        return flare(args, local_weights, detect_model)
    elif args.aggregator == 'EmInspector':
        return eminspector(args, local_weights, detect_model, global_model)



