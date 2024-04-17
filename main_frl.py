import numpy as np

import time
import copy
import os
import gc
import json

import torch
import torchvision

from src.data import *
from src.models import *
from src.fedavg import *
from src.client.client_frl import *
from src.utils import * 
from src.clustering import *

from collections import OrderedDict
from typing import List

start_time = time.time()
communication_time = 0

##################################### Init args
args = args_parser()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.gpu) ## Setting cuda on GPU 

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass
    
save_path = RESULTS_DIR / args.dataset / args.partition / args.alg / f'cluster{args.nclusters}_k1_{args.local_ep}_k2_{args.k2}_gamma{args.gamma}'
if not os.path.isdir(save_path):
    mkdirs(save_path)

template = "Algorithm {}, Clients {}, Dataset {}, Model {}, Non-IID {}, Threshold {}, K {}, Linkage {}, LR {}, Ep {}, Rounds {}, bs {}, frac {}"
s = template.format(args.alg, 
                    args.num_users, 
                    args.dataset, 
                    args.model, 
                    args.partition, 
                    args.cluster_alpha, 
                    args.n_basis, 
                    args.linkage, 
                    args.lr, 
                    args.local_ep,
                    args.rounds, 
                    args.local_ep, 
                    args.frac)
print(s)
print(str(args))

fix_random_seed(args.seed)

##################################### Data partitioning section
print(f'###### Data partitioning section ######')  
args.local_view = True
X_train, y_train, X_test, y_test, net_dataidx_map, net_dataidx_map_test, \
traindata_cls_counts, testdata_cls_counts, position = partition_data_(args.dataset,
DATASETS_DIR, LOG_DIR, args.partition, args.num_users, beta=args.beta, local_view=args.local_view) 

# train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
#                                                                                    DATASETS_DIR,
#                                                                                    args.batch_size,
#                                                                                    32)

# print("len train_ds_global:", len(train_ds_global))
# print("len test_ds_global:", len(test_ds_global))

################################### build model
print(f'###### build model ######')  
def init_nets(args, dropout_p=0.5):

    users_model = []

    for net_i in range(-1, args.num_users):
        if args.dataset == "generated":
            net = PerceptronModel().to(args.device)
        elif args.model == "mlp":
            if args.dataset == 'covtype':
                input_size = 54
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'a9a':
                input_size = 123
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'rcv1':
                input_size = 47236
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'SUSY':
                input_size = 18
                output_size = 2
                hidden_sizes = [16,8]
            net = FcNet(input_size, hidden_sizes, output_size, dropout_p).to(args.device)
        elif args.model == "vgg":
            net = vgg11().to(args.device)
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).to(args.device)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10).to(args.device)
            elif args.dataset == 'celeba':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2).to(args.device)
            elif args.dataset == 'gtsrb':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=43).to(args.device)
            elif args.dataset == 'miotcd':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=11).to(args.device)  
            elif args.dataset == 'vehicle10':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).to(args.device)  
            elif args.dataset == 'cropped_lisa':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=7).to(args.device)
            elif args.dataset == 'tlight10':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).to(args.device)      
        elif args.model =="simple-cnn-3":
            if args.dataset == 'cifar100': 
                net = SimpleCNN_3(input_dim=(16 * 3 * 5 * 5), hidden_dims=[120*3, 84*3], output_dim=100).to(args.device)
            if args.dataset == 'tinyimagenet':
                net = SimpleCNNTinyImagenet_3(input_dim=(16 * 3 * 13 * 13), hidden_dims=[120*3, 84*3], 
                                              output_dim=200).to(args.device)
        elif args.model == "vgg-9":
            if args.dataset in ("mnist", 'femnist'):
                net = ModerateCNNMNIST().to(args.device)
            elif args.dataset in ("cifar10", "cinic10", "svhn"):
                # print("in moderate cnn")
                net = ModerateCNN().to(args.device)
            elif args.dataset == 'celeba':
                net = ModerateCNN(output_dim=2).to(args.device)
        elif args.model == 'resnet34':
            if args.dataset == 'stanford_cars':
                net = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
                num_ftrs = net.fc.in_features
                net.fc = nn.Linear(num_ftrs, 196)
                net = net.to(args.device)
        elif args.model == 'resnet18':
            if args.dataset == 'stanford_cars':
                net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
                num_ftrs = net.fc.in_features
                net.fc = nn.Linear(num_ftrs, 196)
                net = net.to(args.device)        
        elif args.model == 'resnet9': 
            if args.dataset == 'cifar100': 
                net = ResNet9(in_channels=3, num_classes=100)
            elif args.dataset == 'tinyimagenet': 
                net = ResNet9(in_channels=3, num_classes=200, dim=512*2*2)
            elif args.dataset == 'vehicle10': 
                net = ResNet9(in_channels=3, num_classes=10).to(args.device)
        elif args.model == "resnet":
            net = ResNet50_cifar10().to(args.device)
        elif args.model == "vgg16":
            net = vgg16().to(args.device)
        else:
            print("not supported yet")
            exit(1)
        if net_i == -1: 
            net_glob = copy.deepcopy(net)

            initial_state_dict = copy.deepcopy(net_glob.state_dict())
            server_state_dict = copy.deepcopy(net_glob.state_dict())

            w_per_cluster = [copy.deepcopy(OrderedDict({key: value for key, value in net_glob.named_parameters()})) for _ in range(args.nclusters)]
            
            if args.load_initial:
                initial_state_dict = torch.load(args.load_initial)
                server_state_dict = torch.load(args.load_initial)
                net_glob.load_state_dict(initial_state_dict)
        else:
            users_model.append(copy.deepcopy(net))
            users_model[net_i].load_state_dict(initial_state_dict)
    
#     model_meta_data = []
#     layer_type = []
#     for (k, v) in nets[0].state_dict().items():
#         model_meta_data.append(v.shape)
#         layer_type.append(k)

    return users_model, net_glob, w_per_cluster, initial_state_dict, server_state_dict

print(f'MODEL: {args.model}, Dataset: {args.dataset}')

users_model, net_glob, w_per_cluster, initial_state_dict, server_state_dict = init_nets(args, dropout_p=0.5)
print(net_glob)

total = 0 
for name, param in net_glob.named_parameters():
    print(name, param.size())
    total += np.prod(param.size())
    #print(np.array(param.data.cpu().numpy().reshape([-1])))
    #print(isinstance(param.data.cpu().numpy(), np.array))
print(total)

################################# Initializing Clients
print(f'###### Initializing Clients ######')  
clients = []
    
for idx in range(args.num_users):
    
    dataidxs = net_dataidx_map[idx]
    if net_dataidx_map_test is None:
        dataidx_test = None 
    else:
        dataidxs_test = net_dataidx_map_test[idx]

    #print(f'Initializing Client {idx}')

    noise_level = args.noise
    if idx == args.num_users - 1:
        noise_level = 0

    if args.noise_type == 'space':
        train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, 
                                                                       DATASETS_DIR, args.local_bs, 128, 
                                                                       dataidxs, noise_level, idx, 
                                                                       args.num_users-1, 
                                                                       dataidxs_test=dataidxs_test)
    else:
        noise_level = args.noise / (args.num_users - 1) * idx
        train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, 
                                                                       DATASETS_DIR, args.local_bs, 128, 
                                                                       dataidxs, noise_level, 
                                                                       dataidxs_test=dataidxs_test)

    clients.append(Client_FRL(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep, 
               args.lr, args.momentum, args.device, args.model_wd, train_dl_local, test_dl_local))

del users_model
torch.cuda.empty_cache()

###################################### Clustering
print(f'###### Clustering ######')    
num_cls = {'gtsrb':43, 'miotcd': 11, 'vehicle10': 10, 'stanford_cars':196, 'cropped_lisa':7, 'tlight10':10}

kregions = KRegionsClassification(traindata_cls_counts, position, d=num_cls[args.dataset], 
                                  k=args.nclusters, max_iter=args.max_iter, gamma=args.gamma)

_, _, cluster_identifier, _ = kregions.fit()

clusters = {cid: [] for cid in range(args.nclusters)}
for idx in range(args.num_users):
    clusters[cluster_identifier[idx]].append(idx)

print(cluster_identifier)
print(clusters)

###################################### build hypernetwork
print(f'###### build hypernetwork ######') 
cluster_hypernetwork = [
    HyperNetwork(
            embedding_dim=args.embedding_dim,
            clients=clusters[cid],
            hidden_dim=args.hidden_dim,
            device=args.device,
            init_way=args.init_way,
        ) for cid in range(args.nclusters)
]

global_hypernetwork = HyperNetwork(
            embedding_dim=args.embedding_dim,
            clients=[i for i in range(args.nclusters)],
            hidden_dim=args.hidden_dim,
            device=args.device,
            init_way=args.init_way,
            ) 

###################################### Federation

loss_train = []

init_tacc_pr = []  # initial test accuarcy for each round 
final_tacc_pr = [] # final test accuracy for each round

init_tloss_pr = []  # initial test loss for each round 
final_tloss_pr = [] # final test loss for each round 

clients_best_acc = [0 for _ in range(args.num_users)]
w_locals = [OrderedDict({}) for idx in range(args.num_users)]
loss_locals = []

init_local_tacc = []       # initial local test accuracy at each round 
final_local_tacc = []      # final local test accuracy at each round 

init_local_tloss = []      # initial local test loss at each round 
final_local_tloss = []     # final local test loss at each round 

ckp_avg_tacc = []
ckp_avg_best_tacc = []

users_best_acc = [0 for _ in range(args.num_users)]
best_glob_acc = [0 for _ in range(args.nclusters)]


# init optimizer
cluster_hn_optimizer = []
global_hn_optimizer = []
if not args.manual_update_grad:
    for i in range(args.num_users):
        if args.hn_optim == 'sgd':
            cluster_hn_optimizer.append(
                torch.optim.SGD(
                    [
                        {'params': cluster_hypernetwork[cluster_identifier[i]].mlp_parameters(i), 'lr': args.hn_client_mlp_lr},
                        {'params': cluster_hypernetwork[cluster_identifier[i]].emd_parameters(), 'lr': args.hn_client_em_lr}
                    ], lr=args.hn_client_mlp_lr, momentum=0.9, weight_decay=args.hn_wd
                )
            )
        elif args.hn_optim == 'adam':
            cluster_hn_optimizer.append(
                torch.optim.Adam(params=[
                        {'params': cluster_hypernetwork[cluster_identifier[i]].mlp_parameters(i), 'lr': args.hn_client_mlp_lr},
                        {'params': cluster_hypernetwork[cluster_identifier[i]].emd_parameters(), 'lr': args.hn_client_em_lr}
                    ], lr=args.hn_client_mlp_lr
                )
            )
    for i in range(args.nclusters):
        if args.hn_optim == 'sgd':
            global_hn_optimizer.append(    
                torch.optim.SGD(
                    [
                        {'params': global_hypernetwork.mlp_parameters(i), 'lr': args.hn_cluster_mlp_lr},
                        {'params': global_hypernetwork.emd_parameters(), 'lr': args.hn_cluster_em_lr}
                    ], lr=args.hn_cluster_mlp_lr, momentum=0.9, weight_decay=args.hn_wd
                )
            )
        elif args.hn_optim == 'adam':
            global_hn_optimizer.append(
                torch.optim.Adam(params=[
                        {'params': global_hypernetwork.mlp_parameters(i), 'lr': args.hn_cluster_mlp_lr},
                        {'params': global_hypernetwork.emd_parameters(), 'lr': args.hn_cluster_em_lr}
                    ], lr=args.hn_cluster_mlp_lr
                )
            )


# generate cluster model parameters
personlized_w_per_cluster: List[OrderedDict[str, torch.Tensor]] = [OrderedDict({}) for _ in range(args.nclusters)]
for cid in range(args.nclusters):
    cluster_alpha = global_hypernetwork(cid, args.activation)
    cluster_alpha = cluster_alpha / cluster_alpha.sum()
    print(f"cluster {cid}, alpha {cluster_alpha}")
    personlized_w_per_cluster[cid] = personalize(cid, w_per_cluster, cluster_alpha) # HNCFLV2 is different from the HNCFLV1

w_per_cluster = personlized_w_per_cluster

del personlized_w_per_cluster
torch.cuda.empty_cache()

# distribute personlized clustered model to client
for cid in range(args.nclusters):
    for i in clusters[cid]:
        w_locals[i] = clone_parameters_with_detach(w_per_cluster[cid])


print_flag = False
for iteration in range(args.rounds):

    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    print(f'###### ROUND {iteration+1} ######')
    print(f'Clients {idxs_users}')

    personlized_w_per_client = [None for _ in range(args.num_users)]
    # for idx in range(args.num_users):
    for idx in idxs_users:
        # generate personlized client model parameters
        cid = cluster_identifier[idx]
        alpha = cluster_hypernetwork[cid](idx, args.activation)
        alpha = alpha / alpha.sum()

        # print(f'client {idx}, personized alpha {alpha}, cluster member {clusters[cid]}')

        # w_locals_cluster = [copy.deepcopy(w_locals[i]) for i in clusters[cid]]
        w_locals_cluster = [w_locals[i] for i in clusters[cid]]

        tmp_id = None
        for i in range(len(clusters[cid])):
            if clusters[cid][i] == idx:
                tmp_id = i
                break

        personlized_parameters = personalize(tmp_id, w_locals_cluster, alpha) # HNCFLV2 is different from the HNCFLV1
        personlized_w_per_client[idx] = clone_parameters_without_detach(personlized_parameters)
        clients[idx].set_state_dict(personlized_parameters)
        
        loss, acc = clients[idx].eval_test()
        init_local_tacc.append(acc)
        init_local_tloss.append(loss)

        loss = clients[idx].train()
        loss_locals.append(copy.deepcopy(loss))

        loss, acc = clients[idx].eval_test()
        if acc > clients_best_acc[idx]:
            clients_best_acc[idx] = acc
  
        final_local_tacc.append(acc)
        final_local_tloss.append(loss)   

        diff = clients[idx].dw

        # print(f"client {0}, delta w {diff}")

        # update client hypernetwork
        hn_grads = torch.autograd.grad(
            outputs=list(
                filter(
                    lambda param: param.requires_grad,
                    list(personlized_w_per_client[idx].values()),
                )
            ),
            inputs=cluster_hypernetwork[cid].mlp_parameters(idx)
            + cluster_hypernetwork[cid].emd_parameters(),
            grad_outputs=list(
                map(
                    lambda tup: tup[1],
                    filter(
                        lambda tup: tup[1].requires_grad ,
                        diff.items(),
                    ),
                )
            ),
            allow_unused=True,
        )

        mlp_grads = hn_grads[: len(cluster_hypernetwork[cid].mlp_parameters(idx))]
        emd_grads = hn_grads[len(cluster_hypernetwork[cid].mlp_parameters(idx)) :]

        if args.is_print_grad:
            print(f"mlp_grads {mlp_grads}")
            print(f"emd_grads {emd_grads}")

        if args.manual_update_grad:
            for param, grad in zip(cluster_hypernetwork[cid].mlp_parameters(idx), mlp_grads):
                param.data -= args.hn_client_mlp_lr * grad
            for param, grad in zip(cluster_hypernetwork[cid].emd_parameters(), emd_grads):
                param.data -= args.hn_client_em_lr * grad
        else:
            cluster_hn_optimizer[idx].zero_grad()
            for param, grad in zip(cluster_hypernetwork[cid].mlp_parameters(idx), mlp_grads):
                param.grad = grad
            for param, grad in zip(cluster_hypernetwork[cid].emd_parameters(), emd_grads):
                param.grad = grad
            cluster_hn_optimizer[idx].step()    

        # update client model parameters
        updated_params = OrderedDict(
            {
                k: (p + delta).clone().detach().requires_grad_(delta.requires_grad)
                for (k, p), delta in zip(
                    personlized_w_per_client[idx].items(), diff.values()
                )
            }
        )
        
        personlized_w_per_client[idx] = updated_params

        del w_locals_cluster
        del hn_grads
        del mlp_grads
        del emd_grads
        torch.cuda.empty_cache()

    for idx in idxs_users:
        w_locals[idx] = personlized_w_per_client[idx]

    del personlized_w_per_client
    torch.cuda.empty_cache()

    if (iteration + 1) % args.k2 == 0:
        # aggregation
        w_aggregation_per_cluster: List[OrderedDict[str, torch.Tensor]] = []
        
        for cid in range(args.nclusters):
            
            # w_locals_cluster = [copy.deepcopy(w_locals[i]) for i in clusters[cid]]
            w_locals_cluster = [w_locals[i] for i in clusters[cid]]

            w_avg = FedAvg(w_locals_cluster)
            w_avg = flatten(w_avg)

            beta = []
            total = 0
            for j in range(len(w_locals_cluster)):
                w_local = flatten(w_locals_cluster[j])
                total += penalty_function(torch.norm(w_local - w_avg))

            for j in range(len(w_locals_cluster)):
                w_local = flatten(w_locals_cluster[j])
                beta.append(penalty_function(torch.norm(w_local - w_avg)) / total)

            # print(f"cluster {cid}, beta {beta}")

            w_aggregation_per_cluster.append(aggregate(w_locals_cluster, beta))

            del w_locals_cluster
            torch.cuda.empty_cache()
        
        delta_w_per_cluster: List[OrderedDict[str, torch.Tensor]] = []
        for cid in range(args.nclusters):
            delta_w = OrderedDict(
                {key : torch.zeros_like(value) for key, value in w_per_cluster[cid].items()}
            )
            subtract_(delta_w, w_aggregation_per_cluster[cid], w_per_cluster[cid])
            delta_w_per_cluster.append(delta_w)

        del w_aggregation_per_cluster
        torch.cuda.empty_cache()

        # update cluster hypernetwork
        for cid in range(args.nclusters):
            global_hn_grads = torch.autograd.grad(
                outputs=list(
                    filter(
                        lambda param: param.requires_grad,
                        list(w_per_cluster[cid].values()),
                    )
                ),
                inputs=global_hypernetwork.mlp_parameters(cid)
                + global_hypernetwork.emd_parameters(),
                grad_outputs=list(
                    map(
                        lambda tup: tup[1],
                        filter(
                            lambda tup: tup[1].requires_grad,
                            delta_w_per_cluster[cid].items(),
                        ),
                    )
                ),
                allow_unused=True,
            )
            
            global_mlp_grads = global_hn_grads[: len(global_hypernetwork.mlp_parameters(cid))]
            global_emd_grads = global_hn_grads[len(global_hypernetwork.mlp_parameters(cid)) :]

            if args.is_print_grad:
                print(f"cluster {cid}, global_mlp_grads {global_mlp_grads}")
                print(f"cluster {cid}, global_emd_grads {global_emd_grads}")

            if args.manual_update_grad:
                for param, grad in zip(global_hypernetwork.mlp_parameters(cid), global_mlp_grads):
                    param.data -= args.hn_cluster_mlp_lr * grad

                for param, grad in zip(global_hypernetwork.emd_parameters(), global_emd_grads):
                    param.data -= args.hn_cluster_em_lr * grad
            else:
                global_hn_optimizer[cid].zero_grad()
                for param, grad in zip(global_hypernetwork.mlp_parameters(cid), global_mlp_grads):
                    param.grad = grad
                for param, grad in zip(global_hypernetwork.emd_parameters(), global_emd_grads):
                    param.grad = grad
                global_hn_optimizer[cid].step()
                
        # update cluster model parameters
        for cid in range(args.nclusters):
            with torch.no_grad():
                cluster_updated_params = OrderedDict(
                    {
                        k: (p + delta).clone().detach().requires_grad_(delta.requires_grad)
                        for (k, p), delta in zip(
                            w_per_cluster[cid].items(), delta_w_per_cluster[cid].values()
                        )
                    }
                )
                
                w_per_cluster[cid] = cluster_updated_params

        del delta_w_per_cluster
        del global_hn_grads
        del global_mlp_grads
        del global_emd_grads
        torch.cuda.empty_cache()

        # generate cluster model parameters
        personlized_w_per_cluster: List[OrderedDict[str, torch.Tensor]] = [OrderedDict({}) for _ in range(args.nclusters)]
        for cid in range(args.nclusters):
            cluster_alpha = global_hypernetwork(cid, args.activation)
            cluster_alpha = cluster_alpha / cluster_alpha.sum()
            print(f"cluster {cid}, alpha {cluster_alpha}")
            personlized_w_per_cluster[cid] = personalize(cid, w_per_cluster, cluster_alpha) # HNCFLV2 is different from the HNCFLV1

        w_per_cluster = personlized_w_per_cluster

        del personlized_w_per_cluster
        torch.cuda.empty_cache()

        # distribute personlized clustered model to client
        for cid in range(args.nclusters):
            for i in clusters[cid]:
                w_locals[i] = clone_parameters_with_detach(w_per_cluster[cid])
                
                torch.cuda.empty_cache()
                gc.collect()


    # print loss
    loss_avg = sum(loss_locals) / len(loss_locals)
    avg_init_tloss = sum(init_local_tloss) / len(init_local_tloss)
    avg_init_tacc = sum(init_local_tacc) / len(init_local_tacc)
    avg_final_tloss = sum(final_local_tloss) / len(final_local_tloss)
    avg_final_tacc = sum(final_local_tacc) / len(final_local_tacc)
    
    print('## END OF ROUND ##')
    template = 'Average Train loss {:.3f}'
    print(template.format(loss_avg))
    
    template = "AVG Init Test Loss: {:.3f}, AVG Init Test Acc: {:.3f}"
    print(template.format(avg_init_tloss, avg_init_tacc))
    
    template = "AVG Final Test Loss: {:.3f}, AVG Final Test Acc: {:.3f}"
    print(template.format(avg_final_tloss, avg_final_tacc))
    
    print_flag = False
    # if iteration < 60:
    #     print_flag = True
    if iteration%args.print_freq == 0: 
        print_flag = True

    if print_flag:
        print('--- PRINTING ALL CLIENTS STATUS ---')
        current_acc = []
        for k in range(args.num_users):
        # for k in idxs_users:
            loss, acc = clients[k].eval_test() 
            current_acc.append(acc)
            
            template = ("Client {:3d}, labels {}, count {}, best_acc {:3.3f}, current_acc {:3.3f} \n")
            print(template.format(k, traindata_cls_counts[k], clients[k].get_count(),
                                  clients_best_acc[k], current_acc[-1]))
            
        template = ("Round {:1d}, Avg current_acc {:3.3f}, Avg best_acc {:3.3f}")
        print(template.format(iteration+1, np.mean(current_acc), np.mean(clients_best_acc)))
        
        ckp_avg_tacc.append(np.mean(current_acc))
        ckp_avg_best_tacc.append(np.mean(clients_best_acc))

    print('----- Analysis End of Round -------')
    for idx in idxs_users:
        print(f'Client {idx}, Count: {clients[idx].get_count()}, Labels: {traindata_cls_counts[idx]}')

    print('')
    loss_train.append(loss_avg)
    
    init_tacc_pr.append(avg_init_tacc)
    init_tloss_pr.append(avg_init_tloss)
    
    final_tacc_pr.append(avg_final_tacc)
    final_tloss_pr.append(avg_final_tloss)

    # clear the placeholders for the next round
    loss_locals.clear()
    init_local_tacc.clear()
    init_local_tloss.clear()
    final_local_tacc.clear()
    final_local_tloss.clear()
    
    ## calling garbage collector 
    gc.collect()

end_time = time.time()

############################### Saving Training Results
result = {
    'args': str(args),
    'loss_train': [x.item() for x in loss_train] if isinstance(loss_train[0], torch.Tensor) else loss_train,
    'init_tacc_pr': [x.item() for x in init_tacc_pr] if isinstance(init_tacc_pr[0], torch.Tensor) else init_tacc_pr,
    'init_tloss_pr': [x.item() for x in init_tloss_pr] if isinstance(init_tloss_pr[0], torch.Tensor) else init_tloss_pr,
    'final_tacc_pr': [x.item() for x in final_tacc_pr] if isinstance(final_tacc_pr[0], torch.Tensor) else final_tacc_pr,
    'final_tloss_pr': [x.item() for x in final_tloss_pr] if isinstance(final_tloss_pr[0], torch.Tensor) else final_tloss_pr,
}

with open(save_path / f'{args.trial}_result.json', 'w') as fp:
    json.dump(result, fp)    

with open(save_path / f'{args.trial}_loss_train.npy', 'wb') as fp:
    loss_train = np.array(loss_train)
    np.save(fp, loss_train)

with open(save_path / f'{args.trial}_init_tacc_pr.npy', 'wb') as fp:
    init_tacc_pr = np.array(init_tacc_pr)
    np.save(fp, init_tacc_pr)
    
with open(save_path / f'{args.trial}_init_tloss_pr.npy', 'wb') as fp:
    init_tloss_pr = np.array(init_tloss_pr)
    np.save(fp, init_tloss_pr)

with open(save_path / f'{args.trial}_final_tacc_pr.npy', 'wb') as fp:
    final_tacc_pr = np.array(final_tacc_pr)
    np.save(fp, final_tacc_pr)
    
with open(save_path / f'{args.trial}_final_tloss_pr.npy', 'wb') as fp:
    final_tloss_pr = np.array(final_tloss_pr)
    np.save(fp, final_tloss_pr)

import pandas as pd
del result['args']
df = pd.DataFrame(result)
filename = save_path / f'{args.trial}_result.csv'
df.to_csv(filename, index=False)

############################### Printing Final Test and Train ACC / LOSS
test_loss = []
test_acc = []
train_loss = []
train_acc = []

for idx in range(args.num_users):        
    loss, acc = clients[idx].eval_test()

    test_loss.append(loss)
    test_acc.append(acc)
    
    loss, acc = clients[idx].eval_train()
    
    train_loss.append(loss)
    train_acc.append(acc)

test_loss = sum(test_loss) / len(test_loss)
test_acc = sum(test_acc) / len(test_acc)

train_loss = sum(train_loss) / len(train_loss)
train_acc = sum(train_acc) / len(train_acc)

print(f'Train Loss: {train_loss}, Test_loss: {test_loss}')
print(f'Train Acc: {train_acc}, Test Acc: {test_acc}')

print(f'Best Clients AVG Acc: {np.mean(clients_best_acc)}')

print(f'Computation Time: {end_time - start_time} s')

############################# Saving Print Results
with open(save_path / f'{args.trial}_final_results.txt', 'a') as text_file:
    print(f'Train Loss: {train_loss}, Test_loss: {test_loss}', file=text_file)
    print(f'Train Acc: {train_acc}, Test Acc: {test_acc}', file=text_file)

    print(f'Best Clients AVG Acc: {np.mean(clients_best_acc)}', file=text_file)