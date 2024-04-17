import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--rounds', type=int, default=200, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--nclass', type=int, default=2, help="classes or shards per user")
    parser.add_argument('--nsample_pc', type=int, default=250, 
                        help="number of samples per class or shard for each client")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=20, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--warmup_epoch', type=int, default=0, help="the number of pretrain local ep")
    parser.add_argument('--trial', type=float, default=1, help="the trial number")
    parser.add_argument('--mu', type=float, default=0.001, help="FedProx Regularizer")
    parser.add_argument('--k2', type=int, default=5, help="")

    # model arguments
    parser.add_argument('--model', type=str, default='lenet5', help='model name')
    parser.add_argument('--ks', type=int, default=5, help='kernel size to use for convolutions')
    parser.add_argument('--in_ch', type=int, default=3, help='input channels of the first conv layer')
    parser.add_argument("--model_wd", type=float, default=5e-5, help="model weight decay")

    # hypernetwork arguments
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--hn_client_mlp_lr", type=float, default=1e-3)
    parser.add_argument("--hn_client_em_lr", type=float, default=5e-3)
    parser.add_argument("--hn_cluster_mlp_lr", type=float, default=1e-3)
    parser.add_argument("--hn_cluster_em_lr", type=float, default=5e-3)
    parser.add_argument("--activation", type=str, default='relu')
    parser.add_argument("--init_way", type=str, default='xavier')
    parser.add_argument("--hn_wd", type=float, default=1e-3, help="hypernetwork weight decay")
    parser.add_argument("--manual_update_grad", action='store_true', help='update mlp gradient')
    parser.add_argument("--hn_optim", type=str, default='sgd',help='optimter')

    # dataset partitioning arguments
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        help="name of dataset: mnist, cifar10, cifar100")
    parser.add_argument('--noniid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--shard', action='store_true', help='whether non-i.i.d based on shard or not')
    parser.add_argument('--label', action='store_true', help='whether non-i.i.d based on label or not')
    parser.add_argument('--split_test', action='store_true', 
                        help='whether split test set in partitioning or not')
    
    # NIID Benchmark dataset partitioning 
    parser.add_argument('--savedir', type=str, default='../save_results/', help='save directory')
    parser.add_argument('--datadir', type=str, default='../data/', help='data directory')
    parser.add_argument('--logdir', type=str, default='../logs/', help='logs directory')
    parser.add_argument('--partition', type=str, default='noniid-#label2', help='method of partitioning')
    parser.add_argument('--alg', type=str, default='hncfl', help='Algorithm')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data'\
                        'partitioning')
    parser.add_argument('--local_view', action='store_true', help='whether from client perspective or not')
    parser.add_argument('--batch_size', type=int, default=64, help="test batch size")
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    
    # clustering arguments 
    parser.add_argument('--gamma', type=float, default=0.5, help="the clustering threshold for FRL")
    parser.add_argument('--max_iter', type=int, default=500, help="the maximum number of iterations for FRL")
    parser.add_argument('--cluster_alpha', type=float, default=3.5, help="the clustering threshold")
    parser.add_argument('--n_basis', type=int, default=5, help="number of basis per label")
    parser.add_argument('--linkage', type=str, default='average', help="Type of Linkage for HC")

    parser.add_argument('--nclasses', type=int, default=10, help="number of classes")
    parser.add_argument('--nsamples_shared', type=int, default=2500, help="number of shared data samples")
    parser.add_argument('--nclusters', type=int, default=3, help="Number of Clusters")
    parser.add_argument('--num_incluster_layers', type=int, default=2, help="Number of Clusters for IFCA")
    
    # pruning arguments 
    parser.add_argument('--pruning_percent', type=float, default=10, 
                        help="Pruning percent for layers (0-100)")
    parser.add_argument('--pruning_target', type=int, default=30, 
                        help="Total Pruning target percentage (0-100)")
    parser.add_argument('--dist_thresh', type=float, default=0.0001, 
                        help="threshold for fcs masks difference ")
    parser.add_argument('--acc_thresh', type=int, default=50, 
                        help="accuracy threshold to apply the derived pruning mask")
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    
    # other arguments 
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--is_print', action='store_true', help='verbose print')
    parser.add_argument('--is_print_grad', action='store_true', help='print gradient')

    parser.add_argument('--print_freq', type=int, default=100, help="printing frequency during training rounds")
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 1)')
    parser.add_argument('--load_initial', type=str, default='', help='define initial model path')

    args = parser.parse_args()
    return args