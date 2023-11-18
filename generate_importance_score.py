import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import os, sys
import argparse
import pickle

import models.resnet_ap as RNAP
from ccs.core.model_generator import wideresnet, preact_resnet, resnet, convnet
from ccs.core.training import Trainer, TrainingDynamicsLogger
from ccs.core.data import IndexDataset, CIFARDataset, SVHNDataset, CINIC10Dataset
from ccs.core.utils import print_training_info, StdRedirect

from utils.img_loader import load_data_path
from fast_pytorch_kmeans import KMeans

class IntListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, [int(v) for v in values.split(',')])

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

######################### Data Setting #########################
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training.')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--score_file', type=str, default='', help='directory of ccs scores (sorted index in pickle file)')
parser.add_argument('--condense_key', type=str, default='idc', choices=['idc', 'dream', 'dsa', 'kip', 'mtt'], help=['type of condensed dataset'])

######################### Path Setting #########################
parser.add_argument('--data-dir', type=str, default='~/scratch',
                    help='The dir path of the data.')
parser.add_argument('--imagenet-dir', type=str, default='/mnt/data2/usertwo/dataset/Imagenet',
                    help='The dir path of the imagenet.')
parser.add_argument('--base-dir', type=str, default='raid/dynamics_and_scores/cifar10',
                    help='The base dir of this project.')
parser.add_argument('--task-name', type=str, default='tmp',
                    help='The name of the training task.')

######################## Network Setting ########################
parser.add_argument('--network', type=str, default='convnet', choices=['convnet', 'resnet10_ap'])

######################### CUSTOM DATADIR #########################
parser.add_argument('--custom-data-dir', type=str, default=None,
                     help='The dir path of the data.')
parser.add_argument('--custom-out-name', type=str, default=None,
                     help='The dir path of the data.')

######################### GPU Setting #########################
parser.add_argument('--gpuid', type=str, default='0',
                    help='The ID of GPU.')

########################### DEPENDENCY ###########################
parser.add_argument('--ipc', type=int, help='images per class')
parser.add_argument('--factor', type=int, default='2', help='factor of multi-formation')

parser.add_argument('--load_mtt', action='store_true', help='whether to load dsa model')
parser.add_argument('--load_dsa', action='store_true', help='whether to load dsa model')
parser.add_argument('--load_kip', action='store_true', help='whether to load kip model')

parser.add_argument('--random_selection', action='store_true', help='whether to select random samples') # can ignore
parser.add_argument('--reproduce_exp', type=str, default='reproduce_3', help='reproduce mark of experiment')

############################ HYPERPARAMETER ###########################
parser.add_argument('--stop_epoch', type=int, default=100, help='compute LBPE score before epoch E')
parser.add_argument('--topk', action=IntListAction, default=[10], help='top-K LBPE averaged')
parser.add_argument('--use-loss', action='store_true', help='whether to use top-K acc or least-K loss')

args = parser.parse_args()

if args.dataset in ['cifar10', 'cifar100']:

    args.factor = 2
    args.network = 'convnet'
    args.task_name = f"ipc{args.ipc}"

    if args.condense_key == 'idc':
        if args.dataset == 'cifar10':
            args.stop_epoch = 100
        elif args.dataset == 'cifar100':
            args.stop_epoch = 200
        args.topk = [10]
        args.base_dir=f"raid/{args.reproduce_exp}/dynamics_and_scores/idc/{args.dataset}"
        args.custom_data_dir = f"raid/condensed_img/idc/{args.dataset}/conv3in_grad_mse_nd2000_cut_niter2000_factor2_lr0.005_mix_ipc{args.ipc}"
    elif args.condense_key == 'dream':
        args.stop_epoch = 200
        args.topk = [10]
        args.base_dir=f"raid/{args.reproduce_exp}/dynamics_and_scores/dream/{args.dataset}"
        args.custom_data_dir = f"raid/condensed_img/dream/cifar10/ipc{args.ipc}"
    elif (args.dataset != 'cifar10') or (args.ipc != 10):
        raise NotImplementedError
    elif args.condense_key == 'mtt':
        args.factor = 1
        args.load_mtt = True
        args.stop_epoch = 10
        args.topk = [3]
        args.base_dir=f"raid/{args.reproduce_exp}/dynamics_and_scores/mtt/{args.dataset}"
        args.custom_data_dir = f"raid/condensed_img/mtt/ConvNet_baseline"
    elif args.condense_key == 'dsa':
        args.factor = 1
        args.load_dsa = True
        args.stop_epoch = 100
        args.topk = [3]
        args.base_dir=f"raid/{args.reproduce_exp}/dynamics_and_scores/dsa/{args.dataset}"
        args.custom_data_dir = f"raid/condensed_img/dsa/res_DSA_CIFAR10_ConvNet_ipc10"
    elif args.condense_key == 'kip':
        args.factor = 1
        args.load_kip = True
        args.stop_epoch = 50
        args.topk = [3]
        args.base_dir=f"raid/{args.reproduce_exp}/dynamics_and_scores/kip/{args.dataset}"
        args.custom_data_dir = f"raid/condensed_img/kip/kip_ipc10"
    else:
        raise NotImplementedError

    args.custom_out_name = f"data-score-{args.task_name}-ep{args.stop_epoch}"
elif 'imagenet' in args.dataset:
    args.stop_epoch = 200
    args.topk = [10]

    args.factor = 3
    args.network = 'resnet10_ap'
    args.task_name = f"ipc{args.ipc}"
    args.base_dir=f"raid/{args.reproduce_exp}/dynamics_and_scores/idc/imagenet10"
    args.custom_data_dir=f"raid/condensed_img/idc/imagenet10/resnet10apin_grad_l1_ely10_nd500_cut_factor3_mix_ipc{args.ipc}"
    args.custom_out_name = f"data-score-{args.task_name}-ep{args.stop_epoch}"
else:
    raise NotImplementedError

######################### Set path variable #########################
task_dir = os.path.join(args.base_dir, args.task_name)
ckpt_path = os.path.join(task_dir, f'ckpt-last.pt')
td_path = os.path.join(task_dir, f'td-{args.task_name}.pickle')

if args.custom_out_name is None:
    data_score_path = os.path.join(task_dir, f'data-score-{args.task_name}.pickle')
else:
    data_score_path = os.path.join(task_dir, f'{args.custom_out_name}.pickle')

######################### Print setting #########################
print_training_info(args, all=True)

#########################
dataset = args.dataset
if dataset in ['cifar10', 'imagenet']:
    num_classes=10
elif dataset == 'cifar100':
    num_classes=100

######################### Ftn definition #########################
def kmeans_metric(model, trainset, data_importance, factor):
    def get_embeddings(model, data):
        embed=model.embed
        features = []
        with torch.no_grad():
            for i_batch, datum in enumerate(data): 
                img = datum[0].float().cuda()
                output = embed(img)
                features.append(output)
            features = torch.cat(features, dim=0).detach()
        return features


    def euclidean_dist(x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def query(unlabeled_idxs, unlabeled_data, model, n):
        embeddings = get_embeddings(model, unlabeled_data)
        kmeans = KMeans(n_clusters=n, mode='euclidean', verbose=1)
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.centroids
        dist_matrix = euclidean_dist(centers, embeddings).cuda()
        unlabeled_idxs = unlabeled_idxs.cuda()
        q_idxs = unlabeled_idxs[torch.argmin(dist_matrix,dim=1)]
        return q_idxs
    
    for ipc in [1, 2, 5, 10]:
        cluster_num = ipc*factor**2
        query_idx_list = []
        for c in range(trainset.dataset.targets.max().item()+1):
            idxs = (trainset.dataset.targets == c).nonzero().squeeze()
            # add corresponding data to trainloader
            data = torch.utils.data.Subset(trainset.dataset, idxs)
            data_loader = torch.utils.data.DataLoader(data, batch_size=256, shuffle=False, num_workers=2)
            query_idx = query(idxs, data_loader, model, cluster_num)
            query_idx_list += query_idx.tolist()
        data_importance[f'kmeans_{ipc}'] = query_idx_list

    for ipc in [1, 2, 5, 10]:
        query_idx_list = []
        cluster_num = ipc*factor**2* (trainset.dataset.targets.max().item()+1)
        idxs = torch.tensor(range(len(trainset.dataset)))
        data = torch.utils.data.Subset(trainset.dataset, idxs)
        data_loader = torch.utils.data.DataLoader(data, batch_size=256, shuffle=False, num_workers=2)
        query_idx = query(idxs, data_loader, model, cluster_num)
        query_idx_list += query_idx.tolist()
        data_importance[f'imbalance_kmeans_{ipc}'] = query_idx_list

    return data_importance
    
    
"""Calculate loss and entropy"""
def post_training_metrics(model, dataloader, data_importance, device):
    model.eval()
    data_importance['entropy'] = torch.zeros(len(dataloader.dataset))
    data_importance['loss'] = torch.zeros(len(dataloader.dataset))

    for batch_idx, (idx, (inputs, targets)) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        logits = model(inputs)
        prob = nn.Softmax(dim=1)(logits)

        entropy = -1 * prob * torch.log(prob + 1e-10)
        entropy = torch.sum(entropy, dim=1).detach().cpu()

        loss = nn.CrossEntropyLoss(reduction='none')(logits, targets).detach().cpu()

        data_importance['entropy'][idx] = entropy
        data_importance['loss'][idx] = loss

"""Calculate td metrics"""
def training_dynamics_metrics(td_log, dataset, data_importance):
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        _, (_, y) = dataset[i]
        targets.append(y)
    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)

    data_importance['correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['forgetting'] = torch.zeros(data_size).type(torch.int32)
    data_importance['last_correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['accumulated_margin'] = torch.zeros(data_size)
    data_importance['last_correct_count'] = torch.zeros(data_size).type(torch.float32)

    l2_loss = torch.nn.MSELoss(reduction='none')
    epoch = td_log[-1]['epoch'] + 1

    data_importance['correct_num'] = torch.zeros(epoch).type(torch.int32)  # 1000 is hard code for cifar10
    data_importance['loss_epoch'] = torch.zeros(epoch).type(torch.float32)  # 1000 is hard code for cifar10

    def record_training_dynamics(td_log):
        # 64
        output = torch.exp(td_log['output'].type(torch.float))
        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)

        label = targets[index]

        correctness = (predicted == label).type(torch.int)

        label_onehot = torch.nn.functional.one_hot(label, num_classes=num_classes)

        data_importance['forgetting'][index] += torch.logical_and(data_importance['last_correctness'][index] == 1, correctness == 0)
        data_importance['last_correctness'][index] = correctness
        data_importance['correctness'][index] += data_importance['last_correctness'][index]

        batch_idx = range(output.shape[0])
        target_prob = output[batch_idx, label]
        output[batch_idx, label] = 0
        other_highest_prob = torch.max(output, dim=1)[0]
        margin = target_prob - other_highest_prob
        data_importance['accumulated_margin'][index] += margin

        data_importance['correct_num'][td_log["epoch"]] += correctness.sum()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(td_log['output'].type(torch.float), label)
        data_importance['loss_epoch'][td_log["epoch"]] += loss.detach().cpu()

    for i, item in enumerate(td_log):
        if i % 10000 == 0:
            print(i)

        record_training_dynamics(item)
    
"""Calculate td metrics"""
def LBPE(td_log, dataset, data_importance, start_epoch=0, max_epoch=10):
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        _, (_, y) = dataset[i]
        targets.append(y)
    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)
    data_importance[f'LBPE_{start_epoch}_{max_epoch}'] = torch.zeros(data_size).type(torch.float32)

    l2_loss = torch.nn.MSELoss(reduction='none')

    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))
        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)

        label = targets[index]

        label_onehot = torch.nn.functional.one_hot(label, num_classes=num_classes)
        LBPE_score = torch.sqrt(l2_loss(label_onehot,output).sum(dim=1))

        data_importance[f'LBPE_{start_epoch}_{max_epoch}'][index] += LBPE_score

    for i, item in enumerate(td_log):
        if i % 10000 == 0:
            print(i)
            
        if item['epoch'] == max_epoch:
            return

        if (item['epoch'] >= start_epoch) and (item['epoch'] < max_epoch):
            record_training_dynamics(item)

"""Calculate topk LBPE metrics"""
def compute_topk(stop_epoch, topk):
    if stop_epoch < topk:
        raise ValueError(f'stop_epoch {stop_epoch} < topk {topk}')


    if args.use_loss:
        print("use loss")
        data_importance["neg_loss_epoch"] = data_importance["loss_epoch"].clone() * (-1)
        topk_list = data_importance["neg_loss_epoch"][:stop_epoch].topk(topk)[1].tolist()
        name_flag = 'loss_'
    else:
        topk_list = data_importance["correct_num"][:stop_epoch].topk(topk)[1].tolist()    # take top k indices
        name_flag = ''

    print(f"top-{topk} acc", topk_list)

    rlist = []
    for i in topk_list:
        rlist.append((i, i+1))

    rlist = list(set(rlist))
    print(rlist)
    for start, end in rlist:
        LBPE(training_dynamics, trainset, data_importance, start_epoch =start, max_epoch=end)

    key_name = f"{name_flag}LBPE_top{topk}"
    for i in topk_list:
        if key_name not in data_importance.keys():
            data_importance[key_name] = data_importance['LBPE_{}_{}'.format(i, i+1)].clone()
        else:
            data_importance[key_name] += data_importance['LBPE_{}_{}'.format(i, i+1)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_identical = transforms.Compose([
            transforms.ToTensor(),
        ])

if args.custom_data_dir is not None:
    print("Loading custom data dir...")
    if 'cifar' in args.dataset:
        if args.load_mtt:
            ipc=10
        else:
            args.ipc = int(args.custom_data_dir.split('ipc')[-1])
        if args.dataset == 'cifar10':
            args.nclass = 10
        elif args.dataset == 'cifar100':
            args.nclass = 100
        else:
            raise NotImplementedError
        args.slct_type = 'idc'
        args.slct_ipc = 0   # dont select
        args.pretrained = False
        args.augment = False
        args.decode_type = "single"
        args.batch_syn_max = 128
        args.rrc = True
        args.pruning_indices = ''
        args.pruning_key = ''
        
        trainset, testset = load_data_path(args)
    elif args.dataset == 'imagenet':
        args.ipc = int(args.custom_data_dir.split('ipc')[-1])
        args.nclass = 10
        args.nch = 3
        args.slct_type = 'idc'
        args.slct_ipc = 0   # dont select
        args.dataset = 'imagenet'
        args.pretrained = False
        args.augment = True
        args.aug_type = "color_crop_cutout"
        args.factor = 3
        args.decode_type = "single"
        args.batch_syn_max = 128
        args.rrc = True
        args.pruning_indices = ''
        args.pruning_key = ''
        args.size = 224
        args.dseed = 0
        args.load_memory = True
        args.save_dir = args.custom_data_dir

        trainset, testset = load_data_path(args)

else:
    if args.dataset == 'cifar10':
        data_dir =  args.data_dir
    else:
        data_dir =  os.path.join(args.data_dir, dataset)
        
    print(f'dataset: {dataset}')
    if dataset == 'cifar10':
        trainset = CIFARDataset.get_cifar10_train(data_dir, transform = transform_identical)
    elif dataset == 'cifar100':
        trainset = CIFARDataset.get_cifar100_train(data_dir, transform = transform_identical)

trainset = IndexDataset(trainset)
print(len(trainset))

data_importance = {}

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=False, num_workers=16)

if args.network == 'convnet':
    model = convnet(num_classes)
elif args.network == 'resnet10_ap':
    print('resnet10_ap')
    model = RNAP.ResNetAP(args.dataset, depth=10, num_classes=args.nclass, norm_type='instance', nch=args.nch)

model = model.to(device)

print(f'Ckpt path: {ckpt_path}.')
checkpoint = torch.load(ckpt_path)['model_state_dict']
model.load_state_dict(checkpoint)
model.eval()

with open(td_path, 'rb') as f:
     pickled_data = pickle.load(f)

training_dynamics = pickled_data['training_dynamics']

# =================== PRUNING METRICS ===================
kmeans_metric(model, trainset, data_importance, args.factor)
post_training_metrics(model, trainloader, data_importance, device)
training_dynamics_metrics(training_dynamics, trainset, data_importance)
import time
start = time.time()
LBPE(training_dynamics, trainset, data_importance)
data_importance['el2n'] = data_importance['LBPE_0_10']
print("Times used for LBPE:", time.time() - start)

stop_epoch = args.stop_epoch
for k in args.topk:
    print(k)
    compute_topk(stop_epoch, k)

print(f'Saving data score at {data_score_path}')
with open(data_score_path, 'wb') as handle:
    pickle.dump(data_importance, handle)