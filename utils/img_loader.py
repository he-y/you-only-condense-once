import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import pickle
from torch.utils.data import Subset
from coreset import randomselect, herding
from math import ceil
import random
from random import shuffle
from data import save_img, transform_imagenet, transform_cifar, transform_svhn, transform_mnist, transform_fashion, get_save_img
from data import TensorDataset, ImageFolder
import models.resnet as RN
import models.resnet_ap as RNAP
import models.densenet_cifar as DN
import models.convnet as CN
from efficientnet_pytorch import EfficientNet
import itertools
import matplotlib.pyplot as plt


def return_data_path(args):
    if args.factor > 1:
        init = 'mix'
    else:
        init = 'random'
    if args.dataset == 'imagenet' and args.nclass == 100:
        args.slct_type = 'idc_cat'
        args.nclass_sub = 20

    if 'idc' in args.slct_type:
        name = args.name
        if name == '':
            if args.dataset == 'cifar10':
                name = f'cifar10/conv3in{args.tag}' # tag is modified in argument.py already!
                name = name[:name.find('_ipc')] # remove ipc in the modifed tag
            elif args.dataset == 'cifar100':
                name = f'cifar100/conv3in_grad_mse_nd2000_cut_niter2000_factor{args.factor}_lr0.005_{init}'

            elif args.dataset == 'imagenet':
                if args.nclass == 10:
                    name = f'imagenet10/resnet10apin_grad_l1_ely10_nd500_cut_factor{args.factor}_{init}'
                elif args.nclass == 100:
                    name = f'imagenet100/resnet10apin_grad_l1_pt5_nd500_cut_nlr0.1_wd0.0001_factor{args.factor}_lr0.001_b_real128_{init}'

            elif args.dataset == 'svhn':
                name = f'svhn/conv3in_grad_mse_nd500_cut_niter2000_factor{args.factor}_lr0.005_{init}'
                if args.factor == 1 and args.ipc == 1:
                    args.mixup = 'vanilla'
                    args.dsa_strategy = 'color_crop_cutout_scale_rotate'

            elif args.dataset == 'mnist':
                if args.factor == 1:
                    name = f'mnist/conv3in_grad_l1_nd500_cut_niter2000_factor{args.factor}_lr0.0001_{init}'
                else:
                    name = f'mnist/conv3in_grad_l1_nd500_niter2000_factor{args.factor}_color_crop_lr0.0001_{init}'
                    args.mixup = 'vanilla'
                    args.dsa_strategy = 'color_crop_scale_rotate'

            elif args.dataset == 'fashion':
                name = f'fashion/conv3in_grad_l1_nd500_cut_niter2000_factor{args.factor}_lr0.0001_{init}'

        if args.slct_ipc > 0:
            path_list = [f'{name}_ipc{args.ipc_origin}']
        else:
            path_list = [f'{name}_ipc{args.ipc}']

    elif args.slct_type == 'dsa':
        path_list = [f'cifar10/dsa/res_DSA_CIFAR10_ConvNet_{args.ipc}ipc']
    elif args.slct_type == 'kip':
        path_list = [f'cifar10/kip/kip_ipc{args.ipc}']
    else:
        path_list = ['']

    return path_list



def load_ckpt(model, file_dir, verbose=True):
    checkpoint = torch.load(file_dir)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    checkpoint = remove_prefix_checkpoint(checkpoint, 'module')
    model.load_state_dict(checkpoint)

    if verbose:
        print(f"\n=> loaded checkpoint '{file_dir}'")


def remove_prefix_checkpoint(dictionary, prefix):
    keys = sorted(dictionary.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) + 1:]
            dictionary[newkey] = dictionary.pop(key)
    return dictionary


def decode_zoom(img, target, factor, size=-1):
    if size == -1:
        size = img.shape[-1]
    resize = nn.Upsample(size=size, mode='bilinear')

    h = img.shape[-1]
    remained = h % factor
    if remained > 0:
        img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
    s_crop = ceil(h / factor)
    n_crop = factor**2

    cropped = []
    for i in range(factor):
        for j in range(factor):
            h_loc = i * s_crop
            w_loc = j * s_crop
            cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
    cropped = torch.cat(cropped)
    data_dec = resize(cropped)
    target_dec = torch.cat([target for _ in range(n_crop)])

    return data_dec, target_dec


def decode_zoom_multi(img, target, factor_max):
    data_multi = []
    target_multi = []
    for factor in range(1, factor_max + 1):
        decoded = decode_zoom(img, target, factor)
        data_multi.append(decoded[0])
        target_multi.append(decoded[1])

    return torch.cat(data_multi), torch.cat(target_multi)


def decode_fn(data, target, factor, decode_type, bound=128):
    if factor > 1:
        if decode_type == 'multi':
            data, target = decode_zoom_multi(data, target, factor)
        elif decode_type == 'single':
            data, target = decode_zoom(data, target, factor)

    return data, target


def decode(args, data, target):
    data_dec = []
    target_dec = []
    ipc = len(data) // args.nclass
    for c in range(args.nclass):
        idx_from = ipc * c
        idx_to = ipc * (c + 1)
        data_ = data[idx_from:idx_to].detach()
        target_ = target[idx_from:idx_to].detach()
        data_, target_ = decode_fn(data_,
                                   target_,
                                   args.factor,
                                   args.decode_type,
                                   bound=args.batch_syn_max)
        data_dec.append(data_)
        target_dec.append(target_)

    data_dec = torch.cat(data_dec)
    target_dec = torch.cat(target_dec)

    print("Dataset is decoded! ", data_dec.shape)
    return data_dec, target_dec

def get_index_by_class(ordered_indx, n_class, four_in_one=False, class_size=-1):
    if class_size == -1:
        class_size = len(ordered_indx) // n_class
        if four_in_one:
            class_size = class_size * len(ordered_indx[0])

    classes = [[] for _ in range(n_class)]

    for idx in ordered_indx:
        if four_in_one:
            idx_ = idx[0]
        else:
            idx_ = idx
        class_idx = idx_ // class_size
        if class_idx == 10: # last class
            class_idx -= 1
        classes[class_idx].append(idx)
    
    return classes

def get_idx_to_remove(args, idx_by_class, target_ipc, four_in_one=False, step=-1):
    '''
    return the `elements to remove` list
    '''
    assert step != -1 
    idx_to_remove_by_class = []
    factor = 1 if four_in_one else 4

    num_to_remove = int(target_ipc*factor*(4/step))
    for idx_class in idx_by_class:
        if args.reverse_selection:
            idx_to_remove_by_class += idx_class[-(len(idx_class)-num_to_remove):]
        else: # default: remove lowest scores
            idx_to_remove_by_class += idx_class[:len(idx_class)-num_to_remove]
    
    if four_in_one: # flatten the index list
        idx_to_remove_by_class = list(itertools.chain(*idx_to_remove_by_class))

    return idx_to_remove_by_class

def get_four_in_one_indices(args, ordered_index, ordered_count, step=-1):
    # zip them together
    combined = list(zip(ordered_index, ordered_count))
    combined_sorted = sorted(combined)
    
    # applay same sorting to index and count according to index number
    sorted_order_index = [elem[0] for elem in combined_sorted]
    sorted_order_count = [elem[1] for elem in combined_sorted]
    
    if step == -1:
        # default is using factor^2
        step = args.factor ** 2
        
    grouped_index = [sorted_order_index[i:i+step] for i in range(0, len(sorted_order_index), step)]
    grouped_count = [sum(sorted_order_count[i:i+step]) for i in range(0, len(sorted_order_count), step)]

    assert len(grouped_index) == len(grouped_count)
    assert len(grouped_index) == (len(ordered_index) // step)

    # re-ordering according to the grouped forgetting count
    combined = list(zip(grouped_count, grouped_index))
    combined_sorted = sorted(combined)
    
    ordered_grouped_index = [elem[1] for elem in combined_sorted]
    ordered_grouped_count = [elem[0] for elem in combined_sorted]

    return ordered_grouped_index, ordered_grouped_count

def extract_indices_by_range(score_index, ranges):
    '''
    This function returns the indices of score indices!
    '''
    range_indices = []
    range_values = []
    for r in ranges:
        start, end = r
        indices = [i for i, value in enumerate(score_index) if start <= value < end]
        values = [score_index[i].item() for i in indices]
        range_indices.append(indices)   # store the indices to apply for the score
        range_values.append(values)     # store the values to verify the range
    return range_indices, range_values

def sort_list_using_indices(list_, indices):
    return [list_[i].item() for i in indices]


def load_data_path(args, epoch=None, no_imagenet_aug=False):
    """Load condensed data from the given path
    """
    if args.pretrained:
        args.augment = False

    print()
    if args.dataset == 'imagenet':
        traindir = os.path.join(args.imagenet_dir, 'train')
        valdir = os.path.join(args.imagenet_dir, 'val')

        if no_imagenet_aug:
            args.augment = None
        train_transform, test_transform = transform_imagenet(augment=args.augment,
                                                             from_tensor=False,
                                                             size=args.size,
                                                             rrc=args.rrc)
        # Load condensed dataset
        if 'idc' in args.slct_type:
            if args.slct_type == 'idc': 
                data, target = torch.load(os.path.join(f'{args.save_dir}', 'data.pt'))

            elif args.slct_type == 'idc_cat':
                data_all = []
                target_all = []
                for idx in range(args.nclass // args.nclass_sub):
                    path = f'{args.save_dir}_{args.nclass_sub}_phase{idx}'
                    data, target = torch.load(os.path.join(path, 'data.pt'))
                    data_all.append(data)
                    target_all.append(target)
                    print(f"Load data from {path}")

                data = torch.cat(data_all)
                target = torch.cat(target_all)

            if args.factor > 1:
                data, target = decode(args, data, target)

            if args.random_selection:
                assert args.score_file == '', "Cannot use random selection and ccs score at the same time"

                if args.balance:
                    class_size = args.ipc_origin * args.factor ** 2
                    all_indices = range(class_size * args.nclass)
                    class_indices = [[] for _ in range(args.nclass)]

                    selected_indices = []
                    for i in range(args.nclass):
                        start = i * class_size
                        end = (i + 1) * class_size
                        class_indices[i] = all_indices[start:end]
                        selected_indices += random.sample(class_indices[i], args.ipc*args.factor**2)
                else:
                    selected_indices = random.sample(range(args.ipc_origin * args.factor ** 2 * args.nclass), args.ipc*args.factor**2 * args.nclass)

                print(selected_indices)
                data = data[selected_indices]
                target = target[selected_indices]

                # print class distribution
                class_idx = get_index_by_class(selected_indices, n_class=args.nclass, four_in_one=False, class_size=args.ipc_origin*args.factor**2)
                for i in range(args.nclass):
                    print('class', i, len(class_idx[i]))

            elif 'kmeans' in args.pruning_key:
                with open(args.score_file, 'rb') as f:
                    from ccs.core.data import CoresetSelection, IndexDataset, CIFARDataset, SVHNDataset, CINIC10Dataset
                    data_score = pickle.load(f)
                
                coreset_index = data_score[args.pruning_key]

                data = data[coreset_index]
                target = target[coreset_index]

                print(coreset_index)

                # print class distribution
                class_idx = get_index_by_class(coreset_index, n_class=args.nclass, four_in_one=False, class_size=args.ipc_origin*args.factor**2)
                for i in range(args.nclass):
                    print('class', i, len(class_idx[i]))
                    
                print(f"Select ipc{args.ipc} from ipc{args.ipc_origin}")
                print(f"Selected shape: {data.shape}")

            elif args.score_file != '':
                with open(args.score_file, 'rb') as f:
                    from ccs.core.data import CoresetSelection, IndexDataset, CIFARDataset, SVHNDataset, CINIC10Dataset
                    data_score = pickle.load(f)
                    data_score_cp = data_score.copy()
                    
                total_num = len(data_score[args.pruning_key])
                diff_prune_ratio = args.prune_score_ratio
                if not args.stratified_sampling:
                    diff_prune_ratio = 0.0
                mis_num = int(diff_prune_ratio * total_num)

                is_descending = True
                if args.pruning_key == 'accumulated_margin':
                    args.use_hard = not args.use_hard
                    if args.stratified_sampling:
                        is_descending = False   # used for ccs
                # DATA SCORE IS SORTED according to the score index
                data_score, score_index = CoresetSelection.mislabel_mask(data_score, mis_key=args.pruning_key, mis_num=mis_num, mis_descending=is_descending, pruning_key=args.pruning_key)

                if args.balance:
                    ipc_class_num = args.ipc_origin * args.factor ** 2
                    class_ranges = [(i, i + ipc_class_num) for i in range(0, total_num, ipc_class_num)]

                    # Extract indices for each range
                    range_indices, range_values = extract_indices_by_range(score_index, class_ranges)
                    class_sorted_indices = [sort_list_using_indices(score_index, indices) for indices in range_indices]
                    class_sorted_score = [sort_list_using_indices(data_score[args.pruning_key], indices) for indices in range_indices]

                    coreset_index_class = []
                    for i, (score, score_indices) in enumerate(zip(class_sorted_score, class_sorted_indices)):
                        total_num = ipc_class_num
                        corset_ratio = args.ipc / args.ipc_origin
                        coreset_num = int(corset_ratio * total_num)

                        if args.stratified_sampling:
                            coreset_index, _ = CoresetSelection.stratified_sampling(data_score=score, pruning_key=None, coreset_num=coreset_num, random_select=args.strata_fixed_select)
                            coreset_index = torch.tensor(score_indices)[coreset_index]

                            coreset_index = coreset_index.tolist()
                        else:
                            # use reverse metric
                            print("Using reverse metric: keep easiest samples")
                            print('data_score')
                            print(torch.tensor(sorted(score)))

                            if args.use_hard:
                                coreset_index = score_indices[:args.ipc * args.factor ** 2]
                            else:
                                coreset_index = score_indices[-args.ipc * args.factor ** 2:]

                        print('index', len(coreset_index), coreset_index)
                        coreset_index_class.append(coreset_index)
                        
                    flattened = torch.tensor(coreset_index_class).flatten()
                    coreset_index = flattened
                else:
                    corset_ratio = args.ipc / args.ipc_origin
                    coreset_num = int(corset_ratio * total_num)

                    if args.stratified_sampling:
                        coreset_index, _ = CoresetSelection.stratified_sampling(data_score=data_score, pruning_key=args.pruning_key, coreset_num=coreset_num, random_select=args.strata_fixed_select)
                        coreset_index = score_index[coreset_index]
                    else:
                        # use reverse metric
                        print("Using reverse metric: keep easiest samples")
                        if args.use_hard:
                            coreset_index = score_index[:coreset_num]
                        else:
                            coreset_index = score_index[-coreset_num:]

                data = data[coreset_index]
                target = target[coreset_index]

                print(coreset_index)
                print('forgetting counts\n', data_score_cp[args.pruning_key][coreset_index])

                # print class distribution
                class_idx = get_index_by_class(coreset_index, n_class=args.nclass, four_in_one=False, class_size=args.ipc_origin*args.factor**2)
                for i in range(args.nclass):
                    print('class', i, len(class_idx[i]))
                    
                print(f"Select ipc{args.ipc} from ipc{args.ipc_origin}")
                print(f"Selected shape: {data.shape}")

            print("Load condensed data ", data.shape, args.save_dir)

            if no_imagenet_aug:
                args.augment = None
            train_transform, _ = transform_imagenet(augment=args.augment,
                                                    from_tensor=True,
                                                    size=args.size,
                                                    rrc=args.rrc)
            train_dataset = TensorDataset(data, target, train_transform)
        else:
            train_dataset = ImageFolder(traindir,
                                        train_transform,
                                        nclass=args.nclass,
                                        seed=args.dseed,
                                        slct_type=args.slct_type,
                                        ipc=args.ipc,
                                        load_memory=args.load_memory)
            print(f"Test {args.dataset} random selection {args.ipc} (total {len(train_dataset)})")
        val_dataset = ImageFolder(valdir,
                                  test_transform,
                                  nclass=args.nclass,
                                  seed=args.dseed,
                                  load_memory=args.load_memory)

    else:
        if args.dataset[:5] == 'cifar':
            transform_fn = transform_cifar
        train_transform, test_transform = transform_fn(augment=args.augment, from_tensor=False)

            
        # Load condensed dataset
        if ('idc' == args.slct_type) or ('replace' in args.slct_type):
            if hasattr(args, 'custom_data_dir') and (args.custom_data_dir != ''):
                args.save_dir = args.custom_data_dir
            
            if args.load_mtt:
                args.factor = 1
                data = torch.load(os.path.join(f'{args.save_dir}', 'images_best.pt'))
                target = torch.load(os.path.join(f'{args.save_dir}', 'labels_best.pt'))
            elif args.load_dsa:
                args.factor = 1
                data_dict = torch.load(os.path.join(f'{args.save_dir}', 'data.pt'))
                data, target = data_dict['data'][0] # only use the first experiment
            elif args.load_kip:
                args.factor = 1
                data, target = torch.load(os.path.join(f'{args.save_dir}', 'data.pt'))
                data = data.permute(0, 3, 1, 2)
                target = torch.argmax(target, dim=1)
            else:
                data, target = torch.load(os.path.join(f'{args.save_dir}', 'data.pt'))
                if not isinstance(data, torch.FloatTensor):
                    data = data.float()
                print("Load condensed data ", args.save_dir, data.shape)

            # This does not make difference to the performance
            # data = torch.clamp(data, min=0., max=1.)
            if args.factor >= 1:

                if args.factor > 1:
                    data, target = decode(args, data, target)

                if args.random_selection:
                    assert args.score_file == '', "Cannot use random selection and ccs score at the same time"

                    if args.balance:
                        class_size = args.ipc_origin * args.factor ** 2
                        all_indices = range(class_size * args.nclass)
                        class_indices = [[] for _ in range(args.nclass)]

                        selected_indices = []
                        for i in range(args.nclass):
                            start = i * class_size
                            end = (i + 1) * class_size
                            class_indices[i] = all_indices[start:end]
                            selected_indices += random.sample(class_indices[i], args.ipc*args.factor**2)
                    else:
                        selected_indices = random.sample(range(args.ipc_origin * args.factor ** 2 * args.nclass), args.ipc*args.factor**2 * args.nclass)
                    
                    print(selected_indices)
                    data = data[selected_indices]
                    target = target[selected_indices]

                    # print class distribution
                    class_idx = get_index_by_class(selected_indices, n_class=args.nclass, four_in_one=False, class_size=args.ipc_origin*args.factor**2)
                    for i in range(args.nclass):
                        print('class', i, len(class_idx[i]))
                
                elif 'kmeans' in args.pruning_key:
                    with open(args.score_file, 'rb') as f:
                        from ccs.core.data import CoresetSelection, IndexDataset, CIFARDataset, SVHNDataset, CINIC10Dataset
                        data_score = pickle.load(f)
                    
                    coreset_index = data_score[args.pruning_key]

                    data = data[coreset_index]
                    target = target[coreset_index]

                    print(coreset_index)

                    # print class distribution
                    class_idx = get_index_by_class(coreset_index, n_class=args.nclass, four_in_one=False, class_size=args.ipc_origin*args.factor**2)
                    for i in range(args.nclass):
                        print('class', i, len(class_idx[i]))
                        
                    print(f"Select ipc{args.ipc} from ipc{args.ipc_origin}")
                    print(f"Selected shape: {data.shape}")

                elif args.score_file != '':
                    with open(args.score_file, 'rb') as f:
                        from ccs.core.data import CoresetSelection, IndexDataset, CIFARDataset, SVHNDataset, CINIC10Dataset
                        data_score = pickle.load(f)
                        data_score_cp = data_score.copy()
                        
                    total_num = len(data_score[args.pruning_key])
                    diff_prune_ratio = args.prune_score_ratio
                    if (not args.stratified_sampling) or args.ccs_type == 'ccs_2':
                        diff_prune_ratio = 0.0
                    mis_num = int(diff_prune_ratio * total_num)

                    is_descending = True
                    if args.pruning_key == 'accumulated_margin':
                        args.use_hard = not args.use_hard
                        if args.stratified_sampling:
                            is_descending = False   # used for ccs

                    # DATA SCORE IS SORTED according to the score index
                    data_score, score_index = CoresetSelection.mislabel_mask(data_score, mis_key=args.pruning_key, mis_num=mis_num, mis_descending=is_descending, pruning_key=args.pruning_key)


                    if args.balance:
                        ipc_class_num = args.ipc_origin * args.factor ** 2
                        class_ranges = [(i, i + ipc_class_num) for i in range(0, total_num, ipc_class_num)]

                        # Extract indices for each range
                        range_indices, range_values = extract_indices_by_range(score_index, class_ranges)
                        class_sorted_indices = [sort_list_using_indices(score_index, indices) for indices in range_indices]
                        class_sorted_score = [sort_list_using_indices(data_score[args.pruning_key], indices) for indices in range_indices]

                        coreset_index_class = []
                        for i, (score, score_indices) in enumerate(zip(class_sorted_score, class_sorted_indices)):
                            total_num = ipc_class_num
                            corset_ratio = args.ipc / args.ipc_origin
                            coreset_num = int(corset_ratio * total_num)

                            if args.stratified_sampling:
                                if args.ccs_type == 'ccs_2' and (args.prune_score_ratio > 0):
                                    score = score[:int(len(score) * (1 - args.prune_score_ratio))]
                                    score_indices = score_indices[:int(len(score_indices) * (1 - args.prune_score_ratio))]
                                coreset_index, _ = CoresetSelection.stratified_sampling(data_score=score, pruning_key=None, coreset_num=coreset_num, random_select=args.strata_fixed_select)
                                coreset_index = torch.tensor(score_indices)[coreset_index]

                                coreset_index = coreset_index.tolist()
                            else:
                                # use reverse metric
                                print("Using reverse metric: keep easiest samples")
                                print('data_score')
                                print(torch.tensor(sorted(score)))

                                if args.use_hard:
                                    coreset_index = score_indices[:args.ipc * args.factor ** 2]
                                else:
                                    coreset_index = score_indices[-args.ipc * args.factor ** 2:]
                                print()

                            print('index', len(coreset_index), coreset_index)
                            coreset_index_class.append(coreset_index)
                            
                        flattened = torch.tensor(coreset_index_class).flatten()
                        coreset_index = flattened

                    else:
                        corset_ratio = args.ipc / args.ipc_origin
                        coreset_num = int(corset_ratio * total_num)

                        if args.stratified_sampling:
                            coreset_index, _ = CoresetSelection.stratified_sampling(data_score=data_score, pruning_key=args.pruning_key, coreset_num=coreset_num, random_select=args.strata_fixed_select)
                            coreset_index = score_index[coreset_index]
                        else:
                            # use reverse metric
                            print("Using reverse metric: keep easiest samples")
                            if args.use_hard:
                                coreset_index = score_index[:coreset_num]
                            else:
                                coreset_index = score_index[-coreset_num:]

                    data = data[coreset_index]
                    target = target[coreset_index]

                    print(coreset_index)
                    print('forgetting counts\n', data_score_cp[args.pruning_key][coreset_index])

                    class_idx = get_index_by_class(coreset_index, n_class=args.nclass, four_in_one=False, class_size=args.ipc_origin*args.factor**2)
                    for i in range(args.nclass):
                        print('class', i, len(class_idx[i]))
                        
                    print(f"Select ipc{args.ipc} from ipc{args.ipc_origin}")
                    print(f"Selected shape: {data.shape}")

            train_transform, _ = transform_fn(augment=args.augment, from_tensor=True)
            train_dataset = TensorDataset(data, target, train_transform)
        
        else:    # original
            if args.dataset == 'cifar10':
                train_dataset = torchvision.datasets.CIFAR10(args.data_dir,
                                                             train=True,
                                                             transform=train_transform,
                                                             download=True)
            elif args.dataset == 'cifar100':
                train_dataset = torchvision.datasets.CIFAR100(args.data_dir,
                                                              train=True,
                                                              transform=train_transform,
                                                              download=True)
            indices = randomselect(train_dataset, args.ipc, nclass=args.nclass)
            train_dataset = Subset(train_dataset, indices)
            print(f"Random select {args.ipc} data (total {len(indices)})")

        # Test dataset
        if args.dataset == 'cifar10':
            val_dataset = torchvision.datasets.CIFAR10(args.data_dir,
                                                       train=False,
                                                       transform=test_transform,
                                                       download=True)
        elif args.dataset == 'cifar100':
            val_dataset = torchvision.datasets.CIFAR100(args.data_dir,
                                                        train=False,
                                                        transform=test_transform,
                                                        download=True)
    # For sanity check
    print("Training data shape: ", train_dataset[0][0].shape)

    return train_dataset, val_dataset


def define_model(args, nclass, logger=None, size=None):
    """Define neural network models
    """
    if size == None:
        size = args.size

    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset,
                          args.depth,
                          nclass,
                          norm_type=args.norm_type,
                          size=size,
                          nch=args.nch)
    elif args.net_type == 'resnet_ap':
        model = RNAP.ResNetAP(args.dataset,
                              args.depth,
                              nclass,
                              width=args.width,
                              norm_type=args.norm_type,
                              size=size,
                              nch=args.nch)
    elif args.net_type == 'efficient':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=nclass)
    elif args.net_type == 'densenet':
        model = DN.densenet_cifar(nclass)
    elif args.net_type == 'convnet':
        width = int(128 * args.width)
        model = CN.ConvNet(nclass,
                           net_norm=args.norm_type,
                           net_depth=args.depth,
                           net_width=width,
                           channel=args.nch,
                           im_size=(args.size, args.size))
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    if logger is not None:
        logger(f"=> creating model {args.net_type}-{args.depth}, norm: {args.norm_type}")
        # logger('# model parameters: {:.1f}M'.format(
        #     sum([p.data.nelement() for p in model.parameters()]) / 10**6))

    return model