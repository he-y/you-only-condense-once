import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torch.nn.utils.prune as prune
from train import train
from data import TensorDataset, ImageFolder, MultiEpochsDataLoader, get_save_img
from torch.utils.data import Subset
from coreset import randomselect, herding
from efficientnet_pytorch import EfficientNet
import models.resnet as RN
import models.resnet_ap as RNAP
import models.densenet_cifar as DN

import time

from utils.img_loader import return_data_path, load_data_path, define_model

import matplotlib.pyplot as plt
    
def test_data(args,
              train_loader,
              val_loader,
              model_fn=define_model,
              repeat=1,
              logger=print,
              num_val=4,
              aim_run=None,
              step=None,
              context=None):
    """Train neural networks on condensed data
    """

    args.epoch_print_freq = args.epochs // num_val

    existing_fns = [None, define_model,  resnet10_bn, resnet10_in, densenet]
    model_fn_ls = [model_fn]

    model_list = []
    for model_fn_ in model_fn_ls:
        best_acc_l = []
        acc_l = []
        list_of_result = []
        for _ in range(repeat):
            if model_fn in existing_fns:
                model = model_fn_(args, args.nclass, logger=logger)
            else:   # custom model or pruned model
                # import pdb; pdb.set_trace()
                model = model_fn_
            # breakpoint()
            best_acc, acc = train(args, model, train_loader, val_loader, logger=print, aim_run=aim_run, model_name=model_fn_.__name__)
            best_acc_l.append(best_acc)
            acc_l.append(acc)
            list_of_result.append((model.state_dict(), best_acc, acc))
        logger(
            f'Repeat {repeat} => Best, last acc: {np.mean(best_acc_l):.2f} {np.mean(acc_l):.2f}')
        # log standard deviation
        logger(
            f'Repeat {repeat} => Best, last std: {np.std(best_acc_l):.2f} {np.std(acc_l):.2f}\n')

        if args.eval:
            # save evaluation model
            torch.save(list_of_result, f'{args.log_dir}/{run.name}_{args.cur_time}.pt')

        if aim_run:
            if context is None:
                aim_run.track(np.mean(best_acc_l), name="best acc", step=step, context={"subset": model_fn_.__name__})
                aim_run.track(np.mean(acc_l), name="last acc", step=step, context={"subset": model_fn_.__name__})
                aim_run.track(np.std(best_acc_l), name="std best", step=step, context={"subset": model_fn_.__name__})
                aim_run.track(np.std(acc_l), name="std last", step=step, context={"subset": model_fn_.__name__})
            else:
                aim_run.track(np.mean(best_acc_l), name="best acc", step=step, context={"subset": context})
                aim_run.track(np.mean(acc_l), name="last acc", step=step, context={"subset": context})
                aim_run.track(np.std(best_acc_l), name="std best", step=step, context={"subset": context})
                aim_run.track(np.std(acc_l), name="std last", step=step, context={"subset": context})
        
        model_list.append(model)
    return model_list

def resnet10_in(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 10, nclass, 'instance', args.size, nch=args.nch)
    if logger is not None:
        logger(f"=> creating model resnet-10, norm: instance")
    return model

def resnet10_bn(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 10, nclass, 'batch', args.size, nch=args.nch)
    if logger is not None:
        logger(f"=> creating model resnet-10, norm: batch")
    return model

def resnet18_bn(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 18, nclass, 'batch', args.size, nch=args.nch)
    if logger is not None:
        logger(f"=> creating model resnet-18, norm: batch")
    return model

def densenet(args, nclass, logger=None):
    if 'cifar' == args.dataset[:5]:
        model = DN.densenet_cifar(nclass)
    else:
        raise AssertionError("Not implemented!")

    if logger is not None:
        logger(f"=> creating DenseNet")
    return model


def efficientnet(args, nclass, logger=None):
    if args.dataset == 'imagenet':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=nclass)
    else:
        raise AssertionError("Not implemented!")

    if logger is not None:
        logger(f"=> creating EfficientNet")
    return model


if __name__ == '__main__':
    from argument import args
    import torch.backends.cudnn as cudnn
    import numpy as np
    cudnn.benchmark = True

    from aim import Run, Image
    # set to eval mode
    args.eval = True
    # create a log file
    args.cur_time = time.strftime("%Y%m%d-%H%M%S")

    global run
    if args.exp_name != '':
        run = Run(experiment=args.exp_name)
    elif args.dataset == 'cifar10':
        run = Run(experiment='YOCO-cifar10-reproduce')
    elif args.dataset == 'cifar100':
        run = Run(experiment='YOCO-cifar100-reproduce')
    elif args.dataset == 'imagenet':
        run = Run(experiment='YOCO-imagenet-reproduce')
    else:
        run = Run(experiment='idc-eval-selection')
            
    run.name = args.run_name
    run.add_tag(args.run_tag)

    if args.same_compute and args.factor > 1:
        args.epochs = int(args.epochs / args.factor**2)

    if args.custom_data_dir == '':
        path_list = return_data_path(args)
    else:
        args.save_dir = args.custom_data_dir
        path_list = [args.custom_data_dir]

    args.log_dir = args.save_dir.replace('condensed_img', f'log_{args.reproduce_exp}')
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    for p in path_list:
        train_dataset, val_dataset = load_data_path(args)

        train_loader = MultiEpochsDataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.workers if args.augment else 0,
                                             persistent_workers=args.augment > 0)
        val_loader = MultiEpochsDataLoader(val_dataset,
                                           batch_size=args.batch_size // 2,
                                           shuffle=False,
                                           persistent_workers=True,
                                           num_workers=4)

        test_data(args, train_loader, val_loader, repeat=args.repeat, aim_run=run)
        if args.dataset[:5] == 'cifar':
            if not args.no_resnet:
                test_data(args, train_loader, val_loader, repeat=args.repeat, model_fn=resnet10_bn, aim_run=run)
                if (not args.same_compute) and (args.ipc >= 50 and args.factor > 1):
                    args.epochs = 400
                test_data(args, train_loader, val_loader, repeat=args.repeat, model_fn=densenet, aim_run=run)
        elif args.dataset == 'imagenet':
            if not args.no_resnet:
                test_data(args, train_loader, val_loader, repeat=args.repeat, model_fn=resnet18_bn)
                test_data(args, train_loader, val_loader, repeat=args.repeat, model_fn=efficientnet)

    from misc.aim_export import aim_log
    if run:
        aim_log(run, args.log_dir, args)