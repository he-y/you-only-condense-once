import torch
import os

from data import save_img, transform_imagenet, transform_cifar, transform_svhn, transform_mnist, transform_fashion, get_save_img
from data import TensorDataset, ImageFolder

import torchvision

def prediction_correct(true, preds):
    """
    Computes prediction_hit.
    Arguments:
        true (torch.Tensor): true labels.
        preds (torch.Tensor): predicted labels.
    Returns:
        Prediction_hit for each img.
    """
    rst = (torch.softmax(preds, dim=1).argmax(dim=1) == true)
    return rst.detach().cpu().type(torch.int)

def get_model_directory(base_dir, model_name):
    model_dir = os.join(base_dir, model_name)
    ckpt_dir = os.join(model_dir, 'ckpt')
    data_dir = os.join(model_dir, 'data')
    log_dir = os.join(model_dir, 'log')

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    return ckpt_dir, data_dir, log_dir

def l2_distance(tensor1, tensor2):
    dist = (tensor1 - tensor2).pow(2).sum().sqrt()
    return dist

def load_data_path(args, epoch=None):
    """Load condensed data from the given path
    """
    if args.dataset[:5] == 'cifar':
        transform_fn = transform_cifar
    train_transform, test_transform = transform_fn(augment=args.augment, from_tensor=False)

    # Load condensed dataset
    if hasattr(args, 'custom_data_dir') and (args.custom_data_dir != ''):
        args.save_dir = args.custom_data_dir
    data, target = torch.load(os.path.join(f'{args.save_dir}', 'data.pt'))
    print("Load condensed data ", args.save_dir, data.shape)

    # This does not make difference to the performance
    # data = torch.clamp(data, min=0., max=1.)
    if args.factor > 1:
        data, target = decode(args, data, target)
        # breakpoint()

    train_transform, _ = transform_fn(augment=args.augment, from_tensor=True)
    train_dataset = TensorDataset(data, target, train_transform)
    
    # Test dataset
    val_dataset = torchvision.datasets.CIFAR10(args.data_dir,
                                                train=False,
                                                transform=test_transform)

    # For sanity check
    print("Training data shape: ", train_dataset[0][0].shape)
    # os.makedirs('./raid/condensed_img', exist_ok=True)
    # save_img('./raid/condensed_img/test.png',
    #          torch.stack([d[0] for d in train_dataset]),
    #          dataname=args.dataset)
    # print()

    return train_dataset, val_dataset