import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from data import transform_imagenet, transform_cifar
from data import TensorDataset, ImageFolder, save_img, get_save_img
from data import ClassDataLoader, ClassMemDataLoader, MultiEpochsDataLoader
from data import MEANS, STDS
from train import define_model, train_epoch
from test import test_data, resnet10_bn
from utils.img_loader import return_data_path, load_data_path, load_ckpt
from misc.augment import DiffAug
from misc import utils
from math import ceil
import glob

from aim import Run, Image

class Synthesizer():
    """Condensed data class
    """
    def __init__(self, args, nclass, nchannel, hs, ws, ipc=-1, device='cuda'):
        if ipc == -1:
            self.ipc = args.ipc
        else:
            self.ipc = ipc
        self.nclass = nclass
        self.nchannel = nchannel
        self.size = (hs, ws)
        self.device = device
        self.grad = None

        self.data = torch.randn(size=(self.nclass * self.ipc, self.nchannel, hs, ws),
                                dtype=torch.float,
                                requires_grad=True,
                                device=self.device)
        self.data.data = torch.clamp(self.data.data / 4 + 0.5, min=0., max=1.)  # data value clipping
        self.targets = torch.tensor([np.ones(self.ipc) * i for i in range(nclass)],
                                    dtype=torch.long,
                                    requires_grad=False,
                                    device=self.device).view(-1)
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(self.data.shape[0]):
            self.cls_idx[self.targets[i]].append(i)

        print("\nDefine synthetic data: ", self.data.shape)

        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode='bilinear')
        print(f"Factor: {self.factor} ({self.decode_type})")

    def init(self, loader, init_type='noise'):
        """Condensed data initialization
        """
        if init_type == 'random':
            print("Random initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc)
                self.data.data[self.ipc * c:self.ipc * (c + 1)] = img.data.to(self.device)

        elif init_type == 'mix':
            print("Mixed initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc * self.factor**2)
                img = img.data.to(self.device)

                s = self.size[0] // self.factor
                remained = self.size[0] % self.factor
                k = 0
                n = self.ipc

                h_loc = 0
                for i in range(self.factor):
                    h_r = s + 1 if i < remained else s
                    w_loc = 0
                    for j in range(self.factor):
                        w_r = s + 1 if j < remained else s
                        img_part = F.interpolate(img[k * n:(k + 1) * n], size=(h_r, w_r))
                        self.data.data[n * c:n * (c + 1), :, h_loc:h_loc + h_r,
                                       w_loc:w_loc + w_r] = img_part
                        w_loc += w_r
                        k += 1
                    h_loc += h_r

        elif init_type == 'noise':
            pass

    def parameters(self):
        parameter_list = [self.data]
        return parameter_list

    def subsample(self, data, target, max_size=-1):
        if (data.shape[0] > max_size) and (max_size > 0):
            indices = np.random.permutation(data.shape[0])
            data = data[indices[:max_size]]
            target = target[indices[:max_size]]

        return data, target

    def decode_zoom(self, img, target, factor):
        """Uniform multi-formation
        """
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
        data_dec = self.resize(cropped)
        target_dec = torch.cat([target for _ in range(n_crop)])

        return data_dec, target_dec

    def decode_zoom_multi(self, img, target, factor_max):
        """Multi-scale multi-formation
        """
        data_multi = []
        target_multi = []
        for factor in range(1, factor_max + 1):
            decoded = self.decode_zoom(img, target, factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

        return torch.cat(data_multi), torch.cat(target_multi)

    def decode_zoom_bound(self, img, target, factor_max, bound=128):
        """Uniform multi-formation with bounded number of synthetic data
        """
        bound_cur = bound - len(img)
        budget = len(img)

        data_multi = []
        target_multi = []

        idx = 0
        decoded_total = 0
        for factor in range(factor_max, 0, -1):
            decode_size = factor**2
            if factor > 1:
                n = min(bound_cur // decode_size, budget)
            else:
                n = budget

            decoded = self.decode_zoom(img[idx:idx + n], target[idx:idx + n], factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

            idx += n
            budget -= n
            decoded_total += n * decode_size
            bound_cur = bound - decoded_total - budget

            if budget == 0:
                break

        data_multi = torch.cat(data_multi)
        target_multi = torch.cat(target_multi)
        return data_multi, target_multi

    def decode(self, data, target, bound=128):
        """Multi-formation
        """
        if self.factor > 1:
            if self.decode_type == 'multi':
                data, target = self.decode_zoom_multi(data, target, self.factor)
            elif self.decode_type == 'bound':
                data, target = self.decode_zoom_bound(data, target, self.factor, bound=bound)
            elif self.decode_type == 'single':
                data, target = self.decode_zoom(data, target, self.factor)
            # skip decoding if identity

        return data, target

    def sample(self, c, max_size=128):
        """Sample synthetic data per class
        """
        idx_from = self.ipc * c
        idx_to = self.ipc * (c + 1)
        data = self.data[idx_from:idx_to]
        target = self.targets[idx_from:idx_to]

        data, target = self.decode(data, target, bound=max_size)
        data, target = self.subsample(data, target, max_size=max_size)
        return data, target
    
    def loader(self, args, augment=True, ipcx=-1, indices=None):
        """Data loader for condensed data
        """
        if args.dataset == 'imagenet':
            train_transform, _ = transform_imagenet(augment=augment,
                                                    from_tensor=True,
                                                    size=0,
                                                    rrc=args.rrc,
                                                    rrc_size=self.size[0])
        elif args.dataset[:5] == 'cifar':
            train_transform, _ = transform_cifar(augment=augment, from_tensor=True)

        data_dec = []
        target_dec = []
        for c in range(self.nclass):
            idx_from = self.ipc * c
            if ipcx > 0:
                '''BUG FIXED:
                Since continues samples (i.e., 0-3) are not grouped into the same image
                decoding the first image gives different four images from selected indices
                '''
                idx_to = self.ipc * c + ipcx
            else:
                idx_to = self.ipc * (c + 1)
            
            data = self.data[idx_from:idx_to].detach()
            target = self.targets[idx_from:idx_to].detach()
            data, target = self.decode(data, target)

            if indices is not None: # use indices after decoding
                data = data[indices[c]]
                target = target[indices[c]]

            data_dec.append(data)
            target_dec.append(target)

        data_dec = torch.cat(data_dec)
        target_dec = torch.cat(target_dec)

        train_dataset = TensorDataset(data_dec.cpu(), target_dec.cpu(), train_transform)

        print("Decode condensed data: ", data_dec.shape)
        nw = 0 if not augment else args.workers
        train_loader = MultiEpochsDataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=nw,
                                             persistent_workers=nw > 0)
        return train_loader

    def test(self, args, val_loader, logger, ipcx=-1, aim_run=None, step=None, context=None, indices=None):
        """Condensed data evaluation
        """
        loader = self.loader(args, args.augment, ipcx=ipcx, indices=indices)
        model_list = test_data(args, loader, val_loader, logger=logger, aim_run=aim_run, step=step, context=context)

        if not args.no_resnet and not (args.dataset in ['mnist', 'fashion']):
            test_data(args, loader, val_loader, model_fn=resnet10_bn, logger=logger, aim_run=aim_run, step=step, context=context)
        
        return model_list

def encode_grad(args, grad, indices):
    '''
    input: grad.shape -> (40, 3, 16, 16)
    output: encoded_grad.shape -> (10, 3, 32, 32)
    '''
    s = args.size // args.factor
    remained = args.size % args.factor
    k = 0
    n = args.ipc
    h_loc = 0

    encoded_grad = torch.zeros((n, grad.shape[1], args.size, args.size))    # should return 10, 3, 32, 32

    ordered_grad = torch.zeros((n*args.factor**2, grad.shape[1], grad.shape[2], grad.shape[3]))

    for i in range(indices.shape[0]):
        idx = indices[i]    # index in original grad
        ordered_grad[idx] = grad[i]

    for i in range(args.factor):
        h_r = s + 1 if i < remained else s
        w_loc = 0
        for j in range(args.factor):
            w_r = s + 1 if j < remained else s
            # :n -> reshape to (n, , , ,)
            # : ->  retain all channels
            # h_loc:h_loc + h_r -> horizontal range
            # w_loc:w_loc + w_r -> vertical range
            encoded_grad[:n, :, h_loc:h_loc + h_r, w_loc:w_loc + w_r] = ordered_grad[n*k:n*(k+1)]
            w_loc += w_r
            k += 1
        h_loc += h_r

    return encoded_grad

def decode_grad(grad, factor):
    '''
    This function does not resize the gradient, but just returns the gradient
    e.g., if (n, 3, 32, 32) is the input, then (n*factor**2, 3, 16, 16) is the output
    '''
    h = grad.shape[-1]
    remained = h % factor
    if remained > 0:
        grad = F.pad(grad, pad=(0, factor - remained, 0, factor - remained), value=0.5)
    s_crop = ceil(h / factor)

    cropped = []
    for i in range(factor):
        for j in range(factor):
            h_loc = i * s_crop
            w_loc = j * s_crop
            cropped.append(grad[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
    cropped = torch.cat(cropped)

    return cropped


def load_resized_data(args):
    """Load original training data (fixed spatial size and without augmentation) for condensation
    """
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, transform=transforms.ToTensor(), download=True)
        normalize = transforms.Normalize(mean=MEANS['cifar10'], std=STDS['cifar10'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(args.data_dir,
                                          train=True,
                                          transform=transforms.ToTensor(),
                                          download=True)

        normalize = transforms.Normalize(mean=MEANS['cifar100'], std=STDS['cifar100'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR100(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 100

    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.imagenet_dir, 'train')
        valdir = os.path.join(args.imagenet_dir, 'val')

        # We preprocess images to the fixed size (default: 224)
        resize = transforms.Compose([
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.PILToTensor()
        ])

        if args.load_memory:  # uint8
            transform = None
            load_transform = resize
        else:
            transform = transforms.Compose([resize, transforms.ConvertImageDtype(torch.float)])
            load_transform = None

        _, test_transform = transform_imagenet(size=args.size)
        train_dataset = ImageFolder(traindir,
                                    transform=transform,
                                    nclass=args.nclass,
                                    phase=args.phase,
                                    seed=args.dseed,
                                    load_memory=args.load_memory,
                                    load_transform=load_transform)
        val_dataset = ImageFolder(valdir,
                                  test_transform,
                                  nclass=args.nclass,
                                  phase=args.phase,
                                  seed=args.dseed,
                                  load_memory=False)

    val_loader = MultiEpochsDataLoader(val_dataset,
                                       batch_size=args.batch_size // 2,
                                       shuffle=False,
                                       persistent_workers=True,
                                       num_workers=4)

    assert train_dataset[0][0].shape[-1] == val_dataset[0][0].shape[-1]  # width check

    return train_dataset, val_loader


def remove_aug(augtype, remove_aug):
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)


def diffaug(args, device='cuda'):
    """Differentiable augmentation for condensation
    """
    aug_type = args.aug_type
    normalize = utils.Normalize(mean=MEANS[args.dataset], std=STDS[args.dataset], device=device)
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True, seed=args.aug_seed if args.aug_seed > 0 else -1)   # fix seed for augmentation of images
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup_net == 'cut':
        aug_type = remove_aug(aug_type, 'cutout')
    print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand


def dist(x, y, method='mse'):
    """Distance objectives
    """
    if method == 'mse':
        dist_ = (x - y).pow(2).sum()
    elif method == 'l1':
        dist_ = (x - y).abs().sum()
    elif method == 'l1_mean':
        n_b = x.shape[0]
        dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()
    elif method == 'cos':
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        dist_ = torch.sum(1 - torch.sum(x * y, dim=-1) /
                          (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))

    return dist_


def add_loss(loss_sum, loss):
    if loss_sum == None:
        return loss
    else:
        return loss_sum + loss


def matchloss(args, img_real, img_syn, lab_real, lab_syn, model, c=None):
    """Matching losses (feature or gradient)
    """
    loss = None

    if args.match == 'feat':
        with torch.no_grad():
            feat_tg = model.get_feature(img_real, args.idx_from, args.idx_to)
        feat = model.get_feature(img_syn, args.idx_from, args.idx_to)

        for i in range(len(feat)):
            loss = add_loss(loss, dist(feat_tg[i].mean(0), feat[i].mean(0), method=args.metric))

    elif args.match == 'grad':
        criterion = nn.CrossEntropyLoss()

        # import pdb; pdb.set_trace()

        output_real = model(img_real)
        loss_real = criterion(output_real, lab_real)
        g_real = torch.autograd.grad(loss_real, model.parameters())

        # import pdb; pdb.set_trace()
        g_real = list((g.detach() for g in g_real))


        output_syn = model(img_syn)
        loss_syn = criterion(output_syn, lab_syn)

        g_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

        for i in range(len(g_real)):
            if (len(g_real[i].shape) == 1) and not args.bias:  # bias, normliazation
                continue
            if (len(g_real[i].shape) == 2) and not args.fc:
                continue

            loss = add_loss(loss, dist(g_real[i], g_syn[i], method=args.metric))    # simply add loss

    return loss


def pretrain_sample(args, model, verbose=False):
    """Load pretrained networks
    """
    folder_base = f'./pretrained/{args.datatag}/{args.modeltag}_cut'
    folder_list = glob.glob(f'{folder_base}*')
    tag = np.random.randint(len(folder_list))
    folder = folder_list[tag]

    epoch = args.pt_from
    if args.pt_num > 1:
        epoch = np.random.randint(args.pt_from, args.pt_from + args.pt_num)
    ckpt = f'checkpoint{epoch}.pth.tar'

    file_dir = os.path.join(folder, ckpt)
    load_ckpt(model, file_dir, verbose=verbose)


def condense(args, logger, device='cuda'):
    """Optimize condensed data
    """
    global run
    import time
    # Define real dataset and loader
    print(time.time(), 'Loading real data')
    trainset, val_loader = load_resized_data(args)

    print(time.time(), 'Creating loader')

    if args.load_memory:
        loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
    else:
        loader_real = ClassDataLoader(trainset,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)
    
    print(time.time(), 'Finished creating loader')

    nclass = trainset.nclass
    nch, hs, ws = trainset[0][0].shape

    # initialize model
    model = define_model(args, nclass).to(device)
    model.train()

    optim_net = optim.SGD(model.parameters(),
                            args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Define syn dataset
    synset = Synthesizer(args, nclass, nch, hs, ws)
    ipcx_index_class = None

    if args.load_checkpoint != '':
        print("======= LOAD CHECKPOINT SETTING ========")
        synset.data, _ = torch.load(os.path.join(args.load_checkpoint, 'data.pt'))
        synset.data = synset.data.cuda().requires_grad_(True)
        args.resume_epoch = torch.load(os.path.join(args.load_checkpoint, 'it.pt'))
        print("RESUME FROM ITERATION:", args.resume_epoch)

    else:
        synset.init(loader_real, init_type=args.init)

    step = args.resume_epoch
    aim_img = get_save_img(os.path.join(args.save_dir, 'init.png'),
                synset.data,
                unnormalize=False,
                dataname=args.dataset)

    run.track(Image(aim_img, 'init'), name='init.png', step=step)

    # Define augmentation function
    aug, aug_rand = diffaug(args)
    aim_img = get_save_img(os.path.join(args.save_dir, f'aug.png'),
             aug(synset.sample(0, max_size=args.batch_syn_max)[0]),
             unnormalize=True,
             dataname=args.dataset)

    run.track(Image(aim_img, 'aug'), name='aug.png', step=step)

    if not args.test:
        synset.test(args, val_loader, logger, aim_run=run, step=step)

    optim_img = torch.optim.SGD(synset.parameters(), lr=args.lr_img, momentum=args.mom_img)

    ts = utils.TimeStamp(args.time)
    n_iter = args.niter * 100 // args.inner_loop    # default=500 * 100 // 100
    it_log = n_iter // 200                           # log frequency = n_iter=500 // 50 = 10
    it_test = [n_iter // 10, n_iter // 5, n_iter // 2, n_iter]
                # evaluate frequency = n_iter=500 // 10 = 50, n_iter=500 // 5 = 100, n_iter=500 // 2 = 250, n_iter=50

    logger(f"\nStart condensing with {args.match} matching for {n_iter} iteration")
    args.fix_iter = max(1, args.fix_iter)

    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        
    for it in range(args.resume_epoch, n_iter): # n_iter (outer loop) default = 2000

        if args.fix_seed:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)

        if it % args.fix_iter == 0 and it != 0:
            model = define_model(args, nclass).to(device)
            model.train()

            optim_net = optim.SGD(model.parameters(),
                                  args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
            criterion = nn.CrossEntropyLoss()

        loss_total = 0
        loss_total_update = None
        synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)
        grads_img = []
        grads_net = []

        for ot in range(args.inner_loop):   # inner_loop default = 100
            ts.set()

            total_class_loss = 0
            # Update synset

            optim_img.zero_grad()

            for c in range(nclass):
                img, lab = loader_real.class_sample(c)   # c is class index
                
                img_syn, lab_syn = synset.sample(c, max_size=args.batch_syn_max)
                ts.stamp("data")

                n = img.shape[0]
                if not args.no_aug:
                    img_aug = aug(torch.cat([img, img_syn]))
                else:
                    img_aug = torch.cat([img, img_syn])

                ts.stamp("aug")

                optim_img.zero_grad()
                loss = matchloss(args, img_aug[:n], img_aug[n:], lab, lab_syn, model, c=c)

                loss_total += loss.item()
                loss_total_update = add_loss(loss_total_update, loss)
                total_class_loss += loss
                ts.stamp("loss")


                # optim_img actually clears the gradient of other class
                optim_img.zero_grad()
                loss.backward()

                optim_img.step()
                ts.stamp("backward")
            
            # Net update
            if args.n_data > 0:
                for _ in range(args.net_epoch):
                    train_epoch(args,
                                loader_real,
                                model,
                                criterion,
                                optim_net,
                                n_data=args.n_data,
                                aug=aug_rand,
                                mixup=args.mixup_net)
            ts.stamp("net update")

            if (ot + 1) % 10 == 0:
                ts.flush()
            
        # Logging
        # it_log = n_iter // 50 = 10
        if it % it_log == 0:
            logger(
                f"{utils.get_time()} (Iter {it:3d}) loss: {loss_total/nclass/args.inner_loop:.1f}")
            run.track(loss_total/nclass/args.inner_loop, name="loss", step=it, context={"subset": "train"})

        # Overwrite the save frequency
        it_test = [i*40 for i in range(args.niter//40+1)]

        if (it + 1) in it_test:

            aim_img = get_save_img(os.path.join(args.save_dir, f'img{it+1}.png'),
                     synset.data,
                     unnormalize=False,
                     dataname=args.dataset)

            run.track(Image(aim_img, f'img{it+1}.png'), name='syn image', step=it+1, context={"subset": "test"})

            # It is okay to clamp data to [0, 1] at here.
            # synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)
            torch.save(
                [synset.data.detach().cpu(), synset.targets.cpu()],
                os.path.join(args.save_dir, f'data.pt'))
            torch.save(it+1, os.path.join(args.save_dir, f'it.pt'))
            print("img and data saved!")

            if not args.test:
                synset.test(args, val_loader, logger, aim_run=run, step=it+1)


if __name__ == '__main__':
    import shutil
    from misc.utils import Logger
    from argument import args
    import torch.backends.cudnn as cudnn
    import json

    assert args.ipc > 0

    cudnn.benchmark = True
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    cur_file = os.path.join(os.getcwd(), __file__)
    shutil.copy(cur_file, args.save_dir)

    logger = Logger(args.save_dir)
    logger(f"Save dir: {args.save_dir}")
    with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    global run
    # initialize the run
    if args.load_checkpoint != '':
        run_hash = torch.load(os.path.join(args.load_checkpoint, 'run_hash.pt'))
        force_resume = True
    elif os.path.isfile(os.path.join(args.save_dir, 'run_hash.pt')):
        print("ENTERED")
        run_hash = torch.load(os.path.join(args.save_dir, 'run_hash.pt'))
        force_resume = True
        args.load_checkpoint = args.save_dir
    else:
        run_hash = None
        force_resume = False

    run = Run(experiment='condense', run_hash=run_hash, force_resume=force_resume)
    run.name = f"{args.dataset}_ipc{args.ipc}"
    torch.save(run.hash, os.path.join(args.save_dir, 'run_hash.pt'))

    hyperparams = dict()
    for key, value in vars(args).items():
        hyperparams.update({key: value})
    run["hparams"] = hyperparams

    condense(args, logger)
    
    from misc.aim_export import aim_log
    if run:
        dir_name = args.save_dir
        aim_log(run, dir_name, args)