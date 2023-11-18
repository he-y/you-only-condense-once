import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import os, sys
import argparse
import pickle
from datetime import datetime

from torchvision import models

from ccs.core.model_generator import convnet
import models.resnet_ap as RNAP
from ccs.core.training import Trainer, TrainingDynamicsLogger
from ccs.core.data import IndexDataset, CIFARDataset
from ccs.core.utils import print_training_info, StdRedirect

from utils.img_loader import load_data_path
from misc.augment import DiffAug
from data import MultiIndexEpochsDataLoader

from aim import Run

model_names = ['resnet18', 'wrn-34-10', 'preact_resnet18']

parser = argparse.ArgumentParser(description='PyTorch CIFAR10,CIFAR100 Training')

######################### Training Setting #########################
parser.add_argument('--epochs', type=int, default="1000", metavar='N',
                    help='The number of epochs to train a model.')
parser.add_argument('--iterations', type=int, metavar='N',
                    help='The number of iteration to train a model; conflict with --epoch.')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--network', type=str, default='resnet18', choices=['convnet', 'resnet10_ap'])
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--condense_key', type=str, default='idc', choices=['idc', 'dream', 'dsa', 'kip', 'mtt'], help=['type of condensed dataset'])

######################### Print Setting #########################
parser.add_argument('--iterations-per-testing', type=int, default=800, metavar='N',
                    help='The number of iterations for testing model')
parser.add_argument('--test-period', type=int, default=None,
                    help='period of testing model')

######################### Path Setting #########################
parser.add_argument('--data-dir', type=str, default='~/scratch',
                    help='The dir path of the data.')
parser.add_argument('--imagenet-dir', type=str, default='/mnt/data2/usertwo/dataset/Imagenet',
                    help='The dir path of the imagenet.')
parser.add_argument('--base-dir', type=str, default='raid/dynamics_and_scores/cifar10',
                    help='The base dir of this project.')
parser.add_argument('--task-name', type=str, default='tmp',
                    help='The name of the training task.')

######################### CUSTOM DATADIR #########################
parser.add_argument('--custom-data-dir', type=str, default=None,
                     help='The dir path of the data.')
parser.add_argument('--score_file', type=str, default='', help='directory of ccs scores (sorted index in pickle file)')

######################### GPU Setting #########################
parser.add_argument('--gpuid', type=str, default='0',
                    help='The ID of GPU.')

########################### DEPENDENCY ###########################
parser.add_argument('--ipc', type=int, help='images per class')
parser.add_argument('--factor', type=int, default='2', help='factor of multi-formation')

parser.add_argument('--load_mtt', action='store_true', help='whether to load dsa model')
parser.add_argument('--load_dsa', action='store_true', help='whether to load dsa model')
parser.add_argument('--load_kip', action='store_true', help='whether to load kip model')

parser.add_argument('--dsa', type=str, default=True, help='whether to use dsa strategy')
parser.add_argument('--dsa_strategy', type=str, default='color_crop_flip_scale_rotate')
parser.add_argument('--mixup', type=str, default=True, help='whether to use mixup')

parser.add_argument('--random_selection', action='store_true', help='whether to select random samples') # can ignore

parser.add_argument('--reproduce_exp', type=str, default='reproduce_3', help='reproduce mark of experiment')

######################### AIM Setting #########################
parser.add_argument('--run_tag', type=str, default='', help='tag of the run')
parser.add_argument('--reproduce_exp', type=str, default='', help='reproduce mark of experiment')

args = parser.parse_args()

if args.dataset in ['cifar10', 'cifar100']:
    args.factor = 2
    args.network = 'convnet'
    args.epoch = 1000
    args.lr = 0.01
    args.batch_size = 64
    args.mixup = True
    args.dsa = True
    args.test_period = 10
    args.task_name = f"ipc{args.ipc}"
    args.base_dir=f"raid/{args.reproduce_exp}/dynamics_and_scores/{args.condense_key}/{args.dataset}"
    if args.condense_key == 'idc':
        args.custom_data_dir = f"raid/condensed_img/idc/{args.dataset}/conv3in_grad_mse_nd2000_cut_niter2000_factor2_lr0.005_mix_ipc{args.ipc}"
    elif args.condense_key == 'dream':
        args.custom_data_dir = f"raid/condensed_img/dream/{args.dataset}/ipc{args.ipc}"
    elif (args.dataset != 'cifar10') or (args.ipc != 10):
        raise NotImplementedError
    elif args.condense_key == 'mtt':
        args.factor = 1
        args.load_mtt = True
        args.custom_data_dir = f"raid/condensed_img/mtt/ConvNet_baseline"
    elif args.condense_key == 'dsa':
        args.factor = 1
        args.load_dsa = True
        args.custom_data_dir = f"raid/condensed_img/dsa/res_DSA_CIFAR10_ConvNet_ipc10"
    elif args.condense_key == 'kip':
        args.factor = 1
        args.load_kip = True
        args.custom_data_dir = f"raid/condensed_img/kip/kip_ipc10"
    else:
        raise NotImplementedError
elif 'imagenet' in args.dataset:
    args.factor = 3
    args.network = 'resnet10_ap'
    args.epoch = 1000
    args.lr = 0.01
    args.batch_size = 64
    args.mixup = True
    args.dsa = True
    args.test_period = 10
    args.task_name = f"ipc{args.ipc}"
    args.base_dir=f"raid/{args.reproduce_exp}/dynamics_and_scores/idc/imagenet10"
    args.custom_data_dir=f"raid/condensed_img/idc/imagenet10/resnet10apin_grad_l1_ely10_nd500_cut_factor3_mix_ipc{args.ipc}"
else:
    raise NotImplementedError

start_time = datetime.now()

assert args.epochs is None or args.iterations is None, "Both epochs and iterations are used!"

run = Run(experiment='generate-score')
run.name = args.task_name
run.add_tag(args.run_tag)

print(f'Dataset: {args.dataset}')
######################### Set path variable #########################
task_dir = os.path.join(args.base_dir, args.task_name)
os.makedirs(task_dir, exist_ok=True)
last_ckpt_path = os.path.join(task_dir, f'ckpt-last.pt')
best_ckpt_path = os.path.join(task_dir, f'ckpt-best.pt')
td_path = os.path.join(task_dir, f'td-{args.task_name}.pickle')
log_path = os.path.join(task_dir, f'log-train-{args.task_name}.log')

######################### Print setting #########################
sys.stdout=StdRedirect(log_path)
print_training_info(args, all=True)
#########################
print(f'Last ckpt path: {last_ckpt_path}')

# GPUID = args.gpuid
# os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.custom_data_dir is not None:
    print("Loading custom data dir...")
    if 'cifar' in args.dataset:
        if args.load_mtt:
            ipc = 10
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
        
        args.custom_data_dir = args.custom_data_dir
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
        args.custom_data_dir = args.custom_data_dir
        args.save_dir = args.custom_data_dir

        trainset, testset = load_data_path(args)

else:
    if args.dataset == 'cifar10':
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(args.data_dir, args.dataset)
    print(f'Data dir: {data_dir}')

    if args.dataset == 'cifar10':
        trainset = CIFARDataset.get_cifar10_train(data_dir)
    elif args.dataset == 'cifar100':
        trainset = CIFARDataset.get_cifar100_train(data_dir)
    elif args.dataset == 'imagenet':
        if hasattr(args, 'save_dir'):   # load condensed dataset
            args.slct_type = 'idc'
            args.ipc = int(args.save_dir.split('ipc')[-1])
        else:
            args.slct_type = '' # load full imagenet10
            args.ipc = -1

        args.nclass = 10
        args.slct_ipc = 0   # dont select
        args.dataset = 'imagenet'
        args.imagenet_dir = '/raid/data_lingao/datasets/imagenet'
        args.pretrained = False
        args.augment = True
        args.aug_type = "color_crop_cutout"
        args.factor = 3
        args.nch = 3
        args.decode_type = "single"
        args.batch_syn_max = 128
        args.rrc = True
        args.pruning_indices = ''
        args.size = 224
        args.dseed = 0
        args.seed = 0
        args.load_memory = True
        args.phase = -1

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

        from data import transform_imagenet, ImageFolder
        _, test_transform = transform_imagenet(size=args.size)
        trainset = ImageFolder(traindir,
                                    transform=transform,
                                    nclass=args.nclass,
                                    phase=args.phase,
                                    seed=args.dseed,
                                    load_memory=args.load_memory,
                                    load_transform=load_transform)
        testset = ImageFolder(valdir,
                                  test_transform,
                                  nclass=args.nclass,
                                  phase=args.phase,
                                  seed=args.dseed,
                                  load_memory=False)

trainset = IndexDataset(trainset)
print(len(trainset))

if args.custom_data_dir is None:
    if args.dataset == 'cifar10':
        testset = CIFARDataset.get_cifar10_test(data_dir)
    elif args.dataset == 'cifar100':
        testset = CIFARDataset.get_cifar100_test(data_dir)

print(len(testset))


if 'imagenet' in args.dataset :
    trainloader = MultiIndexEpochsDataLoader(dataset=trainset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=8,
                                            persistent_workers=True)
    testloader = MultiIndexEpochsDataLoader(dataset=testset,
                                            is_test_set=True,
                                            batch_size=args.batch_size // 2,
                                            shuffle=False,
                                            persistent_workers=True,
                                            num_workers=4)
else:
    trainloader = MultiIndexEpochsDataLoader(dataset=trainset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=8,
                                            persistent_workers=True)
    testloader = MultiIndexEpochsDataLoader(dataset=testset,
                                            is_test_set=True,
                                            batch_size=512,
                                            shuffle=False,
                                            persistent_workers=True,
                                            num_workers=4)

iterations_per_epoch = len(trainloader)
if args.iterations is None:
    num_of_iterations = iterations_per_epoch * args.epochs
else:
    num_of_iterations = args.iterations

if args.dataset in ['cifar10', 'imagenet']:
    num_classes=10
else:
    num_classes=100

if args.network == 'convnet':
    print('convet-D3')
    model = convnet(num_classes)
elif args.network == 'resnet10_ap':
    print('resnet10_ap')
    model = RNAP.ResNetAP(args.dataset, depth=10, num_classes=args.nclass, norm_type='instance', nch=args.nch)

model = model.to(device)

if (args.network == 'convnet') or (args.network == 'resnet10_ap'):
    print("Using IDC training setting")
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(),
                            args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * num_of_iterations // 3, 5 * num_of_iterations // 6], gamma=0.2)

else:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_of_iterations, eta_min=1e-4)


if args.test_period is None:
    epoch_per_testing = args.iterations_per_testing // iterations_per_epoch
else:
    epoch_per_testing = args.test_period

print(f'Total epoch: {num_of_iterations // iterations_per_epoch}')
print(f'Iterations per epoch: {iterations_per_epoch}')
print(f'Total iterations: {num_of_iterations}')
print(f'Epochs per testing: {epoch_per_testing}')

trainer = Trainer(aim_run=run, mixup=args.mixup, nclass=args.nclass)
TD_logger = TrainingDynamicsLogger()

if args.dsa:
    aug = DiffAug(strategy=args.dsa_strategy, batch=False)
    print(f'Performing DSA {args.dsa_strategy} augmentation')
else:
    aug = None

best_acc = 0
best_epoch = -1

current_epoch = 0
while num_of_iterations > 0:
    iterations_epoch = min(num_of_iterations, iterations_per_epoch)
    trainer.train(current_epoch, -1, model, trainloader, optimizer, criterion, scheduler, device, aug=aug, TD_logger=TD_logger, log_interval=60, printlog=True)

    num_of_iterations -= iterations_per_epoch

    if current_epoch % epoch_per_testing == 0 or num_of_iterations == 0:
        test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=20,  printlog=True, epoch=current_epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = current_epoch
            state = {
                'model_state_dict': model.state_dict(),
                'epoch': best_epoch
            }
            torch.save(state, best_ckpt_path)

    current_epoch += 1

# last ckpt testing
if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = current_epoch
            state = {
                'model_state_dict': model.state_dict(),
                'epoch': best_epoch
            }
            torch.save(state, best_ckpt_path)
print('==========================')
print(f'Best acc: {best_acc * 100:.2f}')
print(f'Best acc: {best_acc}')
print(f'Best epoch: {best_epoch}')
print(best_acc)
######################### Save #########################
state = {
    'model_state_dict': model.state_dict(),
    'epoch': current_epoch - 1
}
torch.save(state, last_ckpt_path)
TD_logger.save_training_dynamics(td_path, data_name=args.dataset)

print(f'Total time consumed: {(datetime.now() - start_time).total_seconds():.2f}')