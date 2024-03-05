import logging
import os
import random
from typing import Union
import torch
import torch.utils.data
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from thop import profile
from math import ceil, sqrt
from torch.cuda import amp
import torch.distributed
import argparse
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import colors

from models import cifar10dvs, sew_resnet, cifar10
from models.submodules.sparse import ConvBlock, Mask
from sparsity.penalty_term import PenaltyTerm
from sparsity.temp_scheduler import SplitTemperatureScheduler, TemperatureScheduler
#from models import spiking_resnet, sew_resnet
from utils import RecordDict, GlobalTimer, Timer
from utils import DatasetSplitter, CriterionWarpper, DVStransform, SOPMonitor, CIFAR10Policy, Cutout, Augment, DatasetWarpper
from utils import left_neurons, left_weights, init_mask, set_pruning_mode
from utils import is_main_process, save_on_master, search_tb_record, finetune_tb_record, accuracy, safe_makedirs
from spikingjelly.clock_driven import functional


def parse_args():
    parser = argparse.ArgumentParser(description='Training')

    # training options
    parser.add_argument('--seed', default=12450, type=int)
    parser.add_argument('--epoch-search', default=800, type=int)
    parser.add_argument('--epoch-finetune', default=200, type=int,
                        help='when to fine tune, -1 means will not fine tune')
    parser.add_argument('--not-prune-weight', action='store_true')
    parser.add_argument('--not-prune-neuron', action='store_true')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--T', default=8, type=int, help='simulation steps')
    parser.add_argument('--model', default='Cifar10Net', help='model type')
    parser.add_argument('--dataset', default='CIFAR10', help='dataset type')
    parser.add_argument('--augment', action='store_true', help='Additional augment')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--search-lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--finetune-lr', default=1e-4, type=float, help='finetune learning rate')
    parser.add_argument('--prune-lr', type=float, help='initial learning rate of pruning')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--prune-optimizer', type=str, help='Adam or SGD')
    parser.add_argument('--weight-decay', default=0, type=float, help='weight decay (default: 0)')
    parser.add_argument('--prune-weight-decay', default=0, type=float)
    parser.add_argument('--criterion', type=str, default='MSE', help='MSE or CE')
    parser.add_argument(
        '--search-lr-scheduler', type=str, nargs='+', default=[],
        help='''--lr-scheduler Cosine [<T0> <Tt> <Tmax(period of cosine)>]
            or --lr-scheduler Step [minestones]...''')
    parser.add_argument(
        '--finetune-lr-scheduler', type=str, nargs='+', default=[],
        help='''--lr-scheduler Cosine [<T0> <Tt> <Tmax(period of cosine)>]
            or --lr-scheduler Step [minestones]...''')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='Number of times a debug message is printed in one epoch')
    parser.add_argument('--tb-interval', type=int, default=10)
    parser.add_argument('--data-path', default='./datasets', help='dataset')
    parser.add_argument('--output-dir', default='./logs/temp')
    parser.add_argument('--resume', type=str, help='resume from checkpoint')
    parser.add_argument('--resume-type', type=str, default='test', help='search, finetune or test')
    parser.add_argument('--distributed-init-mode', type=str, default='env://')

    # mask init
    parser.add_argument(
        '--mask-init-factor', type=float, nargs='+', default=[0, 0, 0, 0],
        help='--mask-init-factor <weights mean> <neurons mean> <weights std> <neurons std>')

    # penalty term
    parser.add_argument('--penalty-lmbda', type=float, default=1e-11)

    parser.add_argument(
        '--temp-scheduler', type=float, nargs='+', default=[5, 1000],
        help='''--temp-scheduler <init temp> <final temp>
                or --temp-scheduler <init temp> <final temp> <T0> <Tmax>
                or --temp-scheduler <init temp of weight> <init temp of neuron> 
                <final temp of weight> <final temp of neuron> <T0> <Tmax>''')
    # deprecated
    parser.add_argument('--accumulate-step', type=int, default=1)

    # argument of sew resnet
    parser.add_argument('--zero-init-residual', action='store_true',
                        help='zero init all residual blocks')
    parser.add_argument(
        "--cache-dataset", action="store_true",
        help="Cache the datasets for quicker initialization. It also serializes the transforms")
    parser.add_argument("--sync-bn", action="store_true", help="Use sync batch norm")
    parser.add_argument("--test-only", action="store_true", help="Only test the model")
    parser.add_argument('--amp', action='store_true', help='Use AMP training')

    # argument of TET
    parser.add_argument('--TET', action='store_true', help='Use TET training')
    parser.add_argument('--TET-phi', type=float, default=1.0)
    parser.add_argument('--TET-lambda', type=float, default=0.0)

    parser.add_argument('--save-latest', action='store_true')

    args = parser.parse_args()
    return args


def setup_logger(output_dir):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]%(message)s',
                                  datefmt=r'%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    return logger


def init_distributed(logger: logging.Logger, distributed_init_mode):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        logger.info('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    backend = 'nccl'
    logger.info('Distributed init rank {}'.format(rank))
    torch.distributed.init_process_group(backend=backend, init_method=distributed_init_mode,
                                         world_size=world_size, rank=rank)
    # only master process logs
    if rank != 0:
        logger.setLevel(logging.WARNING)
    return True, rank, world_size, local_rank


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(dataset_dir, cache_dataset, dataset_type, distributed: bool, augment: bool,
              logger: logging.Logger, T: int):

    if dataset_type == 'CIFAR10':
        if augment:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                Cutout(n_holes=1, length=16), ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

        dataset = torchvision.datasets.CIFAR10(root=os.path.join(dataset_dir), train=True,
                                               download=True)
        dataset_test = torchvision.datasets.CIFAR10(root=os.path.join(dataset_dir), train=False,
                                                    download=True)
    elif dataset_type == 'CIFAR100':
        if augment:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                     std=[n / 255. for n in [68.2, 65.4, 70.4]]),
                Cutout(n_holes=1, length=8), ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                     std=[n / 255. for n in [68.2, 65.4, 70.4]]), ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                 std=[n / 255. for n in [68.2, 65.4, 70.4]]), ])

        dataset = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=True,
                                                download=True)
        dataset_test = torchvision.datasets.CIFAR100(root=os.path.join(dataset_dir), train=False,
                                                     download=True)
    elif dataset_type == 'CIFAR10DVS':
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        if augment:
            transform_train = DVStransform(transform=transforms.Compose([
                transforms.Resize(size=(48, 48), antialias=True),
                Augment()]))
        else:
            transform_train = DVStransform(
                transform=transforms.Compose([transforms.Resize(size=(48, 48), antialias=True)]))
        transform_test = DVStransform(transform=transforms.Resize(size=(48, 48), antialias=True))

        dataset = CIFAR10DVS(dataset_dir, data_type='frame', frames_number=T, split_by='number')
        dataset, dataset_test = DatasetSplitter(dataset, 0.9,
                                                True), DatasetSplitter(dataset, 0.1, False)
    elif dataset_type == 'ImageNet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        logger.info('Loading training data')
        traindir = os.path.join(dataset_dir, 'train')
        valdir = os.path.join(dataset_dir, 'val')
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize, ])
        transform_test = transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize, ])
        with Timer('Load training data', logger):
            cache_path = _get_cache_path(traindir)
            if cache_dataset and os.path.exists(cache_path):
                # Attention, as the transforms are also cached!
                dataset, _ = torch.load(cache_path)
                logger.info("Loaded training dataset from {}".format(cache_path))
            else:
                dataset = torchvision.datasets.ImageFolder(traindir)
                if cache_dataset:
                    safe_makedirs(os.path.dirname(cache_path))
                    save_on_master((dataset, traindir), cache_path)
                    logger.info("Cached training dataset to {}".format(cache_path))
                logger.info("Loaded training dataset")

        logger.info("Loading validation data")
        with Timer('Load validation data', logger):
            cache_path = _get_cache_path(valdir)
            if cache_dataset and os.path.exists(cache_path):
                # Attention, as the transforms are also cached!
                dataset_test, _ = torch.load(cache_path)
                logger.info("Loaded test dataset from {}".format(cache_path))
            else:
                dataset_test = torchvision.datasets.ImageFolder(valdir)
                if cache_dataset:
                    safe_makedirs(os.path.dirname(cache_path))
                    save_on_master((dataset_test, valdir), cache_path)
                    logger.info("Cached test dataset to {}".format(cache_path))
                logger.info("Loaded test dataset")
    else:
        raise ValueError(dataset_type)

    dataset_train = DatasetWarpper(dataset, transform_train)
    dataset_test = DatasetWarpper(dataset_test, transform_test)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, train_sampler, test_sampler


def train_one_epoch(model, criterion, penalty_term, optimizer_train, optimizer_prune,
                    data_loader_train, temp_scheduler, logger, epoch, print_freq, factor,
                    scaler=None, accumulate_step=1, prune=False, one_hot=None, TET=False):
    model.train()
    metric_dict = RecordDict({'loss': None, 'acc@1': None, 'acc@5': None})
    timer_container = [0.0]

    set_pruning_mode(model, prune)
    model.zero_grad()
    for idx, (image, target) in enumerate(data_loader_train):
        with GlobalTimer('iter', timer_container):
            image, target = image.float().cuda(), target.cuda()
            if scaler is not None:
                with amp.autocast():
                    output = model(image)
                    if one_hot:
                        loss = criterion(output, F.one_hot(target, one_hot).float())
                    else:
                        loss = criterion(output, target)
            else:
                output = model(image)
                if one_hot:
                    loss = criterion(output, F.one_hot(target, one_hot).float())
                else:
                    loss = criterion(output, target)
            metric_dict['loss'].update(loss.item())

            if prune:
                loss = loss + penalty_term()

            loss = loss / accumulate_step

            if scaler is not None:
                scaler.scale(loss).backward()
                if (idx + 1) % accumulate_step == 0:
                    if prune:
                        scaler.step(optimizer_prune)
                    scaler.step(optimizer_train)
                    scaler.update()
                    model.zero_grad()
                    if temp_scheduler is not None:
                        temp_scheduler.step()

            else:
                loss.backward()
                if (idx + 1) % accumulate_step == 0:
                    if prune:
                        optimizer_prune.step()
                    optimizer_train.step()
                    model.zero_grad()
                    if temp_scheduler is not None:
                        temp_scheduler.step()

            functional.reset_net(model)

            acc1, acc5 = accuracy(output.mean(0), target, topk=(1, 5))
            acc1_s = acc1.item()
            acc5_s = acc5.item()

            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1_s, batch_size)
            metric_dict['acc@5'].update(acc5_s, batch_size)

        if print_freq != 0 and ((idx + 1) % int(len(data_loader_train) / (print_freq))) == 0:
            #torch.distributed.barrier()
            metric_dict.sync()
            logger.debug(' [{}/{}] it/s: {:.5f}, loss: {:.5f}, acc@1: {:.5f}, acc@5: {:.5f}'.format(
                idx + 1, len(data_loader_train),
                (idx + 1) * batch_size * factor / timer_container[0], metric_dict['loss'].ave,
                metric_dict['acc@1'].ave, metric_dict['acc@5'].ave))

    #torch.distributed.barrier()
    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['acc@1'].ave, metric_dict['acc@5'].ave


def evaluate(model, criterion, data_loader, print_freq, logger, prune, one_hot):
    model.eval()
    set_pruning_mode(model, prune)
    metric_dict = RecordDict({'loss': None, 'acc@1': None, 'acc@5': None})
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader):
            image = image.float().to(torch.device('cuda'), non_blocking=True)
            target = target.to(torch.device('cuda'), non_blocking=True)
            output = model(image)
            if one_hot:
                loss = criterion(output, F.one_hot(target, one_hot).float())
            else:
                loss = criterion(output, target)
            metric_dict['loss'].update(loss.item())
            functional.reset_net(model)

            acc1, acc5 = accuracy(output.mean(0), target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1.item(), batch_size)
            metric_dict['acc@5'].update(acc5.item(), batch_size)

            if print_freq != 0 and ((idx + 1) % int(len(data_loader) / print_freq)) == 0:
                #torch.distributed.barrier()
                metric_dict.sync()
                logger.debug(' [{}/{}] loss: {:.5f}, acc@1: {:.5f}, acc@5: {:.5f}'.format(
                    idx + 1, len(data_loader), metric_dict['loss'].ave, metric_dict['acc@1'].ave,
                    metric_dict['acc@5'].ave))

    #torch.distributed.barrier()
    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['acc@1'].ave, metric_dict['acc@5'].ave


def test(model, dataset_type, data_loader_test, inputs, args, logger):

    safe_makedirs(os.path.join(args.output_dir, 'test'))

    set_pruning_mode(model, False)
    mon = SOPMonitor(model)

    logger.info('[Sparsity]')

    conn, total = model.connects()
    logger.info('Connections: left: {:.2e}, total: {:.2e}, connectivity {:.2f}%'.format(
        conn, total, 100 * conn / total))
    neuron_left, neuron_total = left_neurons(model)
    weight_left, weight_total = left_weights(model)
    logger.info('Neurons: left: {:.2e}, total: {:.2e}, percentage: {:.2f}%'.format(
        neuron_left, neuron_total, (neuron_left + 1e-10) / (neuron_total + 1e-10) * 100))
    logger.info('Weights: left: {:.2e}, total: {:.2e}, percentage: {:.2f}%'.format(
        weight_left, weight_total, (weight_left + 1e-10) / (weight_total + 1e-10) * 100))

    logger.info('[Efficiency]')

    model.eval()
    mon.enable()
    logger.debug('Test start')
    metric_dict = RecordDict({'acc@1': None, 'acc@5': None}, test=True)
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader_test):
            image, target = image.cuda(), target.cuda()
            output = model(image).mean(0)
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1.item(), batch_size)
            metric_dict['acc@5'].update(acc5.item(), batch_size)

            if args.print_freq != 0 and ((idx + 1) %
                                         int(len(data_loader_test) / args.print_freq)) == 0:
                logger.debug('Test: [{}/{}]'.format(idx + 1, len(data_loader_test)))

    metric_dict.sync()
    logger.info('Acc@1: {:.5f}, Acc@5: {:.5f}'.format(metric_dict['acc@1'].ave,
                                                      metric_dict['acc@5'].ave))

    ### FIXME: count ops and params of ConvBlock
    #ops, params = profile(model, inputs=(inputs, ), verbose=False)
    #ops, params = (ops / (1000**3)) / args.T, params / (1000**2)
    #functional.reset_net(model)
    #logger.info('MACs: {:.5f} G, params: {:.2f} M.'.format(ops, params))

    sops = 0
    for name in mon.monitored_layers:
        sublist = mon[name]
        sop = torch.cat(sublist).mean().item()
        sops = sops + sop
    sops = sops / (1000**3)
    # input is [N, C, H, W] or [T*N, C, H, W]
    sops = sops / args.batch_size
    logger.info('Avg SOPs: {:.5f} G, Power: {:.5f} mJ.'.format(sops, 0.9 * sops))
    #logger.info('A/S Power Ratio: {:.6f}'.format((4.6 * ops) / (0.9 * sops)))

    #
    # visualize neurons and weights
    #

    logger.info('[Neurons]')

    fig = plt.figure(figsize=(16, 16))
    norm = colors.Normalize(vmin=-1.5, vmax=1.5)
    idx = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, ConvBlock):
                if not module.sparse_neurons:
                    continue
                mask = module.neuron_mask.mask_value
                num_active = (mask > 0).sum().item()
                num = mask.numel()
                logger.info('layer [{}] {}: left: {}, total: {}, percentage: {:.2f}%'.format(
                    idx, name, num_active, num, 100 * num_active / num))
                mask.squeeze_(0).squeeze_(0)
                channels = mask.shape[0]
                ncols = int(sqrt(channels))
                nrows = ceil(channels / ncols)
                for c in range(channels):
                    ax = fig.add_subplot(nrows, ncols, c + 1)
                    ax.matshow((mask[c, ...] > 0).cpu().numpy(), cmap='bwr', norm=norm)
                    ax.axis('off')
                fig.tight_layout()
                fig.savefig(os.path.join(args.output_dir, 'test', f'neuron_{idx}_{name}.png'),
                            bbox_inches='tight')
                fig.clear()
                ax = fig.add_subplot(1, 1, 1)
                mask = mask.flatten().cpu().numpy()
                percentile = np.percentile(np.abs(mask), 99)
                ax.hist(mask[mask > 0], bins=1000, range=(0, percentile))
                ax.hist(mask[mask < 0], bins=1000, range=(-percentile, 0))
                fig.tight_layout()
                fig.savefig(os.path.join(args.output_dir, 'test', f'neuron_plot_{idx}_{name}.png'),
                            bbox_inches='tight')
                fig.clear()
                idx = idx + 1
        plt.close()

    logger.info('[Weights]')
    idx = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, ConvBlock):
                if not module.sparse_weights:
                    continue
                weight: torch.Tensor
                weight = module.weight_mask.mask_value
                num_active = (weight > 0).sum().item()
                num = weight.numel()
                logger.info('layer [{}] {}: left: {}, total: {}, percentage: {:.2f}%'.format(
                    idx, name, num_active, num, 100 * num_active / num))
                ncols = weight.shape[0]
                nrows = weight.shape[1]
                width = weight.shape[2]
                height = weight.shape[3]
                weight_nz = (weight > 0).float().cpu().flatten(0, 2).T
                weight_reshape = torch.zeros((nrows * height, ncols * width))
                for i in range(nrows):
                    weight_reshape[i * height:(i + 1) *
                                   height, :] = weight_nz[:,
                                                          i * ncols * width:(i + 1) * ncols * width]
                ax = fig.add_subplot(1, 1, 1)
                ax.matshow(weight_reshape.numpy(), cmap='bwr', norm=norm)
                ax.axis('off')
                fig.tight_layout()
                fig.savefig(os.path.join(args.output_dir, 'test', f'weight_{idx}_{name}.png'),
                            bbox_inches='tight')
                fig.clear()
                ax = fig.add_subplot(1, 1, 1)
                weight = weight.flatten().cpu().numpy()
                percentile = np.percentile(np.abs(weight), 99)
                ax.hist(weight[weight > 0], bins=1000, range=(0, percentile))
                ax.hist(weight[weight < 0], bins=1000, range=(-percentile, 0))
                fig.tight_layout()
                fig.savefig(os.path.join(args.output_dir, 'test', f'weight_plot_{idx}_{name}.png'),
                            bbox_inches='tight')
                fig.clear()
                idx = idx + 1
        plt.close()


def main():

    ##################################################
    #                       setup
    ##################################################

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    safe_makedirs(args.output_dir)
    logger = setup_logger(args.output_dir)

    distributed, rank, world_size, local_rank = init_distributed(logger, args.distributed_init_mode)

    logger.info(str(args))

    # load data

    dataset_type = args.dataset
    if dataset_type == 'CIFAR10' or dataset_type == 'CIFAR10DVS':
        num_classes = 10
    elif dataset_type == 'CIFAR100':
        num_classes = 100
    elif dataset_type == 'ImageNet':
        num_classes = 1000

    dataset_train, dataset_test, train_sampler, test_sampler = load_data(
        args.data_path, args.cache_dataset, dataset_type, distributed, args.augment, logger, args.T)
    logger.info('dataset_train: {}, dataset_test: {}'.format(len(dataset_train), len(dataset_test)))

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                                    sampler=train_sampler, num_workers=args.workers,
                                                    pin_memory=True, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                   sampler=test_sampler, num_workers=args.workers,
                                                   pin_memory=True, drop_last=False)

    # model

    model: Union[cifar10.Cifar10Net, cifar10dvs.VGGSNN, sew_resnet.SEWResNet_ImageNet,
                 sew_resnet.SEWResNet_CIFAR, sew_resnet.ResNet19]
    if args.model in cifar10.__dict__:
        model = cifar10.__dict__[args.model](T=args.T, num_classes=num_classes).cuda()
    elif args.model in cifar10dvs.__dict__:
        model = cifar10dvs.__dict__[args.model]().cuda()
    elif args.model in sew_resnet.__dict__:
        model = sew_resnet.__dict__[args.model](zero_init_residual=args.zero_init_residual,
                                                T=args.T, num_classes=num_classes).cuda()
    else:
        raise NotImplementedError(args.model)

    if args.not_prune_weight:
        for m in model.modules():
            if isinstance(m, ConvBlock):
                m.sparse_weights = False
    if args.not_prune_neuron:
        for m in model.modules():
            if isinstance(m, ConvBlock):
                m.sparse_neurons = False

    model.cuda()
    if distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # optimzer

    param_without_masks = list(model.parameters())

    if args.optimizer == 'SGD':
        optimizer_train = torch.optim.SGD(param_without_masks, lr=args.search_lr, momentum=0.9,
                                          weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer == 'Adam':
        optimizer_train = torch.optim.Adam(param_without_masks, lr=args.search_lr,
                                           betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        raise ValueError(args.optimizer)

    # init mask

    set_pruning_mode(model, True)
    if dataset_type == 'CIFAR10' or dataset_type == 'CIFAR100':
        inputs = torch.rand(1, 3, 32, 32).cuda()
    elif dataset_type == 'CIFAR10DVS':
        inputs = torch.rand(1, 1, 2, 48, 48).cuda()
    elif dataset_type == 'ImageNet':
        inputs = torch.rand(1, 3, 224, 224).cuda()
    _ = model(inputs)

    masks = init_mask(model, *args.mask_init_factor)
    set_pruning_mode(model, False)
    functional.reset_net(model)

    if not (args.not_prune_weight and args.not_prune_neuron):
        if args.prune_optimizer is None:
            args.prune_optimizer = args.optimizer
        if args.prune_lr is None:
            args.prune_lr = args.search_lr
        if args.prune_optimizer == 'SGD':
            optimizer_prune = torch.optim.SGD(masks, lr=args.prune_lr, momentum=0.9,
                                              weight_decay=args.prune_weight_decay, nesterov=True)
        elif args.prune_optimizer == 'Adam':
            optimizer_prune = torch.optim.Adam(masks, lr=args.prune_lr, betas=(0.9, 0.999),
                                               weight_decay=args.prune_weight_decay)
        else:
            raise ValueError(args.prune_optimizer)

    # loss_fn

    if dataset_type == 'CIFAR10' or dataset_type == 'CIFAR10DVS':
        one_hot = 10
    elif dataset_type == 'CIFAR100':
        one_hot = 100
    elif dataset_type == 'ImageNet':
        one_hot = None

    if args.criterion == 'MSE':
        criterion = nn.MSELoss()
    elif args.criterion == 'CE':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(args.criterion)
    criterion = CriterionWarpper(criterion, args.TET, args.TET_phi, args.TET_lambda)

    # penalty term

    if not (args.not_prune_weight and args.not_prune_neuron):
        penalty_term = PenaltyTerm(model, args.penalty_lmbda)

    # amp speed up

    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None

    # lr scheduler

    milestones = []
    lr_scheduler_train, lr_scheduler_prune = None, None
    lr_scheduler_T0, lr_scheduler_Tmax = 0, args.epoch_search
    if not (args.not_prune_weight and args.not_prune_neuron):
        if len(args.search_lr_scheduler) != 0:
            if args.search_lr_scheduler[0] == 'Step':
                for i in range(1, len(args.search_lr_scheduler)):
                    milestones.append(int(args.search_lr_scheduler[i]))
                lr_scheduler_train = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer_train, milestones=milestones, gamma=0.1)
                lr_scheduler_prune = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer_prune, milestones=milestones, gamma=0.1)
            elif args.search_lr_scheduler[0] == 'Cosine':
                if len(args.search_lr_scheduler) > 1:
                    lr_scheduler_T0, lr_scheduler_Tmax, T_max = int(
                        args.search_lr_scheduler[1]), int(args.search_lr_scheduler[2]), int(
                            args.search_lr_scheduler[3])
                else:
                    T_max = lr_scheduler_Tmax - lr_scheduler_T0
                lr_scheduler_train = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer_train, T_max=T_max)
                lr_scheduler_prune = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer_prune, T_max=T_max)
            else:
                raise ValueError(args.search_lr_scheduler)

    # DDP

    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module

    # threshold scheduler

    if not (args.not_prune_weight and args.not_prune_neuron):
        iter_per_epoch = len(data_loader_train) // args.accumulate_step
        if len(args.temp_scheduler) == 2:
            (args.temp_scheduler).append(0)
            (args.temp_scheduler).append(args.epoch_search)
        if len(args.temp_scheduler) == 4:
            temp_scheduler = TemperatureScheduler(model, args.temp_scheduler[0],
                                                  args.temp_scheduler[1],
                                                  int(args.temp_scheduler[2]) * iter_per_epoch,
                                                  int(args.temp_scheduler[3]) * iter_per_epoch)
        elif len(args.temp_scheduler) == 6:
            temp_scheduler = SplitTemperatureScheduler(model, args.temp_scheduler[0],
                                                       args.temp_scheduler[1],
                                                       args.temp_scheduler[2],
                                                       args.temp_scheduler[3],
                                                       int(args.temp_scheduler[4]) * iter_per_epoch,
                                                       int(args.temp_scheduler[5]) * iter_per_epoch)
        else:
            raise ValueError(args.temp_scheduler)

    # resume

    if args.resume and args.resume_type == 'search':
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer_train.load_state_dict(checkpoint['optimizer_train'])
        optimizer_prune.load_state_dict(checkpoint['optimizer_prune'])
        start_epoch = checkpoint['epoch']
        max_acc1 = checkpoint['max_acc1']
        if lr_scheduler_train is not None:
            lr_scheduler_train.load_state_dict(checkpoint['lr_scheduler_train'])
            lr_scheduler_prune.load_state_dict(checkpoint['lr_scheduler_prune'])
        logger.info('Resume from epoch {}'.format(start_epoch))
        start_epoch += 1
        temp_scheduler.current_step = start_epoch * len(data_loader_train)
    else:
        start_epoch = 0
        max_acc1 = 0

    logger.debug(str(model))

    ##################################################
    #                   test only
    ##################################################

    if args.test_only:
        if args.resume and args.resume_type == 'test':
            checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
        logger.info('Test start')
        if is_main_process():
            test(model, dataset_type, data_loader_test, inputs, args, logger)
        return

    ##################################################
    #                   search
    ##################################################

    tb_writer = None
    if is_main_process():
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'),
                                  purge_step=start_epoch)

    logger.info("Search start")
    for epoch in range(start_epoch, args.epoch_search):
        if args.resume and args.resume_type == 'finetune':
            break
        if distributed:
            train_sampler.set_epoch(epoch)
        logger.info('Epoch [{}] Start, lr {:.6f}, {}'.format(epoch,
                                                             optimizer_train.param_groups[0]["lr"],
                                                             str(temp_scheduler)))

        with Timer(' Train', logger):
            logger.debug('[Training]')
            train_loss, train_acc1, train_acc5 = train_one_epoch(
                model, criterion, penalty_term, optimizer_train, optimizer_prune, data_loader_train,
                temp_scheduler, logger, epoch, args.print_freq, world_size, scaler,
                args.accumulate_step, True, one_hot)
            if lr_scheduler_train is not None and lr_scheduler_T0 <= epoch < lr_scheduler_Tmax:
                lr_scheduler_train.step()
                lr_scheduler_prune.step()

        for n, m in model.named_modules():
            if isinstance(m, Mask):
                if m.mask_value is not None:
                    logger.debug(' {}: {:.3}%'.format(n, m.mask().mean() * 100))

        with Timer(' Test', logger):
            logger.debug('[Test with continuous mask]')
            test_loss_c, test_acc1_c, test_acc5_c = evaluate(model, criterion, data_loader_test,
                                                             args.print_freq, logger, True, one_hot)
            logger.debug('[Test with binary mask]')
            test_loss_s, test_acc1_s, test_acc5_s = evaluate(model, criterion, data_loader_test,
                                                             args.print_freq, logger, False,
                                                             one_hot)
        set_pruning_mode(model, True)
        n_l, n_t = left_neurons(model)
        w_l, w_t = left_weights(model)
        c, t = model_without_ddp.connects()
        neu, wei = 100 * (n_l + 1e-10) / (n_t + 1e-10), 100 * (w_l + 1e-10) / (w_t + 1e-10)
        conn = 100 * (c + 1e-10) / (t + 1e-10)
        search_tb_record(tb_writer, model, train_loss, train_acc1, train_acc5, test_loss_c,
                         test_acc1_c, test_acc5_c, test_loss_s, test_acc1_s, test_acc5_s, epoch,
                         args.tb_interval)

        logger.info(' Test (continuous mask) Acc@1: {:.5f}, Acc@5: {:.5f}'.format(
            test_acc1_c, test_acc5_c))
        logger.info(' Test (binary mask) Acc@1: {:.5f}, Acc@5: {:.5f}'.format(
            test_acc1_s, test_acc5_s))
        logger.info(' left neurons: {:.2f}%, left weights: {:.2f}%, connectivity: {:.2f}%'.format(
            neu, wei, conn))

        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer_train': optimizer_train.state_dict(),
            'optimizer_prune': optimizer_prune.state_dict(),
            'epoch': epoch,
            'max_acc1': max_acc1, }
        if lr_scheduler_train is not None:
            checkpoint['lr_scheduler_train'] = lr_scheduler_train.state_dict()
            checkpoint['lr_scheduler_prune'] = lr_scheduler_prune.state_dict()

        if args.save_latest:
            save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_latest.pth'))

        if (epoch + 1) == args.epoch_search:
            save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_sparsified.pth'))

    logger.info('Search finish.')

    ##################################################
    #                   finetune
    ##################################################

    ##### reset utils #####

    # reset lr
    if args.finetune_lr is None:
        args.finetune_lr = args.search_lr
    for param_group in optimizer_train.param_groups:
        param_group['lr'] = args.finetune_lr

    # lr scheduler

    milestones = []
    lr_scheduler_train = None
    lr_scheduler_T0, lr_scheduler_Tmax = 0, args.epoch_finetune
    if len(args.finetune_lr_scheduler) != 0:
        if args.finetune_lr_scheduler[0] == 'Step':
            for i in range(1, len(args.finetune_lr_scheduler)):
                milestones.append(int(args.finetune_lr_scheduler[i]))
            lr_scheduler_train = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_train,
                                                                      milestones=milestones,
                                                                      gamma=0.1)
        elif args.finetune_lr_scheduler[0] == 'Cosine':
            if len(args.finetune_lr_scheduler) > 1:
                lr_scheduler_T0, lr_scheduler_Tmax, T_max = int(args.finetune_lr_scheduler[1]), int(
                    args.finetune_lr_scheduler[2]), int(args.finetune_lr_scheduler[3])
            else:
                T_max = lr_scheduler_Tmax - lr_scheduler_T0
            lr_scheduler_train = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer_train, T_max=T_max)
        else:
            raise ValueError(args.finetune_lr_scheduler)

    # resume

    if args.resume and args.resume_type == 'finetune':
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer_train.load_state_dict(checkpoint['optimizer_train'])
        start_epoch = checkpoint['epoch']
        max_acc1 = checkpoint['max_acc1']
        if lr_scheduler_train is not None:
            lr_scheduler_train.load_state_dict(checkpoint['lr_scheduler_train'])
        logger.info('Resume from epoch {}'.format(start_epoch))
        start_epoch += 1
    else:
        start_epoch = 0

    ##### finetune #####

    logger.info("Finetune start")
    for epoch in range(start_epoch, args.epoch_finetune):
        save_max = False
        if distributed:
            train_sampler.set_epoch(epoch)
        logger.info('Epoch [{}] Start, lr {:.6f}'.format(epoch,
                                                         optimizer_train.param_groups[0]["lr"]))

        with Timer(' Train', logger):
            logger.debug('[Training]')
            train_loss, train_acc1, train_acc5 = train_one_epoch(
                model, criterion, None, optimizer_train, None, data_loader_train, None, logger,
                epoch, args.print_freq, world_size, scaler, args.accumulate_step, False, one_hot)
            if lr_scheduler_train is not None and lr_scheduler_T0 <= epoch < lr_scheduler_Tmax:
                lr_scheduler_train.step()

        with Timer(' Test', logger):
            logger.debug('[Test]')
            test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_test,
                                                       args.print_freq, logger, False, one_hot)

        finetune_tb_record(tb_writer, train_loss, train_acc1, train_acc5, test_loss, test_acc1,
                           test_acc5, epoch)

        logger.info(' Test Acc@1: {:.5f}, Acc@5: {:.5f}'.format(test_acc1, test_acc5))

        if max_acc1 < test_acc1:
            max_acc1 = test_acc1
            save_max = True

        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer_train': optimizer_train.state_dict(),
            'epoch': epoch,
            'max_acc1': max_acc1, }
        if lr_scheduler_train is not None:
            checkpoint['lr_scheduler_train'] = lr_scheduler_train.state_dict()

        if args.save_latest:
            save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_latest.pth'))

        if save_max:
            save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint_max_acc1.pth'))

    logger.info('Finetune finish.')

    ##################################################
    #                   test
    ##################################################

    ##### reset utils #####

    # reset model

    del model, model_without_ddp

    if args.model in cifar10.__dict__:
        model = cifar10.__dict__[args.model](T=args.T).cuda()
    elif args.model in cifar10dvs.__dict__:
        model = cifar10dvs.__dict__[args.model]().cuda()
    elif args.model in sew_resnet.__dict__:
        model = sew_resnet.__dict__[args.model](zero_init_residual=args.zero_init_residual,
                                                T=args.T, num_classes=num_classes).cuda()
    if args.not_prune_weight:
        for m in model.modules():
            if isinstance(m, ConvBlock):
                m.sparse_weights = False
    if args.not_prune_neuron:
        for m in model.modules():
            if isinstance(m, ConvBlock):
                m.sparse_neurons = False

    model.cuda()

    # init mask

    if dataset_type == 'CIFAR10' or dataset_type == 'CIFAR100':
        inputs = torch.rand(1, 3, 32, 32).cuda()
    elif dataset_type == 'CIFAR10DVS':
        inputs = torch.rand(1, 1, 2, 48, 48).cuda()
    elif dataset_type == 'ImageNet':
        inputs = torch.rand(1, 3, 224, 224).cuda()
    _ = model(inputs)

    masks = init_mask(model, 1, 1, 0, 0)
    functional.reset_net(model)

    try:
        checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoint_max_acc1.pth'),
                                map_location='cpu')
    except:
        logger.warning('Cannot load max acc1 model, skip test.')
        logger.warning('Exit.')
        return

    model.load_state_dict(checkpoint['model'])

    # reload data

    del test_sampler, data_loader_test

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                   sampler=test_sampler, num_workers=args.workers,
                                                   pin_memory=False, drop_last=False)

    ##### test #####

    logger.info('Test start')
    if is_main_process():
        test(model, dataset_type, data_loader_test, inputs, args, logger)
    logger.info('All Done.')


if __name__ == "__main__":
    main()
