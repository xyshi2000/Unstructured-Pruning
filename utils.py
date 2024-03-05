import math
import os
import time
import torch
import torch.distributed
import errno
import datetime
import random
import numpy as np
from torchvision import transforms
from torch import Tensor, nn
from math import nan
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageEnhance, ImageOps

from models.submodules.sparse import ConvBlock


def is_distributed():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_rank():
    if not is_distributed():
        return 0
    return torch.distributed.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def safe_makedirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def search_tb_record(tb_writer: SummaryWriter, model: nn.Module, train_loss, train_acc1, train_acc5,
                     test_loss_c, test_acc1_c, test_acc5_c, test_loss_s, test_acc1_s, test_acc5_s,
                     epoch, interval=1):
    if is_main_process():
        tb_writer.add_scalar('search/train/loss', train_loss, epoch)
        tb_writer.add_scalar('search/train/acc1', train_acc1, epoch)
        tb_writer.add_scalar('search/train/acc5', train_acc5, epoch)
        tb_writer.add_scalar('search/test/continuous/loss', test_loss_c, epoch)
        tb_writer.add_scalar('search/test/continuous/acc1', test_acc1_c, epoch)
        tb_writer.add_scalar('search/test/continuous/acc5', test_acc5_c, epoch)
        tb_writer.add_scalar('search/test/binary/loss', test_loss_s, epoch)
        tb_writer.add_scalar('search/test/binary/acc1', test_acc1_s, epoch)
        tb_writer.add_scalar('search/test/binary/acc5', test_acc5_s, epoch)

        neu_cnt, neu_total = 0, 0
        wei_cnt, wei_total = 0, 0
        for name, module in model.named_modules():
            if isinstance(module, ConvBlock):
                if (epoch + 1) % interval == 0 or epoch == 0:
                    weight = module.conv.weight
                    tb_writer.add_histogram(f'weight/{name}', weight, epoch)
                if module.sparse_weights:
                    w_c, w_t = module.weight_mask.left()
                    wei_cnt, wei_total = wei_cnt + w_c, wei_total + w_t
                    tb_writer.add_scalar(f'left weight/{name}', w_c / w_t, epoch)
                    if (epoch + 1) % interval == 0 or epoch == 0:
                        masked_weight = module.weight_mask(weight)
                        if not torch.all(masked_weight == 0):
                            tb_writer.add_histogram(f'masked weight/{name}',
                                                    masked_weight[masked_weight != 0], epoch)
                        tb_writer.add_histogram(f'weight mask/{name}',
                                                module.weight_mask.mask_value, epoch)
                if module.sparse_neurons:
                    n_c, n_t = module.neuron_mask.left()
                    neu_cnt, neu_total = neu_cnt + n_c, neu_total + n_t
                    tb_writer.add_scalar(f'left neurons/{name}', n_c / n_t, epoch)
                    if (epoch + 1) % interval == 0 or epoch == 0:
                        tb_writer.add_histogram(f'neuron mask/{name}',
                                                module.neuron_mask.mask_value, epoch)
        if wei_total != 0:
            tb_writer.add_scalar('left weight', wei_cnt / wei_total, epoch)
        if neu_total != 0:
            tb_writer.add_scalar('left neurons', neu_cnt / neu_total, epoch)


def finetune_tb_record(tb_writer: SummaryWriter, train_loss, train_acc1, train_acc5, test_loss,
                       test_acc1, test_acc5, epoch):
    if is_main_process():
        tb_writer.add_scalar('finetune/train/loss', train_loss, epoch)
        tb_writer.add_scalar('finetune/train/acc1', train_acc1, epoch)
        tb_writer.add_scalar('finetune/train/acc5', train_acc5, epoch)
        tb_writer.add_scalar('finetune/test/loss', test_loss, epoch)
        tb_writer.add_scalar('finetune/test/acc1', test_acc1, epoch)
        tb_writer.add_scalar('finetune/test/acc5', test_acc5, epoch)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, )):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def init_mask(model: nn.Module, weights_mean: float, neurons_mean: float, weights_std: float = 0,
              neurons_std: float = 0):
    mask_list = []
    for m in model.modules():
        if isinstance(m, ConvBlock):
            masks = m.init_mask(weights_mean, neurons_mean, weights_std, neurons_std)
            for mask in masks:
                mask_list.append(mask)
    return mask_list


def set_pruning_mode(model: nn.Module, mode: bool = False):
    for m in model.modules():
        if isinstance(m, ConvBlock):
            m._pruning(mode)


def left_neurons(model: nn.Module):
    conn = 0
    total = 0
    for m in model.modules():
        if isinstance(m, ConvBlock):
            c, t = m.neuron_mask.left()
            conn, total = conn + c, total + t
    return conn, total


def left_weights(model: nn.Module):
    conn = 0
    total = 0
    for m in model.modules():
        if isinstance(m, ConvBlock):
            c, t = m.weight_mask.left()
            conn, total = conn + c, total + t
    return conn, total


class Record:
    r'''
    Synchronous record
    '''
    def __init__(self, test: bool = False) -> None:
        self.value = torch.tensor([0], dtype=torch.float64, device='cuda')
        self.count = torch.tensor([0], dtype=torch.int64, device='cuda')
        self.global_value = 0.0
        self.global_count = 0
        self.test = test

    def sync(self) -> None:
        r'''
        reduce value and count, and update global ones
        '''
        if is_distributed() and not self.test:
            torch.distributed.all_reduce(self.value, torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(self.count, torch.distributed.ReduceOp.SUM)
        self.global_value += self.value.item()
        self.global_count += self.count.item()
        self.value[0] = 0.0
        self.count[0] = 0

    def update(self, value, count=1) -> None:
        r'''
        update local value and count
        '''
        self.value[0] += value * count
        self.count[0] += count

    def reset(self) -> None:
        self.value[0] = 0.0
        self.count[0] = 0
        self.global_value = 0.0
        self.global_count = 0

    @property
    def ave(self):
        if self.global_count == 0:
            return nan
        return self.global_value / self.global_count


class RecordDict:
    def __init__(self) -> None:
        self.__inner_dict = dict()

    def __init__(self, dic: dict, test: bool = False) -> None:
        self.__inner_dict = dict()
        self.test = test
        for key in dic.keys():
            self.__inner_dict[key] = Record(test)

    def __getitem__(self, key) -> Record:
        return self.__inner_dict[key]

    def __setitem__(self, key, value) -> None:
        assert (isinstance(value, Record))
        self.__inner_dict[key] = value

    def __str__(self) -> str:
        s = []
        for key, value in self.__inner_dict.items():
            s.append('{key}:{value}'.format(key=key, value=value.ave))
        return ', '.join(s)

    def sync(self):
        for value in self.__inner_dict.values():
            value.sync()

    def reset(self):
        for value in self.__inner_dict.values():
            value.reset()

    def add_record(self, key):
        self.__inner_dict[key] = Record(self.test)


class Timer:
    def __init__(self, timer_name, logger):
        self.timer_name = timer_name
        self.logger = logger

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # seconds
        self.logger.debug('{} spent: {}.'.format(
            self.timer_name, str(datetime.timedelta(seconds=int(self.interval)))))


class GlobalTimer:
    def __init__(self, timer_name, container):
        self.timer_name = timer_name
        self.container = container

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # seconds
        self.container[0] += self.interval


class DatasetSplitter(torch.utils.data.Dataset):
    '''To split CIFAR10DVS into training dataset and test dataset'''
    def __init__(self, parent_dataset, rate=0.1, train=True):

        self.parent_dataset = parent_dataset
        self.rate = rate
        self.train = train
        self.it_of_original = len(parent_dataset) // 10
        self.it_of_split = int(self.it_of_original * rate)

    def __len__(self):
        return int(len(self.parent_dataset) * self.rate)

    def __getitem__(self, index):
        base = (index // self.it_of_split) * self.it_of_original
        off = index % self.it_of_split
        if not self.train:
            off = self.it_of_original - off - 1
        item = self.parent_dataset[base + off]

        return item


class CriterionWarpper(nn.Module):
    def __init__(self, criterion, TET=False, TET_phi=1.0, TET_lambda=0.0) -> None:
        super().__init__()
        self.criterion = criterion
        self.TET = TET
        self.TET_phi = TET_phi
        self.TET_lambda = TET_lambda
        self.mse = nn.MSELoss()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if self.TET:
            loss = 0
            for t in range(output.shape[0]):
                loss = loss + (1. - self.TET_lambda) * self.criterion(output[t], target)
            loss = loss / output.shape[0]
            if self.TET_lambda != 0:
                loss = loss + self.TET_lambda * self.mse(
                    output,
                    torch.zeros_like(output).fill_(self.TET_phi))
            return loss
        else:
            return self.criterion(output.mean(0), target)


class DatasetWarpper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.trasnform = transform

    def __getitem__(self, index):
        return self.trasnform(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


class DVStransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = torch.from_numpy(img).float()
        shape = [img.shape[0], img.shape[1]]
        img = img.flatten(0, 1)
        img = self.transform(img)
        shape.extend(img.shape[1:])
        return img.view(shape)


class Augment:
    def __init__(self):
        pass

    class Cutout:
        """Randomly mask out one or more patches from an image.
        Args:
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        """
        def __init__(self, ratio):
            self.ratio = ratio

        def __call__(self, img):
            h = img.size(1)
            w = img.size(2)
            lenth_h = int(self.ratio * h)
            lenth_w = int(self.ratio * w)
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - lenth_h // 2, 0, h)
            y2 = np.clip(y + lenth_h // 2, 0, h)
            x1 = np.clip(x - lenth_w // 2, 0, w)
            x2 = np.clip(x + lenth_w // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask
            return img

    class Roll:
        def __init__(self, off):
            self.off = off

        def __call__(self, img):
            off1 = random.randint(-self.off, self.off)
            off2 = random.randint(-self.off, self.off)
            return torch.roll(img, shifts=(off1, off2), dims=(1, 2))

    def function_nda(self, data, M=1, N=2):
        c = 15 * N
        rotate = transforms.RandomRotation(degrees=c)
        e = N / 6
        cutout = self.Cutout(ratio=e)
        a = N * 2 + 1
        roll = self.Roll(off=a)

        transforms_list = [roll, rotate, cutout]
        sampled_ops = np.random.choice(transforms_list, M)
        for op in sampled_ops:
            data = op(data)
        return data

    def trans(self, data):
        flip = random.random() > 0.5
        if flip:
            data = torch.flip(data, dims=(2, ))
        data = self.function_nda(data)
        return data

    def __call__(self, img):
        return self.trans(img)


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


# code from https://github.com/yhhhli/SNN_Calibration/blob/master/data/autoaugment.py


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2,
                 fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int_),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10}

        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128, ) * 4),
                                   rot).convert(img.mode)

        func = {
            "shearX":
            lambda img, magnitude: img.transform(img.size, Image.AFFINE,
                                                 (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0
                                                  ), Image.BICUBIC, fillcolor=fillcolor),
            "shearY":
            lambda img, magnitude: img.transform(img.size, Image.AFFINE,
                                                 (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0
                                                  ), Image.BICUBIC, fillcolor=fillcolor),
            "translateX":
            lambda img, magnitude: img.transform(img.size, Image.AFFINE,
                                                 (1, 0, magnitude * img.size[0] * random.choice([
                                                     -1, 1]), 0, 1, 0), fillcolor=fillcolor),
            "translateY":
            lambda img, magnitude: img.transform(img.size, Image.AFFINE,
                                                 (1, 0, 0, 0, 1, magnitude * img.size[1] * random.
                                                  choice([-1, 1])), fillcolor=fillcolor),
            "rotate":
            lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color":
            lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([
                -1, 1])),
            "posterize":
            lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize":
            lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast":
            lambda img, magnitude: ImageEnhance.Contrast(img).enhance(1 + magnitude * random.choice(
                [-1, 1])),
            "sharpness":
            lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(1 + magnitude * random.
                                                                       choice([-1, 1])),
            "brightness":
            lambda img, magnitude: ImageEnhance.Brightness(img).enhance(1 + magnitude * random.
                                                                        choice([-1, 1])),
            "autocontrast":
            lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize":
            lambda img, magnitude: ImageOps.equalize(img),
            "invert":
            lambda img, magnitude: ImageOps.invert(img)}

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


class CIFAR10Policy(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),
            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),
            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),
            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),
            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)


def unpack_len1_tuple(x: tuple or torch.Tensor):
    if isinstance(x, tuple) and x.__len__() == 1:
        return x[0]
    else:
        return x


def snn_to_ann_input(x: torch.Tensor):
    if len(x.shape) == 5:
        return x.flatten(0, 1)
    else:
        return x


class BaseMonitor:
    def __init__(self):
        self.hooks = []
        self.monitored_layers = []
        self.records = []
        self.name_records_index = {}
        self._enable = True

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.records[i]
        elif isinstance(i, str):
            y = []
            for index in self.name_records_index[i]:
                y.append(self.records[index])
            return y
        else:
            raise ValueError(i)

    def clear_recorded_data(self):
        self.records.clear()
        for k, v in self.name_records_index.items():
            v.clear()

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def is_enable(self):
        return self._enable

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.remove_hooks()


class SOPMonitor(BaseMonitor):
    def __init__(self, net: nn.Module):
        super().__init__()
        for name, m in net.named_modules():
            if name in net.skip:
                continue
            if isinstance(m, ConvBlock):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                # conv.weight [C_out, C_in, H_k, W_k]
                if m.sparse_weights:
                    connects = (m.weight_mask(m.conv.weight) != 0).float()
                else:
                    connects = torch.ones_like(m.conv.weight)
                if m.sparse_neurons:
                    mask = (m.neuron_mask.mask_value > 0).float().squeeze(0)
                else:
                    mask = None
                self.hooks.append(m.register_forward_hook(self.create_hook(name, connects, mask)))

    def cal_sop(self, x: Tensor, connects: Tensor, mask: Tensor, m: nn.Conv2d):
        out = torch.nn.functional.conv2d(x, connects, None, m.stride, m.padding, m.dilation,
                                         m.groups)
        if mask is None:
            sop = out.sum()
        else:
            sop = (out * mask).sum()
        return sop.unsqueeze(0)

    def create_hook(self, name, connects, mask):
        def hook(m: ConvBlock, x: Tensor, y: Tensor):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(
                    self.cal_sop(
                        snn_to_ann_input(unpack_len1_tuple(x)).detach(), connects, mask, m.conv))

        return hook
