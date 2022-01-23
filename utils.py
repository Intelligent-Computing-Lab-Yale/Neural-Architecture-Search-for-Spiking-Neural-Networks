import os
import time
import torch
import numpy as np
import torchvision.transforms as transforms

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)

        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, iters, tag=''):
    if not os.path.exists("./snapshots"):
        os.makedirs("./snapshots")
    filename = os.path.join("./snapshots/{}_ckpt_{:04}.pth.tar".format(tag, iters))
    torch.save(state, filename)



def data_transforms(args):
    if args.dataset == 'cifar10':
        MEAN = [0.4913, 0.4821, 0.4465]
        STD = [0.2470, 0.2434, 0.2615]
    elif args.dataset == 'cifar100':
        MEAN = [0.5071, 0.4867, 0.4408]
        STD = [0.2673, 0.2564, 0.2762]
    elif args.dataset == 'tinyimagenet':
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

    if (args.dataset== 'tinyimagenet'):
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:  # cifar10 or cifar100
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    return train_transform, valid_transform


def random_choice(num_choice, layers):
    return list(np.random.randint(num_choice, size=layers))



def time_record(start):
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    print('Elapsed time: hour: %d, minute: %d, second: %f' % (hour, minute, second))
