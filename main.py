#!coding:utf-8
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from util import datasets, Trainer
from architectures.arch import arch

from util.datasets import NO_LABEL, MyDataSet


def create_data_loaders(train_transform, 
                        eval_transform, 
                        datadir,
                        config):
    traindir = os.path.join(datadir, config.train_subdir)  # 'datadir': './data-local/images/cifar/cifar10/by-image'; 'train_subdir': 'train+val'
    # trainset = torchvision.datasets.ImageFolder(traindir, train_transform)
    trainset = MyDataSet(train_transform)
    # if config.labels:
    #     with open(config.labels) as f:
    #         labels = dict(line.split(' ') for line in f.read().splitlines())
    #     labeled_idxs, unlabeled_idxs = datasets.relabel_dataset(trainset, labels)
    # assert len(trainset.imgs) == len(labeled_idxs)+len(unlabeled_idxs)

    # 直接读取txt文件，要有id号
    unlabeled_idxs = []
    labeled_idxs = []

    def read_label():
        with open('/home/indemind/Project/pseudo_label/data-local/train.txt', 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                line = line.strip('\n')
                lis = line.split(' ')
                label = int(lis[1])

                if label == -1:
                    unlabeled_idxs.append(idx)
                else:
                    labeled_idxs.append(idx)
            return labeled_idxs, unlabeled_idxs

    labeled_idxs, unlabeled_idxs = read_label()
    assert len(trainset.imgs) == len(labeled_idxs) + len(unlabeled_idxs)

    if config.labeled_batch_size < config.batch_size:  # 64 < 128
        assert len(unlabeled_idxs) > 0
        batch_sampler = datasets.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, config.batch_size, config.labeled_batch_size)
    else:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, config.batch_size, drop_last=True)
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_sampler=batch_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)

    evaldir = os.path.join(datadir, config.eval_subdir)
    evalset = torchvision.datasets.ImageFolder(evaldir, eval_transform)
    eval_loader = torch.utils.data.DataLoader(evalset,
                                              batch_size=config.batch_size,
                                              shuffle=False,
                                              num_workers=config.workers,
                                              pin_memory=True,
                                              drop_last=False)
    return train_loader, eval_loader


def create_loss_fn(config):
    if config.loss == 'soft':
        # for pytorch 0.4.0
        criterion = nn.CrossEntropyLoss(ignore_index=NO_LABEL, reduce=False)
        # for pytorch 0.4.1
        #criterion = nn.CrossEntropyLoss(ignore_index=NO_LABEL, reduction='none')
    return criterion

def create_optim(params, config):
    if config.optim == 'sgd':
        optimizer = optim.SGD(params, config.lr,
                              momentum=config.momentum,
                              weight_decay=config.weight_decay,
                              nesterov=config.nesterov)
    elif config.optim == 'adam':
        optimizer = optim.Adam(params, config.lr)
    return optimizer

def create_lr_scheduler(optimizer, config):
    if config.lr_scheduler == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=config.epochs,
                                                   eta_min=config.min_lr)
    elif config.lr_scheduler == 'multistep':
        if config.steps=="":
            return None
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=config.steps,
                                             gamma=config.gamma)
    elif config.lr_scheduler == 'none':
        scheduler = None
    return scheduler
def relabel_dataset(dataset, labels):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        path, _ = dataset.imgs[idx]
        filename = os.path.basename(path)
        if filename in labels:
            label_idx = dataset.class_to_idx[labels[filename]]
            dataset.imgs[idx] = path, label_idx
            del labels[filename]
        else:
            dataset.imgs[idx] = path, NO_LABEL
            unlabeled_idxs.append(idx)

    if len(labels)!=0:
        message = "List of unlabeled contains {} unknow files: {}, ..."
        some_missing = ', '.join(list(labels.keys())[:5])
        raise LookupError(message.format(len(labels), some_missing))

    labeled_idxs = sorted(set(range(len(dataset.imgs)))-set(unlabeled_idxs))
    return labeled_idxs, unlabeled_idxs


def main(config):
    with SummaryWriter(comment='_{}_{}'.format(config.arch,config.dataset)) as writer:
        dataset_config = datasets.cifar10() if config.dataset=='cifar10' else datasets.cifar100()
        num_classes = dataset_config.pop('num_classes')  # 类别为10
        train_loader, eval_loader = create_data_loaders(**dataset_config, config=config)

        dummy_input = (torch.randn(10,3,32,32),)
        net = arch[config.arch](num_classes)  # 采用的结构是resnet18
        # writer.add_graph(net, dummy_input)  # 它的作用是什么

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = create_loss_fn(config)  # 交叉熵损失函数
        if config.is_parallel:
            net = torch.nn.DataParallel(net).to(device)
        else:
            device = 'cuda:{}'.format(config.gpu) if torch.cuda.is_available() else 'cpu'
            net = net.to(device)
        optimizer = create_optim(net.parameters(), config)  # 随机梯度下降法
        scheduler = create_lr_scheduler(optimizer, config)  # 学习率调整策略，余弦

        trainer = Trainer.PseudoLabel(net, optimizer, criterion, device, config, writer,)
        trainer.loop(config.epochs, train_loader, eval_loader,
                     scheduler=scheduler, print_freq=config.print_freq)
