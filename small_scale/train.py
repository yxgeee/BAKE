from __future__ import print_function

import argparse
import csv
import os, logging
import random

import numpy as np
import torch
from torch.autograd import Variable, grad
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import models
from utils import progress_bar, set_logging_defaults
from datasets import load_dataset

parser = argparse.ArgumentParser(description='Small-scale Datasets Training')

parser.add_argument('--name', default='cifar_res18_train', type=str, help='name of experiment')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--arch', '-a', default="CIFAR_ResNet18", type=str, help='model type (32x32: CIFAR_ResNet18, CIFAR_DenseNet121, 224x224: resnet18, densenet121)')
parser.add_argument('--resume', '-r', default="", help='resume from checkpoint')
parser.add_argument('--eval', action='store_true', help='only evaluate')
parser.add_argument('--sgpu', default=0, type=int, help='gpu index (start)')
parser.add_argument('--ngpu', default=1, type=int, help='number of gpu')
parser.add_argument('--dataroot', default='./data', type=str, help='data directory')
parser.add_argument('--saveroot', default='./results', type=str, help='save directory')
parser.add_argument('--dataset', '-d', default='cifar100', type=str, help='the name for dataset cifar100 | tinyimagenet | CUB200 | STANFORD120 | MIT67')

parser.add_argument('--epoch', default=200, type=int, help='total epochs to run')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--batch-size', '-n', default=128, type=int, help='batch size, N')
parser.add_argument('--intra-imgs', '-m', default=3, type=int, help='intra-class images, M')

parser.add_argument('--temp', default=4.0, type=float, help='temperature scaling')
parser.add_argument('--lamda', default=1.0, type=float, help='kd loss weight ratio')
parser.add_argument('--omega', default=0.5, type=float, help='ensembling weight')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
args.num_instances = args.intra_imgs + 1
args.batch_size = args.batch_size // args.num_instances

best_val = 0  # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

cudnn.benchmark = True

# Data
print('==> Preparing dataset: {}'.format(args.dataset))
trainloader, valloader = load_dataset(args.dataset, args.dataroot,
									batch_size=args.batch_size,
									num_instances=args.num_instances)

num_class = trainloader.dataset.num_classes
print('Number of train dataset: ' ,len(trainloader.dataset))
print('Number of validation dataset: ' ,len(valloader.dataset))

# Model
print('==> Building model: {}'.format(args.arch))
net = models.load_model(args.arch, num_class)

if use_cuda:
    torch.cuda.set_device(args.sgpu)
    net.cuda()
    print(torch.cuda.device_count())
    print('Using CUDA..')

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.sgpu, args.sgpu + args.ngpu)))

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

logdir = os.path.join(args.saveroot, args.dataset, args.arch, args.name)
set_logging_defaults(logdir, args)
logger = logging.getLogger('main')
logname = os.path.join(logdir, 'log.csv')

# Resume
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)

criterion = nn.CrossEntropyLoss()

class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, target)*(self.temp_factor**2)/input.size(0)
        return loss

kdloss = KDLoss(args.temp)
softmax = nn.Softmax(dim=1)

def knowledge_ensemble(feats, logits):
    batch_size = logits.size(0)
    masks = torch.eye(batch_size)
    if use_cuda:
        masks = masks.cuda()
    feats = nn.functional.normalize(feats, p=2, dim=1)
    logits = nn.functional.softmax(logits/args.temp, dim=1)
    W = torch.matmul(feats, feats.permute(1, 0)) - masks * 1e9
    W = softmax(W)
    W = (1 - args.omega) * torch.inverse(masks - args.omega * W)
    return torch.matmul(W, logits)

def train(epoch):
	print('\nEpoch: %d' % epoch)
	net.train()
	train_ce_loss = 0
	correct = 0
	total = 0
	train_kd_loss = 0
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()

		batch_size = inputs.size(0)

		features, outputs = net(inputs)
		loss = criterion(outputs, targets)
		train_ce_loss += loss.item()

		############
		with torch.no_grad():
			kd_targets = knowledge_ensemble(features.detach(), outputs.detach())
		kd_loss = kdloss(outputs, kd_targets.detach())
		loss += args.lamda * kd_loss
		train_kd_loss += kd_loss.item()
		############

		_, predicted = torch.max(outputs, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).sum().float().cpu()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		progress_bar(batch_idx, len(trainloader),
			'CE loss: %.3f | KD loss: %.3f | Acc: %.3f%% (%d/%d)'
		         % (train_ce_loss/(batch_idx+1), train_kd_loss/(batch_idx+1), 100.*correct/total, correct, total))

	logger = logging.getLogger('train')
	logger.info('[Epoch {}] [CE loss {:.3f}] [KD loss {:.3f}] [Acc {:.3f}]'.format(
				epoch,
				train_ce_loss/(batch_idx+1),
				train_kd_loss/(batch_idx+1),
				100.*correct/total))

	return 100.*correct/total

def val(epoch):
    global best_val
    net.eval()
    val_loss = 0.0
    correct = 0.0
    total = 0.0

    # Define a data loader for evaluating
    loader = valloader

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            _, outputs = net(inputs)
            loss = torch.mean(criterion(outputs, targets))

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().float()

            progress_bar(batch_idx, len(loader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d) '
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    if acc > best_val:
        best_val = acc
        checkpoint(acc, epoch)
    logger = logging.getLogger('val')
    logger.info('[Epoch {}] [Loss {:.3f}] [Acc {:.3f}] [Best Acc {:.3f}]'.format(
        epoch,
        val_loss/(batch_idx+1),
        acc, best_val))

    return (val_loss/(batch_idx+1), acc)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    torch.save(state, os.path.join(logdir, 'ckpt.t7'))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 0.5 * args.epoch:
        lr /= 10
    if epoch >= 0.75 * args.epoch:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if (not args.eval):
	# Logs
	for epoch in range(start_epoch, args.epoch):
	    train_acc = train(epoch)
	    val_loss, val_acc = val(epoch)
	    adjust_learning_rate(optimizer, epoch)
else:
	val_loss, val_acc = val(0)

print("Best Accuracy : {}".format(best_val))
logger = logging.getLogger('best')
logger.info('[Acc {:.3f}]'.format(best_val))
