# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from models import *
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT
from models.convmixer import ConvMixer



def kl_div(p_logit, q_logit, T):
    p = F.softmax(p_logit / T, dim=-1)
    kl = torch.sum(p * (F.log_softmax(p_logit / T, dim=-1)
                                  - F.log_softmax(q_logit / T, dim=-1)), 1)
    return torch.mean(kl)


def kl_div_mask(p_logit, q_logit, T, mask, s_ctrl=0.5):
    p = F.softmax(p_logit / T, dim=-1)
    kl = ( 1 - (1-mask) + s_ctrl*(1-mask) ) * ( torch.sum(p * (F.log_softmax(p_logit / T, dim=-1)
                                  - F.log_softmax(q_logit / T, dim=-1)), 1) )
    return torch.mean(kl)


def dist_s_label(y, q):
    q = F.softmax(q, dim=-1)
    dist = torch.sum(torch.abs(q - y), 1)

    return torch.mean(dist)


def dist_s_t(p_logit, q_logit, T):
    p = F.softmax(p_logit / T, dim=-1)
    q = F.softmax(q_logit / T, dim=-1)
    dist = torch.sum(torch.abs(q - p), 1)

    return torch.mean(dist)

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")

args = parser.parse_args()

# take in args
#usewandb = ~args.nowandb
usewandb = False
if usewandb:
    import wandb
    watermark = "{}_lr{}".format(args.net, args.lr)
    wandb.init(project="cifar10-challange",
            name=watermark)
    wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_acc_s = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
])

# Add RandAugment with N, M(hyperparameter)
if aug:  
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

# Prepare dataset
trainset = torchvision.datasets.CIFAR100(root='.../datasets/CIFAR100', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR100(root='.../datasets/CIFAR100', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model factory..
print('==> Building model..')



from models.swin import swin_t
net_t = ResNet18()

from models.swin import swin_t
net_s = swin_t(window_size=args.patch,
                num_classes=100,
                downscaling_factors=(2,2,2,1))




# For Multi-GPU
if 'cuda' in device:
    print(device)
    if args.dp:
        print("using data parallel")
        net_t = torch.nn.DataParallel(net_t) # make parallel
        net_s = torch.nn.DataParallel(net_s)
        cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer_t = optim.Adam(net_t.parameters(), lr=args.lr)
    optimizer_s = optim.Adam(net_s.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer_t = optim.SGD(net_t.parameters(), lr=args.lr) 
    optimizer_s = optim.SGD(net_s.parameters(), lr=args.lr)   
    
# use cosine scheduling
scheduler_t = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_t, args.n_epochs)
scheduler_s = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s, args.n_epochs)

##### Training
scaler_t = torch.cuda.amp.GradScaler(enabled=use_amp)
scaler_s = torch.cuda.amp.GradScaler(enabled=use_amp)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net_t.train()
    net_s.train()

    flag = 0

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        target_onehot = (torch.zeros(inputs.size()[0], 100).to(device)).scatter_(1, targets.view(targets.size()[0], 1).to(device), 1)

        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):

            optimizer_t.zero_grad()
            outputs_t = net_t(inputs)

            optimizer_s.zero_grad()
            outputs_s = net_s(inputs)


            mask_t = (outputs_t.detach().max(1)[1]==target_onehot.max(1)[1]).float()


            s_label = dist_s_label(target_onehot, outputs_s.detach())
            t_label = dist_s_label(target_onehot, outputs_t.detach())
            ps_pt = dist_s_t(outputs_t.detach(), outputs_s.detach(), 1)


            epsilon = torch.exp(- (2-t_label) * (t_label / (t_label + s_label)) )
            delta = s_label - epsilon * t_label
            
        

        if ps_pt > delta and t_label < s_label:
            flag = 1

            loss_s = criterion(outputs_s, targets) + \
                        2 * 2 * 2 * kl_div_mask(outputs_t.detach(), outputs_s, 2, mask_t, s_ctrl=t_label/2)

            scaler_s.scale(loss_s).backward()
            scaler_s.step(optimizer_s)
            scaler_s.update()


        else:
            flag = 0

            loss_t = criterion(outputs_t, targets) + \
                        kl_div(outputs_s.detach(), outputs_t, 1)

            loss_s = criterion(outputs_s, targets) + \
                        2 * kl_div(outputs_t.detach(), outputs_s, 1)


            scaler_t.scale(loss_t).backward()
            scaler_t.step(optimizer_t)
            scaler_t.update()
            

            scaler_s.scale(loss_s).backward()
            scaler_s.step(optimizer_s)
            scaler_s.update()
        

        train_loss += loss_s.item()

    return train_loss/(batch_idx+1)

##### Validation
def test_t(epoch):
    global best_acc
    net_t.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net_t(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 40 == 0:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net_t.state_dict(),
              "optimizer": optimizer_t.state_dict(),
              "scaler": scaler_t.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+'t'+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer_t.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc_t: {(acc):.5f}'
    print(content)
    print('best_acc_t:' + str(best_acc))
    with open(f'log/log_t_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

def test_s(epoch):
    global best_acc_s
    net_s.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net_s(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 40 == 0:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc_s:
        print('Saving..')
        state = {"model": net_s.state_dict(),
              "optimizer": optimizer_s.state_dict(),
              "scaler": scaler_s.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+'s'+'-{}-ckpt.t7'.format(args.patch))
        best_acc_s = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer_s.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc_s: {(acc):.5f}'
    print(content)
    print('best_acc_s:' + str(best_acc_s))
    with open(f'log/log_s_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc


'''
list_loss = []
list_acc = []
'''

if usewandb:
    wandb.watch(net)
    
net_t.cuda()
net_s.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss_t, acc_t = test_t(epoch)
    val_loss_s, acc_s = test_s(epoch)
    
    scheduler_t.step(epoch-1) # step cosine scheduling
    scheduler_s.step(epoch-1)
    

    '''
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # Log training..
    if usewandb:
        wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
        "epoch_time": time.time()-start})

    # Write out csv..
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    print(list_loss)
    '''

# writeout wandb
if usewandb:
    wandb.save("wandb_{}.h5".format(args.net))
    
