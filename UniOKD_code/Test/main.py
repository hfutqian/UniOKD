#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

from models.resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from models.resnetv2 import ResNet50
from models.wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from models.mobilenetv2 import mobile_half as MobileNetV2

from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

def save_state(model, best_acc, flag=''):
    print('==> Saving model ...')
    state = {
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            }
    for key in list(state['state_dict'].keys()):
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, './models/model_' + flag + '_best.pth.tar')


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



def ps_pt_y(y, q):

    q = F.softmax(q, dim=-1)
    dist = torch.sum(torch.abs(q - y), 1)

    return torch.mean(dist)


def ps_pt_dist(p_logit, q_logit, T):
    p = F.softmax(p_logit / T, dim=-1)
    q = F.softmax(q_logit / T, dim=-1)
    dist = torch.sum(torch.abs(q - p), 1)

    return torch.mean(dist)


def train(epoch):

    model_T.train()
    model_S.train()


    for batch_idx, (data, target) in enumerate(trainloader):

        target_onehot = Variable((torch.zeros(data.size()[0], 100).cuda()).scatter_(1, target.view(target.size()[0], 1).cuda(), 1))

        # forwarding
        data, target = Variable(data.cuda()), Variable(target.cuda())

        optimizer.zero_grad()
        output = model_T(data)

        optimizer_S.zero_grad()
        output_S = model_S(data)


        mask = (output.detach().max(1)[1]==target_onehot.max(1)[1]).float()


        ps_y = ps_pt_y(target_onehot, output_S.detach())
        pt_y = ps_pt_y(target_onehot, output.detach())
        #s_t = ps_pt(output.detach(), output_S.detach(), 1)

        #norm
        ps_pt = ps_pt_dist(output.detach(), output_S.detach(), 1)
        epsilon = torch.exp(- (2-pt_y) * (pt_y / (pt_y + ps_y)) )
        delta = ps_pt_y(target_onehot, output_S.detach()) - epsilon * ps_pt_y(target_onehot, output.detach())

        # backwarding
        if ps_pt > delta and pt_y < ps_y:

            loss_S = criterion(output_S, target) + \
                       2 * 4 * 4 * kl_div_mask(output.detach(), output_S, 4, mask, s_ctrl=pt_y/2)

            loss_S.backward()
            optimizer_S.step()

        else:
            loss = criterion(output, target) + \
                        2 * 2 * kl_div_mask(output_S.detach(), output, 2, mask, s_ctrl=1)

            loss_S = criterion(output_S, target) + \
                       2 * 2 * 2 * kl_div_mask(output.detach(), output_S, 2, mask, s_ctrl=1)

            loss.backward()
            loss_S.backward()

            optimizer.step()
            optimizer_S.step()



        if batch_idx % 100 == 0:
            print(epoch)
            print('T criterion:' + str(criterion(output, target).item()) )
            print('S criterion:' + str(criterion(output_S, target).item()) + ' kl_div:' + str(kl_div(output.detach(), output_S, 1).item()))


    #return


def test_T():
    global best_acc
    model_T.eval()
    test_loss = 0
    correct = 0

    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
                                    
        output = model_T(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    acc = 100. * correct / len(testloader.dataset)

    if acc > best_acc:
        best_acc = acc
        save_state(model_T, best_acc, flag='T')
    
    test_loss /= len(testloader.dataset)
    print('\nTest set: T: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))

    return acc



def test_S():
    global best_acc_S
    model_S.eval()
    test_loss = 0
    correct = 0

    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())

        output = model_S(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    acc = 100. * correct / len(testloader.dataset)

    if acc > best_acc_S:
        best_acc_S = acc
        save_state(model_S, best_acc_S, flag='S')

    test_loss /= len(testloader.dataset)
    print('\nTest set: S: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc_S))

    return acc



def adjust_learning_rate(optimizer, epoch):
    update_list = [140, 200, 240]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*0.1
    return

if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='Dataset path',
            help='dataset path')
    parser.add_argument('--arch', action='store', default='resnet',
            help='the architecture for the network: nin')
    parser.add_argument('--lr', action='store', default='0.05',
            help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', default=None,
            help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true',
            help='evaluate the model')
    args = parser.parse_args()
    print('==> Options:', args)

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)



    # prepare the data
    '''
    if not os.path.isfile(args.data+'/train_data'):
        # check the data path
        raise Exception\
                ('Please assign the correct data path with --data <DATA_PATH>')
    '''

    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    train_dataset = torchvision.datasets.CIFAR100(
        root=args.data,
        train=True,
        download=False,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR100(
        root=args.data,
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # define classes
    #classes = ('plane', 'car', 'bird', 'cat',
    #        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



    # define the model
    print('==> building model',args.arch,'...')
    if args.arch == 'resnet':
        model_T = wrn_40_2()
        model_S = wrn_16_2()
    else:
        raise Exception(args.arch+' is currently not supported')

    # initialize the model
    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
        '''
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                c = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, 1.0 / c)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data = m.weight.data.zero_().add(1.0)
        '''


        best_acc_S = 0
        '''
        for m in model_ks2.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                c = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, 1.0 / c)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data = m.weight.data.zero_().add(1.0)
        '''

    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained + str('model_T_best.pth.tar'))
        best_acc = pretrained_model['best_acc']
        model_T.load_state_dict(pretrained_model['state_dict'])

        pretrained_model_S = torch.load(args.pretrained + str('model_S_best.pth.tar'))
        best_acc_S = pretrained_model_S['best_acc']
        model_S.load_state_dict(pretrained_model_S['state_dict'])


    if not args.cpu:
        model_T.cuda()
        model_T = torch.nn.DataParallel(model_T, device_ids=range(torch.cuda.device_count()))

        model_S.cuda()
        model_S = torch.nn.DataParallel(model_S, device_ids=range(torch.cuda.device_count()))
    print("model_T==>",model_T)
    print("model_S==>",model_S)

    # define solver and criterion
    base_lr = float(args.lr)
    #full
    param_dict = dict(model_T.named_parameters())
    params = []
    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr,
            'weight_decay':5e-4}]

        optimizer = optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=5e-4)


    #ks2
    param_dict_S = dict(model_S.named_parameters())
    params_S = []
    for key, value in param_dict_S.items():
        params_S += [{'params': [value], 'lr': base_lr,
                    'weight_decay': 5e-4}]

        optimizer_S = optim.SGD(params_S, lr=base_lr, momentum=0.9, weight_decay=5e-4)




    criterion = nn.CrossEntropyLoss()



    # Evaluation

    test_T()
    test_S()



























