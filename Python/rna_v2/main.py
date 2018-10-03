'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function


print('### STARTING ###')

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

from models import *
from utils import progress_bar
from torch.autograd import Variable

import online_rna
from copy import deepcopy
import csv
from prettytable import PrettyTable



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--K', default=5, type=int)
parser.add_argument('--path_to_log', default=None, type=str)
parser.add_argument('--data_path', default='', type=str)
parser.add_argument('--arch', default='resnet18', type=str)
parser.add_argument('--max_epoch', default='100', type=int)
parser.add_argument('--stabilization', default='10', type=int)
parser.add_argument('--optimizer', default='orna_sgd', type=str)
parser.add_argument('--eval_train_error', default='n', type=str) # n or y
parser.add_argument('--do_average', default='n', type=str) # n or y
args = parser.parse_args()

if(args.data_path == ''):
    raise Exception('Argument data_path should not be empty.')

# To be fixed, later...
if(args.eval_train_error == 'n'):
    eval_train_error = False
if(args.eval_train_error == 'y'):
    eval_train_error = True

if(args.do_average == 'n'):
    do_average = False
if(args.do_average == 'y'):
    do_average = True

use_cuda = torch.cuda.is_available()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=50)

testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=10)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

arch = args.arch;
print('==> Building model..' + arch)

if(arch == 'VGG19'):
    net = VGG('VGG19')
if(arch == 'resnet18'):
    net = ResNet18()
if(arch == 'PreActResNet18'):
    net = PreActResNet18()
if(arch == 'googlenet'):
    net = GoogLeNet()
if(arch == 'densenet121'):
    net = DenseNet121()
if(arch == 'MobileNet'):
    net = MobileNet()
if(arch == 'MobileNetV2'):
    net = MobileNetV2()

args.path_to_log = args.path_to_log + arch + '/'


if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if( (batch_idx%20 == 0) and (batch_idx > 0)):
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('\n')
            
        
def test(epoch,model,loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        if( (batch_idx%20 == 0) and (batch_idx > 0)):
            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('\n')

    err_avg = 100.0-(100.*correct/total)
    loss_avg = test_loss*1.0/total
    output = (loss_avg,err_avg)
    return output
    
def full_test(epoch,model,loader_train,loader_val,test_train=True):
    if(test_train):
        train_stats = test(epoch,model,loader_train)
    else:
        train_stats=(0,0)
    val_stats = test(epoch,model,loader_val)
    output = train_stats + val_stats;
    return output
    



def write_log(logpath,logfile,stats):
    if(logpath is not None):
        filename = logpath+logfile+'.log'
        print('Writing log in : ' +filename)
        with open(filename,'a+') as f:
            to_write = '';
            for i in range(0,len(stats)):
                to_write = to_write + str(stats[i]) + ' '
            to_write = to_write + '\n'
            print('Writing to log : ' + to_write)
            f.write(to_write)
            f.close()

def print_stats(stats_net1,stats_net2,epoch):
    print(' ')
    print('#############     STATS (Epoch : '+ str(epoch) +')     ############')
    print(' ')
    t = PrettyTable(['Train loss', 'Train Err', 'Val Loss', 'Val Err'])
    t.add_row([ stats_net1[0], stats_net1[1], stats_net1[2], stats_net1[3]])
    t.add_row([ stats_net2[0], stats_net2[1], stats_net2[2], stats_net2[3]])
    print(t)
    print(' ')

def lr_scheduler(max_epoch,lr_0,lr_final,epoch):
    epoch_stabilization = args.stabilization;
    if(epoch<max_epoch-epoch_stabilization):
        lr = lr_final + (lr_0-lr_final)*(1-(1.0*epoch/(1.0*(max_epoch-epoch_stabilization-1))))
    else:
        lr = lr_final
    return lr;


weight_decay=1e-5

if(args.optimizer == 'sgd'):
    acceleration_type = 'none'
    lr_0 = 1
    lr_final = 0.01
    momentum = 0
    
if(args.optimizer == 'offline_rna_sgd'):
    acceleration_type = 'offline'
    lr_0 = 1
    lr_final = 0.01
    momentum = 0
    
if(args.optimizer == 'orna_sgd'):
    acceleration_type = 'online'
    lr_0 = 1
    lr_final = 0.01
    momentum = 0
    
if(args.optimizer == 'sgd_momentum'):
    acceleration_type = 'none'
    lr_0 = 0.1
    lr_final = 0.001
    momentum = 0.9
    
if(args.optimizer == 'offline_rna_sgd_momentum'):
    acceleration_type = 'offline'
    lr_0 = 0.1
    lr_final = 0.001
    momentum = 0.9
    
if(args.optimizer == 'orna_sgd_momentum'):
    acceleration_type = 'online'
    lr_0 = 0.1
    lr_final = 0.001
    momentum = 0.9

if(acceleration_type == 'offline'):
    net2 = deepcopy(net)
else:
    net2 = net
	
log_filename = arch + '_' + args.optimizer
if(do_average):
    log_filename = log_filename + '_average'
log_filename = log_filename + '_' + str(args.max_epoch)
    

new_lr = lr_scheduler(args.max_epoch,lr_0,lr_final,0)
optimizer = online_rna.online_rna(net.parameters(),lr=new_lr,momentum=momentum,weight_decay=weight_decay,nesterov=False,K=args.K,reg_acc=0,acceleration_type=acceleration_type,do_average=do_average)


test_train = eval_train_error; # compute loss and accuracy on the train set

for epoch in range(0, args.max_epoch ):
    
    # Schedule
    new_lr = lr_scheduler(args.max_epoch,lr_0,lr_final,epoch)
    optimizer.update_lr(new_lr)
    
    print('\n\n########### EPOCH: ' +str(epoch) + ' WITH LEARNING RATE: ' +str(new_lr)+'\n\n')
    train(epoch)
    stats_net1 = full_test(epoch,net,trainloader,testloader,test_train=test_train)
    optimizer.store(model=net)
    if(epoch == 1):
        optimizer.reset_buffers()
    if(epoch > 1):
        c = optimizer.accelerate(net2)
        print("\n ACCELERATION \n")
        print(c)
        print("\n")
		
    stats_net2 = full_test(epoch,net2,trainloader,testloader,test_train=test_train)
	
    stats_full = stats_net1+stats_net2 # cat all data
    print_stats(stats_net1,stats_net2,epoch)
    write_log(args.path_to_log, log_filename,stats_full)
    
    
    
    
