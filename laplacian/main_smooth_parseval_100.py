'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random 


import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable
from gsp import force_smooth_network

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()



use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
#    transforms.RandomCrop(32, padding=4),
#    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(root='/homes/crosarko/pytorch-cifar/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='/homes/crosarko/pytorch-cifar/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    net = PreActResNet18(classes=100)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0])
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
path = "results/CIFAR100/0.01_0.01_2/"
try:
    os.makedirs(path)        
except:
    pass

beta = 0.01
params = net.parameters()
parseval_parameters = list()
for param in params:
    if len(param.size()) > 1:
        parseval_parameters.append(param)

def do_parseval(parseval_parameters):
    for W in parseval_parameters:
        Ws = W.view(W.size(0),-1)
        W_partial = Ws.data.clone()
        W_partial = (1+beta)*W_partial - beta*(torch.mm(torch.mm(W_partial,torch.t(W_partial)),W_partial))
        new = W_partial 
        new = new.view(W.size())
        W.data.copy_(new)


# Training
def train(epoch, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss1 = 0
    train_loss2 = 0
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        relus, outputs = net(inputs)
#        relus = relus[4:]
        loss1 = criterion(outputs, targets) 
        loss2 = force_smooth_network(relus,targets,classes=100,m=2)
        loss = loss1 + loss2/(100**2)
        loss.backward()
        optimizer.step()
        do_parseval(parseval_parameters)

        train_loss += loss.data[0]
        train_loss1 += loss1.data[0]
        train_loss2 += loss2.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Log Loss: %.3f | Smooth Loss: %.3f | Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss1/(batch_idx+1),train_loss2/(batch_idx+1),train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    f = open(path + 'score_training.txt','a')
    f.write(str(1.*correct/total))
    f.write('\n')
    f.close()

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    test_loss1 = 0
    test_loss2 = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        relus, outputs = net(inputs)
#        relus = relus[4:]
        loss1 = criterion(outputs, targets) 
        loss2 = force_smooth_network(relus,targets,classes=100,m=2)
        loss = loss1 + loss2/(100**2)

        test_loss += loss.data[0]
        test_loss1 += loss1.data[0]
        test_loss2 += loss2.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Log Loss: %.3f | Smooth Loss: %.3f | Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss1/(batch_idx+1),test_loss2/(batch_idx+1),test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    f = open(path + 'score.txt','a')
    f.write(str(1.*correct/total))
    f.write('\n')
    f.close()
    """
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    """
def save(epoch):
    net.forward(examples, True, epoch)

def save_model():
    state = {
        'net': net.module if use_cuda else net,
    }
    torch.save(state, path+'/ckpt.t7')
    

f = open(path + 'score.txt','w')
f.write("0.1\n")
f.close()
f = open(path + 'score_training.txt','w')
f.write("0.1\n")
f.close()
    

for period in range(2):
    if period == 0:
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
                
    for epoch in range(50 * period, 50 * (period + 1)):
        train(epoch, optimizer)
        test(epoch)
save_model()
#save(epoch)

"""
epoch_start = [0,150,250]
epoch_end =  [150,250,350]    
for period in range(3):
    if period == 0:
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    elif period == 1:
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
     
    for epoch in range(epoch_start[period], epoch_end[period]):
        train(epoch, optimizer)
        test(epoch)
save_model()
"""
