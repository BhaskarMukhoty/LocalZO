import argparse
import shutil
import os
import time
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from models.VGG_models import *
import data_loaders
from functions import TET_loss, seed_all, get_logger
from models.resnet_models import resnet19
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Using {device}')


parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('-j',
                    '--workers',
                    default=16,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs',
                    default=150,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch_size',
                    default=128,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning_rate',
                    default=0.001,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--seed',
                    default=1000,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--T',
                    '--time',
                    default=2,
                    type=int,
                    metavar='N',
                    help='snn simulation time (default: 2)')
parser.add_argument('--means',
                    default=1.0,
                    type=float,
                    metavar='N',
                    help='make all the potential increment around the means (default: 1.0)')
parser.add_argument('--TET',
                    default=0,
                    type=int,
                    metavar='N',
                    help='if use Temporal Efficient Training (default: 0)')
parser.add_argument('--lamb',
                    default=0.05,
                    type=float,
                    metavar='N',
                    help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('--cut',
                    default=1,
                    type=int,
                    help='Cutout data augmentation')
parser.add_argument('--dataset',
                    default='cifar10',
                    type=str,
                    help='cifar10, cifar100')
parser.add_argument('--resume',
                    default=0,
                    type=int)
args = parser.parse_args()

print(args)
def train(model, device, train_loader, criterion, optimizer, epoch, args):
    running_loss = 0
    start_time = time.time()
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        outputs = model(images)
        mean_out = outputs.mean(1)
        if args.TET!=0:
            loss = TET_loss(outputs,labels,criterion,args.means,args.lamb)
        else:
            loss = criterion(mean_out,labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = mean_out.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total

@torch.no_grad()
def test(model, test_loader, device, lat):
    correct = 0
    total = 0
    model.eval()
    model.module.T=lat
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        mean_out = outputs.mean(1)
        _, predicted = mean_out.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    final_acc = 100 * correct / total
    return final_acc


if __name__ == '__main__':
    seed_all(args.seed)
    #train_dataset, val_dataset = data_loaders.build_dvscifar('cifar-dvs')
    if args.dataset == 'cifar10':
        train_dataset, val_dataset = data_loaders.build_cifar(cutout=args.cut,use_cifar10=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset, val_dataset = data_loaders.build_cifar(cutout=args.cut,use_cifar10=False)
        num_classes = 100
    elif args.dataset == 'dvscifar':
        train_dataset, val_dataset = data_loaders.build_dvscifar()
        num_classes = 10
                
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)

    #model = VGGSNNwoAP()
    model = resnet19(num_classes = num_classes)
    print(model)



    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    best_acc = 0
    folder_name =f'outputs/{args.dataset}/{args.TET}{args.cut}/' 
    os.makedirs(folder_name, exist_ok=True)
    model_name = folder_name+f'{args.dataset}_resnet19_T{args.T}_TET{args.TET}_seed{args.seed}_cut{args.cut}_batch{args.batch_size}'

    s_epoch=0
    if args.resume == 1:
        path = model_name+'_last_checkpoint.pth'
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        parallel_model = torch.nn.DataParallel(model)
        parallel_model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        s_epoch = checkpoint['epoch']
        loss = checkpoint['loss']    
        model.train()
    else:
        parallel_model = torch.nn.DataParallel(model)
        parallel_model.to(device)  

    
    for epoch in range(s_epoch, args.epochs):      
        t1 = time.time()
        parallel_model.module.T = args.T
        loss, acc = train(parallel_model, device, train_loader, criterion, optimizer, epoch, args)
        t2 = time.time()
        print('Time elapsed: ', t2 - t1)
        print('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , args.epochs, loss, acc ))
        scheduler.step()
        
        acc = test(parallel_model, test_loader, device, args.T)
        print(f'Epoch:[{epoch}/{args.epochs}]\t Test acc={acc:.3f}')

        if best_acc < acc: 
            best_acc = acc
            save_names = model_name + f'_test{args.T}.pth' 
            torch.save(parallel_model.module.state_dict(), save_names)
            print(f'Best Test acc={best_acc:.3f}')
            print('\n')        
        
        path_name = model_name + '_last_checkpoint.pth'
        torch.save({
            'epoch': epoch+1,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),            
            }, path_name)
        

