import argparse
import csv
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
from functions import TET_loss, seed_all
#from main_training_parallel import train, test
from models.resnet_models import resnet19
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
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
                    default=1e-3,
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

args = parser.parse_args()
print(args)

# Data loading code
if args.dataset == 'cifar10':
    train_dataset, val_dataset = data_loaders.build_cifar(cutout=args.cut,use_cifar10=True)
    num_classes = 10
elif args.dataset == 'cifar100':
    train_dataset, val_dataset = data_loaders.build_cifar(cutout=args.cut,use_cifar10=False)
    num_classes = 100

# train_dataset, val_dataset = data_loaders.build_dvscifar()
#train_sampler = torch.utils.data.distributed.DistributedSampler(
#    train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.workers,
                                            pin_memory=True)
#val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.workers,
                                            pin_memory=True)
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        #return res
        return correct


def validate(val_loader, model, criterion):
   
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        correct = 0 
        total = 0
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            mean_out = torch.mean(output, dim=1)
            loss = criterion(mean_out, target)
            _, predicted = torch.max(mean_out.data, 1)
            # measure accuracy and record loss
            correct += (predicted == target).sum().item()
            total += images.shape[0]
            #torch.distributed.barrier()
        return 100*correct/total
        #print("accuracy ",100*correct/total)
          

        # TODO: this should also be done with the ProgressMeter
        



if __name__ == '__main__':
    seed_all(args.seed)
    #train_dataset, val_dataset = data_loaders.build_dvscifar('cifar-dvs') # change to your path
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
    #                                            num_workers=args.workers, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
    #                                           shuffle=False, num_workers=args.workers, pin_memory=True)

    #model = VGGSNNwoAP()
    
    model = resnet19(num_classes=num_classes)    
    model_name = f'outputs/{args.dataset}/{args.TET}{args.cut}/{args.dataset}_resnet19_T{args.T}_TET{args.TET}_seed{args.seed}_cut{args.cut}_batch{args.batch_size}'
    file_name = f'outputs/{args.dataset}/{args.TET}{args.cut}/out_T{args.T}_TET_{args.TET}_Cut_{args.cut}_batch{args.batch_size}'


 
    model_path = model_name + f'_test{args.T}.pth'
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)
    f_path = file_name+f'_test{args.T}.csv'    
    row=[]    
    for t in range(1,args.T+1):
        model.T = t
        parallel_model = torch.nn.DataParallel(model)
        parallel_model.to(device)
        criterion = nn.CrossEntropyLoss().cuda()
        acc = validate(val_loader, parallel_model, criterion)
        row.append(acc)        
    with open(f_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    path_name = model_name + '_last_checkpoint.pth'
    checkpoint = torch.load(path_name)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    epoch = checkpoint['epoch']
    f_path = file_name+f'_epoch{epoch}.csv'
    row=[]    
    for t in range(1,args.T+1):
        model.T = t
        parallel_model = torch.nn.DataParallel(model)
        parallel_model.to(device)
        criterion = nn.CrossEntropyLoss().cuda()
        acc = validate(val_loader, parallel_model, criterion)
        row.append(acc)
    with open(f_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    
