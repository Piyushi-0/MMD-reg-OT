import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn.functional as F
from models import Classifier2, weights_init, Cnn_generator

from jumbot_utils import *
from jumbot import Jumbot
import logging, os, yaml
#_____________________________
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--source_dset", required=True, type = str, help = "source dset")
parser.add_argument("--target_dset", required=True, type = str, help = "target dset")
parser.add_argument("--log", type=str, default="KLUOT")
args = parser.parse_args()

source_dset = args.source_dset
target_dset = args.target_dset

task = "{}2{}".format(source_dset, target_dset)
#_____________________________


logger_fname = f'[{args.log}]_{task}'

def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    import os
    os.environ['main_phd'] = str(seed)


batch_size = 500
nclass = 10

set_seed(1980)

# pre-processing to tensor, and mean subtraction

def get_transform(dset):
    if dset == "usps":
        transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    elif dset == "mnist":
        transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    elif dset == "mmnist":
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif dset == "svhn":
        transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    return transform

def get_dset(train, dset, transform):
    if dset == "usps":
        tset = datasets.USPS('../data', train = train, download=True,
                                transform= transform)
    elif dset == "mnist":
        tset = datasets.MNIST('../data', train=train, download=True,
                                transform=transform)
    elif dset == "mmnist":
        import mmnist
        tset =  mmnist.MNISTM("../mnistm", train=train, download=True, 
                                transform=transform)
    elif dset == "svhn":
        if train:
            tset =  datasets.SVHN('../data', split='train', download=True,
                                transform=transform)
        else:
            tset =  datasets.SVHN('../data', split='test', download=True,
                                transform=transform)
    return tset

transform_source = get_transform(source_dset)

train_source_trainset = get_dset(True, source_dset, transform_source)

# print('nb source data : ', len(train_source_trainset))

source_data = torch.zeros((len(train_source_trainset), 3, 32, 32))
source_labels = torch.zeros((len(train_source_trainset)))

for i, data in enumerate(train_source_trainset):
    source_data[i] = data[0]
    source_labels[i] = data[1]

train_batch_sampler = BalancedBatchSampler(source_labels, batch_size=batch_size)
train_source_loader = torch.utils.data.DataLoader(train_source_trainset, batch_sampler=train_batch_sampler)

transform_target = get_transform(target_dset)

train_target_trainset = get_dset(True, target_dset, transform_target)

train_target_loader = torch.utils.data.DataLoader(train_target_trainset, batch_size=batch_size, shuffle=True)

### TEST sets
test_source_loader = torch.utils.data.DataLoader(get_dset(False, source_dset, transform_source), batch_size=batch_size, shuffle=False)

test_target_loader = torch.utils.data.DataLoader(get_dset(False, target_dset, transform_target), batch_size=batch_size, shuffle=False)

    
####### Main

model_g = Cnn_generator().cuda().apply(weights_init)
model_f = Classifier2(nclass=nclass).cuda().apply(weights_init)

eta1 = 0.1
eta2 = 0.1
tau = 1.0
epsilon = 0.1

model_g.train()
model_f.train()

save_as = f"models_{task}"
os.makedirs(save_as, exist_ok=1)

jumbot = Jumbot(save_as, model_g, model_f, n_class=nclass, eta1=eta1, eta2=eta2, tau=tau, epsilon=epsilon)
loss = jumbot.source_only(train_source_loader)
loss = jumbot.fit(train_source_loader, train_target_loader, test_target_loader, n_epochs=100)

source_acc = jumbot.evaluate(test_source_loader)
target_acc = jumbot.evaluate(test_target_loader)
print ("Method = {}, Task = {}, target_acc = {}".format(args.log, task, target_acc))
