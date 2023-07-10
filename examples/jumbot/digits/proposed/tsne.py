import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE

import torch.nn.functional as F
from models import Classifier2, weights_init, Cnn_generator

from jumbot_utils import *
#_____________________________
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--source_dset", required=True, type = str, help = "source dset")
parser.add_argument("--target_dset", required=True, type = str, help = "target dset")
args = parser.parse_args()
source_dset = args.source_dset
target_dset = args.target_dset
log = "MMDOT"

task = "{}2{}".format(source_dset, target_dset)
#_____________________________


logger_fname = f'[{log}]_{task}'

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

def feature_extraction(model, dataloader):
    embed_list = []
    label_list = []

    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device)
            embed = model(img)
            label_list.append(label)
            embed_list.append(embed)

    return torch.cat(embed_list).cpu().numpy(), torch.cat(label_list).cpu().numpy()

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

transform_target = get_transform(target_dset)

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

fig = plt.figure(figsize=(20, 5))
TICK_SIZE = 14
TITLE_SIZE = 20
MARKER_SIZE = 50
NUM_SAMPLES = 2000

ax = fig.add_subplot()
title = "Proposed"

# model_g.load_state_dict(torch.load(f"models_{source_dset}2{target_dset}/model_g.pt"))
model_g = torch.load(f"models_{source_dset}2{target_dset}/model_g.pt")

source_embed, source_label = feature_extraction(model_g, test_source_loader)
target_embed, target_label = feature_extraction(model_g, test_target_loader)

combined_imgs = np.vstack([source_embed[0:NUM_SAMPLES, :], target_embed[0:NUM_SAMPLES, :]])
combined_labels = np.concatenate([source_label[0:NUM_SAMPLES], target_label[0:NUM_SAMPLES]])
combined_labels = combined_labels.astype("int")
tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=3000)
source_only_tsne = tsne.fit_transform(combined_imgs)
ax.scatter(
    source_only_tsne[:NUM_SAMPLES, 0],
    source_only_tsne[:NUM_SAMPLES, 1],
    c=combined_labels[:NUM_SAMPLES],
    s=MARKER_SIZE,
    alpha=0.5,
    marker="o",
    cmap=cm.jet,
    label="source",
)
ax.scatter(
    source_only_tsne[NUM_SAMPLES:, 0],
    source_only_tsne[NUM_SAMPLES:, 1],
    c=combined_labels[NUM_SAMPLES:],
    s=MARKER_SIZE,
    alpha=0.5,
    marker="+",
    cmap=cm.jet,
    label="target",
)
ax.set_xlim(-125, 125)
ax.set_ylim(-125, 125)
ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
ax.set_title(title, fontsize=TITLE_SIZE)
ax.legend(loc="upper right")

plt.savefig(f"{source_dset}2{target_dset}.jpg")
plt.close()
